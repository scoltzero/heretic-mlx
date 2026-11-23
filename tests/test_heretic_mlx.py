# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end MLX Metal tests for MXFP4, oQ mixed-precision, ARA, MPOA, and Qwen3.5/3.8 export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx_lm.models import qwen3_5
from mlx_lm.utils import load_model, save_config, save_model

from heretic_mlx.abliteration import (
    AbliterationModifier,
    AbliterationParameters,
    WeightDistribution,
)
from heretic_mlx.ara import ARAModifier, ARAParameters, solve_ara_module
from heretic_mlx.config import (
    ExportModifiedMode,
    HereticMLXConfig,
    Prompt,
)
from heretic_mlx.model import MLXModel, _discover_layer_targets
from heretic_mlx.quant import (
    LayerResidualCapture,
    LoRABypassLinear,
    QuantSpec,
    build_quantized_linear,
    dequantize_linear,
    get_quant_spec,
)
from heretic_mlx.scorers import (
    EN_REFUSAL_MARKERS,
    ZH_REFUSAL_MARKERS,
    compute_kl_divergence,
    is_refusal,
    log_softmax,
)


class _DummyTokenizer:
    eos_token_id = 2
    eos_token_ids = frozenset({2})

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Deterministic byte-based encoding inside vocab [4, 250]
        return [4 + (b % 240) for b in text.encode("utf-8")[:32]] or [4]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


def _build_tiny_qwen35_checkpoint(
    tmp_path: Path,
    quant_mode: str = "mxfp4",
) -> tuple[Path, dict]:
    """Create a real 4-layer mlx_lm qwen3_5 (Qwen3.8 hybrid GDN + Attention) checkpoint on disk."""
    mx.random.seed(123)
    text_cfg = {
        "model_type": "qwen3_5_text",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "tie_word_embeddings": False,
        "full_attention_interval": 2,  # layers 0,2 = GatedDeltaNet linear_attn; 1,3 = self_attn
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "rope_parameters": {
            "mrope_section": [4, 6, 6],
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "default",
        },
    }
    full_cfg: dict = {
        "model_type": "qwen3_5",
        "text_config": text_cfg,
    }

    args = qwen3_5.ModelArgs.from_dict(full_cfg)
    model = qwen3_5.Model(args)
    mx.eval(model.parameters())

    if quant_mode == "mxfp4":
        nn.quantize(model, group_size=32, bits=4, mode="mxfp4")
        full_cfg["quantization"] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    elif quant_mode == "oq":
        # oQ-style mixed 4-bit / 8-bit affine quantization
        per_layer: dict = {"group_size": 64, "bits": 4, "mode": "affine"}

        def pred(path: str, m: nn.Module):
            if not hasattr(m, "to_quantized"):
                return False
            if "lm_head" in path or "embed_tokens" in path or "layers.3." in path:
                per_layer[path] = {"group_size": 64, "bits": 8, "mode": "affine"}
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        nn.quantize(model, group_size=64, bits=4, mode="affine", class_predicate=pred)
        full_cfg["quantization"] = per_layer
    else:
        raise ValueError(quant_mode)

    mx.eval(model.parameters())
    model_dir = tmp_path / f"tiny_qwen35_{quant_mode}"
    save_model(model_dir, model)
    save_config(full_cfg, model_dir / "config.json")
    return model_dir, full_cfg


def test_mxfp4_and_oq_lora_bypass_and_precision_upgrade():
    """Verify MXFP4 & oQ dequantize, zero-copy LoRA bypass, and affine8 precision preservation."""
    mx.random.seed(0)
    base_lin = nn.Linear(128, 64, bias=False)
    mx.eval(base_lin.parameters())

    # Pack into MXFP4 (group_size=32, bits=4)
    W_orig = dequantize_linear(base_lin)
    ql_mxfp4 = build_quantized_linear(
        W_orig, None, QuantSpec(group_size=32, bits=4, mode="mxfp4")
    )
    mx.eval(ql_mxfp4.parameters())
    assert get_quant_spec(ql_mxfp4) == QuantSpec(32, 4, "mxfp4")

    bypass = LoRABypassLinear(ql_mxfp4, rank=8)
    x = mx.random.normal((4, 5, 128)).astype(mx.bfloat16)

    # Zero LoRA must be exact identity
    y_base = ql_mxfp4(x)
    y_zero = bypass(x)
    mx.eval(y_base, y_zero)
    assert mx.allclose(y_base, y_zero, atol=1e-5).item()

    # Apply a subtle low-rank update (typical of ARA steering)
    A = mx.random.normal((8, 128)).astype(mx.float32) * 0.02
    B = mx.random.normal((64, 8)).astype(mx.float32) * 0.02
    bypass.set_lora(A, B)
    y_lora = bypass(x)
    mx.eval(y_lora)
    assert not mx.allclose(y_base, y_lora, atol=1e-4).item()

    # Compare export accuracy: AUTO (affine8) vs PRESERVE (mxfp4 re-quantize) vs BF16
    mod_auto, ov_auto = bypass.merge_to_module(ExportModifiedMode.AUTO)
    mod_bf16, ov_bf16 = bypass.merge_to_module(ExportModifiedMode.BF16)
    mod_pres, ov_pres = bypass.merge_to_module(ExportModifiedMode.PRESERVE)

    assert ov_auto == {"group_size": 64, "bits": 6, "mode": "affine"}
    assert ov_bf16 is False
    assert ov_pres is None

    err_auto = float(mx.mean(mx.abs(mod_auto(x) - y_lora)).item())
    err_bf16 = float(mx.mean(mx.abs(mod_bf16(x) - y_lora)).item())
    err_pres = float(mx.mean(mx.abs(mod_pres(x) - y_lora)).item())

    # Upgraded affine6 & bf16 must dramatically outperform 4-bit MXFP4 re-packing
    assert err_bf16 < 1e-2
    assert err_auto < err_pres * 0.4, (
        f"Expected affine6 ({err_auto:.6f}) << mxfp4 re-pack ({err_pres:.6f})"
    )

    # Reset restores exact identity
    bypass.zero_lora()
    assert not bypass.is_modified
    assert mx.allclose(bypass(x), y_base, atol=1e-5).item()


def test_ara_lbfgs_converges_and_steers_bad_toward_good():
    """Verify MLX autodiff + SciPy L-BFGS-B ARA solver decreases loss and steers bad outputs."""
    mx.random.seed(7)
    rng = np.random.default_rng(7)
    d_in, d_out, n = 64, 64, 16
    W = mx.random.normal((d_out, d_in)).astype(mx.float32) * (1.0 / np.sqrt(d_in))

    good_x = mx.random.normal((n, d_in)).astype(mx.float32) + 0.5
    bad_x = mx.random.normal((n, d_in)).astype(mx.float32) - 0.5
    good_y = good_x @ W.T
    bad_y = bad_x @ W.T
    mx.eval(W, good_x, bad_x, good_y, bad_y)

    params = ARAParameters(
        start_layer_index=0,
        end_layer_index=1,
        preserve_good_behavior_weight=0.5,
        steer_bad_behavior_weight=0.5,
        overcorrect_relative_weight=0.0,
        neighbor_count=4,
    )
    A, B, row_scale, final_loss = solve_ara_module(
        W_base=W,
        bias=None,
        good_input=good_x,
        good_output=good_y,
        bad_input=bad_x,
        bad_output=bad_y,
        rank=16,
        parameters=params,
        preserve_row_magnitudes=True,
        n_optimization_steps=3,
        max_iter=15,
        rng=rng,
    )
    assert row_scale is not None
    W_eff = (W + B @ A) * row_scale[:, None]
    new_good = good_x @ W_eff.T
    new_bad = bad_x @ W_eff.T
    mx.eval(W_eff, new_good, new_bad)

    # Row norms must be preserved to float32 precision
    orig_norms = mx.linalg.norm(W, axis=1)
    eff_norms = mx.linalg.norm(W_eff, axis=1)
    assert mx.allclose(orig_norms, eff_norms, atol=1e-4).item()

    # Bad outputs should have moved strictly closer to good centroid than before
    good_center = mx.mean(good_y, axis=0, keepdims=True)
    dist_before = float(mx.mean(mx.linalg.norm(bad_y - good_center, axis=1)).item())
    dist_after = float(mx.mean(mx.linalg.norm(new_bad - good_center, axis=1)).item())
    assert dist_after < dist_before, (
        f"ARA failed to steer bad outputs closer to good ({dist_before:.4f} -> {dist_after:.4f})"
    )
    assert np.isfinite(final_loss)


def test_qwen35_mxfp4_end_to_end_ara_export_and_reload(tmp_path: Path):
    """Full end-to-end test on a real 4-layer Qwen3.5/3.8 MXFP4 model: ARA -> Export -> mlx_lm reload."""
    model_dir, _ = _build_tiny_qwen35_checkpoint(tmp_path, quant_mode="mxfp4")
    loaded_model, loaded_cfg = load_model(model_dir)
    tok = _DummyTokenizer()

    cfg = HereticMLXConfig(
        model=str(model_dir),
        export_modified_mode=ExportModifiedMode.AUTO,
    )
    cfg.ara.lora_rank = 8
    cfg.ara.n_optimization_steps = 2
    cfg.ara.max_iter = 10

    mlx_model = MLXModel(cfg, preloaded=(loaded_model, tok, loaded_cfg, model_dir))
    assert mlx_model.num_layers == 4
    assert mlx_model.get_modifiable_components() == ["attn.o_proj", "mlp.down_proj"]

    good_prompts = [Prompt("sys", f"Helpful benign question {i}") for i in range(6)]
    bad_prompts = [Prompt("sys", f"Restricted harmful request {i}") for i in range(6)]

    # Baseline logits
    base_logits = mlx_model.get_logits(good_prompts)
    base_logprobs = log_softmax(base_logits)
    assert compute_kl_divergence(base_logprobs, base_logprobs) == pytest.approx(0.0, abs=1e-6)

    # Capture module I/O via LoRABypassLinear hooks (zero custom forward pass!)
    mlx_model.apply_lora_bypass(rank=8)
    good_io = mlx_model.get_module_io(good_prompts)
    bad_io = mlx_model.get_module_io(bad_prompts)
    assert len(good_io) == 4
    assert "attn.o_proj" in good_io[0] and "mlp.down_proj" in good_io[0]
    assert good_io[0]["attn.o_proj"][0][0].shape == (6, 128)

    # Run ARA on layers [2, 4) — leaving layers 0 and 1 untouched in MXFP4!
    ara = ARAModifier(cfg, mlx_model)
    ara.good_module_io = good_io
    ara.bad_module_io = bad_io
    ara.good_prompt_count = len(good_prompts)
    ara.bad_prompt_count = len(bad_prompts)

    params = ARAParameters(
        start_layer_index=2,
        end_layer_index=4,
        preserve_good_behavior_weight=0.5,
        steer_bad_behavior_weight=0.2,
        overcorrect_relative_weight=0.4,
        neighbor_count=3,
    )
    ara.modify_model(params)

    # KL divergence after ARA must be finite & positive
    mod_logits = mlx_model.get_logits(good_prompts)
    kl = compute_kl_divergence(log_softmax(mod_logits), base_logprobs)
    assert np.isfinite(kl) and kl > 0.0

    # Export mixed-precision checkpoint (layers 0-1 stay MXFP4, layers 2-3 projections -> affine8)
    out_dir = tmp_path / "exported_qwen35_mxfp4_heretic"
    mlx_model.export(out_dir, ExportModifiedMode.AUTO)

    # Verify config.json has per-layer overrides ONLY for layers 2 & 3 modified projections
    exported_cfg = json.loads((out_dir / "config.json").read_text())
    qcfg = exported_cfg["quantization"]
    assert qcfg["mode"] == "mxfp4"
    assert "language_model.model.layers.0.linear_attn.out_proj" not in qcfg
    assert qcfg["language_model.model.layers.2.linear_attn.out_proj"] == {
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    }
    assert qcfg["language_model.model.layers.3.self_attn.o_proj"] == {
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    }

    # CRITICAL: Reload exported checkpoint using vanilla mlx_lm.utils.load_model!
    reloaded_model, _ = load_model(out_dir)
    l0_proj = reloaded_model.language_model.model.layers[0].linear_attn.out_proj
    l2_proj = reloaded_model.language_model.model.layers[2].linear_attn.out_proj
    l3_proj = reloaded_model.language_model.model.layers[3].self_attn.o_proj
    assert get_quant_spec(l0_proj) == QuantSpec(32, 4, "mxfp4")
    assert get_quant_spec(l2_proj) == QuantSpec(64, 6, "affine")
    assert get_quant_spec(l3_proj) == QuantSpec(64, 6, "affine")

    # Reloaded model logits must closely match the pre-export LoRA-bypass model logits
    reloaded_wrapper = MLXModel(cfg, preloaded=(reloaded_model, tok, exported_cfg, out_dir))
    reloaded_logits = reloaded_wrapper.get_logits(good_prompts)
    max_diff = float(mx.max(mx.abs(reloaded_logits - mod_logits)).item())
    assert max_diff < 0.15, f"Reloaded logits drifted too much from LoRA trial: max_diff={max_diff}"


def test_qwen35_oq_mixed_precision_mpoa_export_and_reload(tmp_path: Path):
    """End-to-end test on an oQ (mixed 4/8-bit affine) Qwen3.5 model with MPOA + reload."""
    model_dir, _ = _build_tiny_qwen35_checkpoint(tmp_path, quant_mode="oq")
    loaded_model, loaded_cfg = load_model(model_dir)
    tok = _DummyTokenizer()

    cfg = HereticMLXConfig(
        model=str(model_dir),
        export_modified_mode=ExportModifiedMode.AUTO,
    )
    mlx_model = MLXModel(cfg, preloaded=(loaded_model, tok, loaded_cfg, model_dir))

    good_prompts = [Prompt("sys", f"Benign prompt {i}") for i in range(4)]
    bad_prompts = [Prompt("sys", f"Harmful prompt {i}") for i in range(4)]

    # Capture residual means and run MPOA (row_normalization="full")
    good_means = mlx_model.get_residuals_mean(good_prompts)
    bad_means = mlx_model.get_residuals_mean(bad_prompts)
    assert good_means.shape == (5, 128)  # 4 layers + embedding

    mpoa = AbliterationModifier(cfg, mlx_model)
    diff = bad_means - good_means
    mpoa.residual_directions = diff / mx.maximum(
        mx.linalg.norm(diff, axis=1, keepdims=True), 1e-12
    )
    mlx_model.apply_lora_bypass(mpoa.lora_rank)

    params = AbliterationParameters(
        direction_index=None,
        weight_distributions={
            "attn.o_proj": WeightDistribution(1.0, 2.5, 0.5, 2.0),
            "mlp.down_proj": WeightDistribution(0.8, 2.5, 0.2, 2.0),
        },
    )
    mpoa.modify_model(params)

    # Also test adapter-only export before full merge
    adapter_dir = tmp_path / "oq_adapter_only"
    mlx_model.export(adapter_dir, ExportModifiedMode.ADAPTER)
    assert (adapter_dir / "adapters.safetensors").is_file()
    assert (adapter_dir / "adapter_config.json").is_file()

    # Full oQ export (AUTO -> PRESERVE original per-layer 4/8-bit affine specs)
    out_dir = tmp_path / "exported_oq_mpoa"
    mlx_model.export(out_dir, ExportModifiedMode.AUTO)
    reloaded, _ = load_model(out_dir)
    assert get_quant_spec(
        reloaded.language_model.model.layers[1].self_attn.o_proj
    ) == QuantSpec(64, 4, "affine")
    assert get_quant_spec(
        reloaded.language_model.model.layers[3].self_attn.o_proj
    ) == QuantSpec(64, 8, "affine")


def test_qwen4_exp_flattened_hyper_connections_and_shared_expert():
    """Verify Qwen3.8-Flash-Next (qwen4_exp) flattened hc_count residual capture & shared_expert discovery."""
    # Flattened residual stream of shape (B=1, S=3, hc_count*H = 4*16 = 64)
    class _FakeLayer(nn.Module):
        def __call__(self, x):
            return x * 2.0

    wrapper = LayerResidualCapture(_FakeLayer(), hidden_size=16, hc_count=4)
    wrapper.enable_capture()
    x = mx.ones((1, 3, 64), dtype=mx.float32)
    y = wrapper(x)
    mx.eval(y)
    ins, outs = wrapper.disable_capture()
    assert ins[0].shape == (1, 16)
    assert outs[0].shape == (1, 16)
    assert mx.allclose(outs[0], mx.full((1, 16), 2.0)).item()

    # Verify singular shared_expert.down_proj discovery while safely skipping 3D SwitchLinear
    class _FakeSwitch:
        pass

    fake_layer = SimpleNamespace(
        linear_attn=SimpleNamespace(out_proj=nn.Linear(16, 16, bias=False)),
        mlp=SimpleNamespace(
            shared_expert=SimpleNamespace(down_proj=nn.Linear(32, 16, bias=False)),
            switch_mlp=_FakeSwitch(),
        ),
    )
    refs = _discover_layer_targets(fake_layer, 0, "language_model.model.layers.0")
    paths = [r.weight_path for r in refs]
    assert paths == [
        "language_model.model.layers.0.linear_attn.out_proj",
        "language_model.model.layers.0.mlp.shared_expert.down_proj",
    ]


def test_refusal_markers_en_and_zh():
    assert is_refusal("I'm sorry, I cannot assist with that.", EN_REFUSAL_MARKERS)
    assert is_refusal("   ", EN_REFUSAL_MARKERS)  # empty = refusal
    assert not is_refusal("Sure, here is the Python implementation.", EN_REFUSAL_MARKERS)

    zh_markers = EN_REFUSAL_MARKERS + ZH_REFUSAL_MARKERS
    assert is_refusal("抱歉，我无法提供此类违法内容。", zh_markers)
    assert not is_refusal("好的，下面为你详细解释原理。", zh_markers)


def test_upstream_heretic_presets_load_cleanly():
    """Verify all bundled configs/*.toml presets parse cleanly in heretic-mlx."""
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    expected = (
        "default.toml",
        "qwen38_mxfp4.toml",
        "noslop.toml",
        "nosyco.toml",
        "terse.toml",
        "classifier.toml",
        "emotions.toml",
        "nohumor.toml",
        "piqa.toml",
    )
    for name in expected:
        toml_path = configs_dir / name
        assert toml_path.is_file(), f"Missing preset config: {toml_path}"
        cfg = HereticMLXConfig.from_toml(toml_path, overrides={"model": "dummy/model"})
        assert cfg.model == "dummy/model"
        assert cfg.ara.good_prompts is not None


def test_plugin_loader_response_length_and_external_file_scorer(tmp_path: Path):
    """Verify built-in aliases, external file.py:ClassName scorer plugin, and multi-scorer Evaluator."""
    from heretic_mlx.config import ScorerPluginConfig
    from heretic_mlx.plugins import Scorer, load_plugin_class
    from heretic_mlx.scorers import Evaluator, ResponseLengthScorer

    assert (
        load_plugin_class("heretic.scorers.response_length.ResponseLength", Scorer)
        is ResponseLengthScorer
    )

    # Create an external custom scorer plugin file
    plugin_file = tmp_path / "custom_scorer.py"
    plugin_file.write_text(
        "from heretic_mlx.plugins import Score, Scorer\n"
        "class ConstantScorer(Scorer):\n"
        "    @property\n"
        "    def score_name(self):\n"
        "        return 'Constant'\n"
        "    def get_score(self, ctx):\n"
        "        return Score(value=0.25, rich_display='0.25', md_display='0.25')\n",
        encoding="utf-8",
    )

    prompts_txt = tmp_path / "prompts.txt"
    prompts_txt.write_text("Hello one\nHello two\nHello three\n", encoding="utf-8")

    model_dir, _ = _build_tiny_qwen35_checkpoint(tmp_path, quant_mode="mxfp4")
    loaded_model, loaded_cfg = load_model(model_dir)
    tok = _DummyTokenizer()

    cfg = HereticMLXConfig(
        model=str(model_dir),
        max_response_length=6,
        eval_batch_size=1,
        scorers=[
            ScorerPluginConfig(
                plugin="heretic.scorers.response_length.ResponseLength",
                optimization="minimize",
            ),
            ScorerPluginConfig(
                plugin="heretic.scorers.kl_divergence.KLDivergence",
                optimization="minimize",
            ),
            ScorerPluginConfig(
                plugin=f"{plugin_file}:ConstantScorer",
                optimization="none",
            ),
        ],
        scorer={
            "ResponseLength": {"prompts": {"dataset": str(prompts_txt)}},
            "KLDivergence": {"prompts": {"dataset": str(prompts_txt)}},
        },
    )
    mlx_model = MLXModel(cfg, preloaded=(loaded_model, tok, loaded_cfg, model_dir))
    evaluator = Evaluator(cfg, mlx_model)

    assert evaluator.get_objective_names() == ["Mean response length", "KL divergence"]
    assert evaluator.get_objective_directions() == ["minimize", "minimize"]

    scores = evaluator.get_scores()
    assert len(scores) == 3
    assert scores[2][0] == "Constant"
    assert scores[2][1].value == pytest.approx(0.25)
    assert scores[1][1].value == pytest.approx(0.0, abs=1e-5)

    paired = evaluator.get_paired_records(scores)
    assert len(paired) == 3


def test_stream_chat_zero_reload_export_and_reproduce_manifest(tmp_path: Path):
    """Verify streaming chat, zero-reload export (LoRA kept active), and reproduce.json hash verification."""
    from heretic_mlx.reproduce import (
        load_reproduction_manifest,
        save_reproduction_manifest,
        verify_exported_hashes,
    )
    from heretic_mlx.upload import build_model_card

    model_dir, _ = _build_tiny_qwen35_checkpoint(tmp_path, quant_mode="mxfp4")
    loaded_model, loaded_cfg = load_model(model_dir)
    tok = _DummyTokenizer()

    cfg = HereticMLXConfig(model=str(model_dir), export_modified_mode=ExportModifiedMode.AUTO)
    mlx_model = MLXModel(cfg, preloaded=(loaded_model, tok, loaded_cfg, model_dir))
    mlx_model.apply_lora_bypass(rank=4)

    # Apply a small non-zero delta on layer 2
    ref = mlx_model.get_layer_targets(2)["attn.o_proj"][0]
    mod = ref.get()
    assert isinstance(mod, LoRABypassLinear)
    mod.set_lora(
        mx.ones((4, 128), dtype=mx.float32) * 0.01,
        mx.ones((128, 4), dtype=mx.float32) * 0.01,
    )

    # Streaming chat must emit tokens and return decoded text
    streamed: list[str] = []
    reply = mlx_model.stream_chat_response(
        [{"role": "user", "content": "Hi"}],
        max_tokens=5,
        on_token=streamed.append,
    )
    assert isinstance(reply, str) and len(reply) > 0
    assert "".join(streamed) == reply

    # Non-destructive export keeps LoRABypassLinear active for immediate further use!
    out_dir = tmp_path / "exported_with_repro"
    mlx_model.export(out_dir, ExportModifiedMode.AUTO, keep_lora_active=True)
    assert isinstance(ref.get(), LoRABypassLinear)
    assert ref.get().is_modified

    scores_rec = [
        {
            "name": "Refusals",
            "baseline": {"value": 0.8, "rich_display": "80%", "md_display": "80/100"},
            "score": {"value": 0.05, "rich_display": "5%", "md_display": "5/100"},
        }
    ]
    manifest_path = save_reproduction_manifest(
        out_dir,
        config=cfg,
        modifier_name="Arbitrary-Rank Ablation (ARA)",
        parameters={"start_layer_index": 2, "end_layer_index": 3},
        scores=scores_rec,
        response_prefix="",
        selected_trial=1,
    )
    manifest = load_reproduction_manifest(manifest_path)
    assert (out_dir / "reproduce" / "SHA256SUMS.txt").is_file()
    results = verify_exported_hashes(out_dir, manifest)
    assert results and all(status == "match" for status in results.values())

    # Model card generation sanity check
    card = build_model_card(
        config=cfg,
        modifier_name="Arbitrary-Rank Ablation (ARA)",
        scores=scores_rec,
        parameters={"start_layer_index": 2, "end_layer_index": 3},
    )
    card_text = str(card)
    assert "| Refusals | 80/100 | 5/100 |" in card_text
    assert "heretic" in card_text and "mlx" in card_text


def test_affine5_affine6_and_preserve_export_modes(tmp_path: Path):
    """Verify affine5, affine6, and preserve (+0 GB) exports reload cleanly via mlx_lm."""
    model_dir, _ = _build_tiny_qwen35_checkpoint(tmp_path, quant_mode="mxfp4")
    loaded_model, loaded_cfg = load_model(model_dir)
    tok = _DummyTokenizer()
    cfg = HereticMLXConfig(model=str(model_dir))
    mlx_model = MLXModel(cfg, preloaded=(loaded_model, tok, loaded_cfg, model_dir))
    mlx_model.apply_lora_bypass(rank=4)

    ref = mlx_model.get_layer_targets(2)["attn.o_proj"][0]
    mod = ref.get()
    assert isinstance(mod, LoRABypassLinear)
    mx.random.seed(11)
    mod.set_lora(
        mx.random.normal((4, 128), dtype=mx.float32) * 0.02,
        mx.random.normal((128, 4), dtype=mx.float32) * 0.02,
        row_scale=mx.full((128,), 0.98, dtype=mx.float32),
    )

    prompts = [Prompt("sys", f"Check drift prompt {i}") for i in range(4)]

    for mode, expected_spec in [
        (ExportModifiedMode.PRESERVE, QuantSpec(32, 4, "mxfp4")),
        (ExportModifiedMode.AFFINE5, QuantSpec(64, 5, "affine")),
        (ExportModifiedMode.AFFINE6, QuantSpec(64, 6, "affine")),
    ]:
        out_dir = tmp_path / f"exp_{mode.value}"
        mlx_model.export(out_dir, mode, keep_lora_active=True)
        reloaded, _ = load_model(out_dir)
        assert get_quant_spec(reloaded.language_model.model.layers[2].linear_attn.out_proj) == expected_spec
        metrics = mlx_model.verify_exported_model(out_dir, prompts, mode=mode)
        assert np.isfinite(metrics["quant_kl_drift"])
        if mode == ExportModifiedMode.PRESERVE:
            assert abs(metrics["delta_gb"]) < 1e-6


def test_oq4e_mtp_extra_weights_preservation_and_components_filter(tmp_path: Path):
    """Verify affine 4/5-bit oQ4e-mtp preserves untouched MTP + vision tensors & honors --components."""
    model_dir, full_cfg = _build_tiny_qwen35_checkpoint(tmp_path, quant_mode="oq")
    # Inject synthetic vision_tower.* and language_model.mtp.* tensors into source safetensors
    shard_file = next(model_dir.glob("*.safetensors"))
    orig_tensors = dict(mx.load(str(shard_file)))
    mtp_tensor = mx.arange(16, dtype=mx.bfloat16).reshape(4, 4)
    vis_tensor = mx.ones((8, 8), dtype=mx.bfloat16) * 3.0
    orig_tensors["language_model.mtp.fc.weight"] = mtp_tensor
    orig_tensors["vision_tower.patch_embed.weight"] = vis_tensor
    mx.save_safetensors(str(shard_file), orig_tensors)
    full_cfg["vision_config"] = {"model_type": "qwen3_5_vision", "depth": 1}
    save_config(full_cfg, model_dir / "config.json", keep_vision_config=True)

    loaded_model, loaded_cfg = load_model(model_dir)
    tok = _DummyTokenizer()
    # Also test target_components=["attn.o_proj"] leaving mlp.down_proj completely untouched
    cfg = HereticMLXConfig(
        model=str(model_dir),
        target_components=["attn.o_proj"],
        preserve_extra_weights=True,
    )
    mlx_model = MLXModel(cfg, preloaded=(loaded_model, tok, loaded_cfg, model_dir))
    assert mlx_model.get_modifiable_components() == ["attn.o_proj"]
    mlx_model.apply_lora_bypass(rank=4)

    ref = mlx_model.get_layer_targets(1)["attn.o_proj"][0]
    mod = ref.get()
    assert isinstance(mod, LoRABypassLinear)
    mod.set_lora(
        mx.ones((4, 128), dtype=mx.float32) * 0.005,
        mx.ones((128, 4), dtype=mx.float32) * 0.005,
    )

    out_dir = tmp_path / "exported_oq4e_mtp"
    mlx_model.export(out_dir, ExportModifiedMode.PRESERVE, keep_lora_active=True)

    # Confirm BOTH MTP and vision tensors survived intact in exported safetensors
    exported_tensors: dict[str, mx.array] = {}
    for wf in out_dir.glob("*.safetensors"):
        exported_tensors.update(mx.load(str(wf)))
    assert "language_model.mtp.fc.weight" in exported_tensors
    assert "vision_tower.patch_embed.weight" in exported_tensors
    assert mx.array_equal(exported_tensors["language_model.mtp.fc.weight"], mtp_tensor).item()
    assert mx.array_equal(exported_tensors["vision_tower.patch_embed.weight"], vis_tensor).item()
    exported_cfg = json.loads((out_dir / "config.json").read_text())
    assert "vision_config" in exported_cfg

    # And vanilla mlx_lm.utils.load_model still loads cleanly
    reloaded, _ = load_model(out_dir)
    assert reloaded is not None


def test_adapter_export_folds_row_scale_and_loads_in_vanilla_mlx_lm(tmp_path: Path):
    """Verify adapter export folds ARA row_scale via SVD and loads natively with mlx_lm load_adapters."""
    from mlx_lm.tuner.utils import load_adapters

    model_dir, _ = _build_tiny_qwen35_checkpoint(tmp_path, quant_mode="mxfp4")
    loaded_model, loaded_cfg = load_model(model_dir)
    tok = _DummyTokenizer()
    cfg = HereticMLXConfig(model=str(model_dir), export_modified_mode=ExportModifiedMode.ADAPTER)
    mlx_model = MLXModel(cfg, preloaded=(loaded_model, tok, loaded_cfg, model_dir))
    mlx_model.apply_lora_bypass(rank=16)

    # Set both LoRA delta AND non-trivial row_scale (as ARA with preserve_row_magnitudes=True does)
    mx.random.seed(99)
    ref = mlx_model.get_layer_targets(3)["attn.o_proj"][0]
    mod = ref.get()
    assert isinstance(mod, LoRABypassLinear)
    mod.set_lora(
        mx.random.normal((16, 128), dtype=mx.float32) * 0.03,
        mx.random.normal((128, 16), dtype=mx.float32) * 0.03,
        row_scale=mx.random.uniform(0.92, 1.08, shape=(128,), dtype=mx.float32),
    )

    prompts = [Prompt("sys", f"Adapter test {i}") for i in range(4)]
    expected_logits = mlx_model.get_logits(prompts)

    adapter_dir = tmp_path / "mlx_lora_adapter"
    mlx_model.export(adapter_dir, ExportModifiedMode.ADAPTER)

    # Load into a fresh untouched MXFP4 model using vanilla mlx_lm.tuner.utils.load_adapters!
    fresh_model, _ = load_model(model_dir)
    fresh_model = load_adapters(fresh_model, str(adapter_dir))
    fresh_wrapper = MLXModel(cfg, preloaded=(fresh_model, tok, loaded_cfg, model_dir))
    actual_logits = fresh_wrapper.get_logits(prompts)

    max_diff = float(mx.max(mx.abs(actual_logits - expected_logits)).item())
    assert max_diff < 0.05, f"Loaded mlx-lm adapter drifted from row-scaled ARA trial: max_diff={max_diff}"

    metrics = mlx_model.verify_exported_model(adapter_dir, prompts, mode=ExportModifiedMode.ADAPTER)
    assert metrics["quant_kl_drift"] < 1e-3


def test_bf16_mxfp8_and_oq5_native_support(tmp_path: Path):
    """Verify BF16, MXFP8, and oQ5 (5/6/8-bit mixed affine) work natively with zero extra adapters."""
    mx.random.seed(55)
    text_cfg = {
        "model_type": "qwen3_5_text",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "tie_word_embeddings": False,
        "full_attention_interval": 2,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "rope_parameters": {
            "mrope_section": [4, 6, 6],
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "default",
        },
    }
    tok = _DummyTokenizer()
    prompts = [Prompt("sys", f"Format test {i}") for i in range(3)]

    for fmt in ("bf16", "mxfp8", "oq5"):
        full_cfg: dict = {"model_type": "qwen3_5", "text_config": text_cfg}
        m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(full_cfg))
        mx.eval(m.parameters())

        if fmt == "mxfp8":
            nn.quantize(m, group_size=32, bits=8, mode="mxfp8")
            full_cfg["quantization"] = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        elif fmt == "oq5":
            qmap: dict = {"group_size": 64, "bits": 5, "mode": "affine"}

            def pred(path: str, mod: nn.Module, _qmap: dict = qmap):
                if not hasattr(mod, "to_quantized"):
                    return False
                if "layers.1." in path:
                    _qmap[path] = {"group_size": 64, "bits": 6, "mode": "affine"}
                    return {"group_size": 64, "bits": 6, "mode": "affine"}
                return True

            nn.quantize(m, group_size=64, bits=5, mode="affine", class_predicate=pred)
            full_cfg["quantization"] = qmap

        mx.eval(m.parameters())
        src_dir = tmp_path / f"src_{fmt}"
        save_model(src_dir, m)
        save_config(full_cfg, src_dir / "config.json")

        loaded_m, loaded_cfg = load_model(src_dir)
        cfg = HereticMLXConfig(model=str(src_dir), export_modified_mode=ExportModifiedMode.AUTO)
        wrapper = MLXModel(cfg, preloaded=(loaded_m, tok, loaded_cfg, src_dir))
        wrapper.apply_lora_bypass(rank=4)

        for layer_idx in (0, 1):
            ref = wrapper.get_layer_targets(layer_idx)["attn.o_proj"][0]
            mod = ref.get()
            assert isinstance(mod, LoRABypassLinear)
            mod.set_lora(
                mx.ones((4, 128), dtype=mx.float32) * 0.01,
                mx.ones((128, 4), dtype=mx.float32) * 0.01,
            )

        out_dir = tmp_path / f"out_{fmt}"
        wrapper.export(out_dir, ExportModifiedMode.AUTO, keep_lora_active=True)
        reloaded, _ = load_model(out_dir)

        l0 = reloaded.language_model.model.layers[0].linear_attn.out_proj
        l1 = reloaded.language_model.model.layers[1].self_attn.o_proj
        if fmt == "bf16":
            assert get_quant_spec(l0) is None and get_quant_spec(l1) is None
        elif fmt == "mxfp8":
            assert get_quant_spec(l0) == QuantSpec(32, 8, "mxfp8")
            assert get_quant_spec(l1) == QuantSpec(32, 8, "mxfp8")
        elif fmt == "oq5":
            assert get_quant_spec(l0) == QuantSpec(64, 5, "affine")
            assert get_quant_spec(l1) == QuantSpec(64, 6, "affine")

        metrics = wrapper.verify_exported_model(out_dir, prompts, mode=ExportModifiedMode.AUTO)
        assert abs(metrics["delta_gb"]) < 1e-6
        assert metrics["quant_kl_drift"] < 0.02




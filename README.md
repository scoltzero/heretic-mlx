<div align="center">

<img width="128" alt="Heretic-MLX logo" src="assets/logo.png" />

# Heretic-MLX

### Training-free language model modification on Apple Silicon (MLX)

**Remove censorship**, suppress slop, reduce sycophancy, and more.<br>
All in an afternoon, natively on your Mac — directly on **MXFP4**, **MXFP8**, **oQ**, and **BF16** weights.

[![Upstream Heretic](https://img.shields.io/badge/Upstream-p--e--w%2Fheretic-black?logo=github&style=for-the-badge)](https://github.com/p-e-w/heretic)&nbsp;&nbsp;&nbsp;&nbsp;
[![Apple Silicon MLX](https://img.shields.io/badge/Apple%20Silicon-MLX%20Native-black?logo=apple&style=for-the-badge)](https://github.com/ml-explore/mlx)&nbsp;&nbsp;&nbsp;&nbsp;
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue?style=for-the-badge)](LICENSE)

</div>

-----

&nbsp;

## Modify any MLX LLM that you can run on your Apple Silicon Mac[^1]

[^1]: Unlike PyTorch-based abliteration tools that fall back to CPU on modern hybrid layers (such as Qwen3.5 / Qwen3.8 GatedDeltaNet) and require full-precision checkpoints or CUDA `bitsandbytes`, Heretic-MLX runs end-to-end on Apple Metal unified memory and operates **directly on quantized MXFP4, MXFP8, oQ (mixed 4/5/6/8-bit affine), and BF16 weights** without duplicating base weights during optimization.

**Heretic-MLX** is an Apple Silicon / MLX native port of [Heretic](https://github.com/p-e-w/heretic) (v2.2). Just like upstream Heretic, it lets you do what only corporations with access to massive computing power usually do: *Modify,* rather than merely run, language models. It does not perform expensive training; rather, it uses a collection of mathematical techniques to steer behavior based on contrasting example prompts and matching scorers.

Heretic-MLX can, depending on the configuration:

* **Suppress refusals/remove censorship** (this is the [default action](configs/default.toml))
* Reduce slop ([`configs/noslop.toml`](configs/noslop.toml))
* Reduce humor ([`configs/nohumor.toml`](configs/nohumor.toml))
* Reduce sycophancy ([`configs/nosyco.toml`](configs/nosyco.toml))
* Alter emotional alignment ([`configs/emotions.toml`](configs/emotions.toml))
* Shorten response length ([`configs/terse.toml`](configs/terse.toml))
* Use a ModernBERT refusal classifier ([`configs/classifier.toml`](configs/classifier.toml))

and many, many other things.

### :bangbang: Using Heretic-MLX does NOT require understanding how language models work internally :bangbang:

In fact, you can go from zero to removing censorship from your first MLX model [in a matter of seconds](#usage). Heretic-MLX supports most dense models, multimodal VLM checkpoints (preserving untouched `vision_tower.*` tensors), MTP speculative decoding checkpoints (preserving untouched `language_model.mtp.*` tensors), modern hybrid architectures (**Qwen3.8-27B** GatedDeltaNet + Attention, Liquid LFM, Granite Hybrid), and MoE architectures (**Qwen3.8-Flash-Next** / `qwen4_exp` with flattened Hyper-Connections and singular `shared_expert`).

#### Supported Quantization Formats

Heretic-MLX inspects every modifiable linear layer dynamically, supporting all standard MLX formats out of the box:

| Input Format | Supported | Default `auto` Export Behavior | Size Delta |
| :--- | :---: | :--- | :---: |
| **BF16 / FP16 / FP32** | Yes | `bf16` | `+0.00 GB` |
| **MXFP8** (`mxfp8`, `g=32`) | Yes | `preserve` (re-pack as MXFP8) | `+0.00 GB` |
| **oQ4e / oQ5 / oQ6 / oQ8** (mixed-precision `affine`) | Yes | `preserve` (keeps exact per-layer bit-width & `group_size`) | `+0.00 GB` |
| **Standard MLX 4/5/6/8-bit `affine`** | Yes | `preserve` | `+0.00 GB` |
| **MXFP4** (`mxfp4`, `g=32`) | Yes | `affine6` (or pass `--export-modified-mode preserve` / `adapter`) | `+1.61 GB` (`preserve`: `+0 GB`, `adapter`: `~164 MB`) |

#### Verified Apple Silicon Benchmarks (M5 Ultra 256 GB)

Measured on `mlx-community/Qwen3.8-27B-mxfp4` (64 layers, 128 MXFP4 projections, 102 modified modules from a single ARA trial with pre-export `KL = 0.1360`, 333 `vision_tower` tensors preserved):

| Export Mode | Output Size | Size Delta vs. 14.17 GB | Re-Quant KL Drift | KL vs. Original Base | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`preserve`** *(pure MXFP4)* | **14.17 GB** | **+0.00 GB** | `0.0549` | `0.1858` | Pure MXFP4, zero overrides, byte-exact `--reproduce` SHA256 verified |
| **`adapter`** *(LoRA only)* | **164.4 MB** | **+0.16 GB** | `0.0186` | `0.1433` | Folds ARA `row_scale` via SVD; loads natively with `mlx_lm` `--adapter-path` |
| **`affine5`** *(5-bit `g=64`)* | 15.06 GB | +0.90 GB | `0.0177` | `0.1431` | Compact mixed-precision upgrade |
| **`affine6`** *(6-bit `g=64`, `auto` default)* | **15.78 GB** | **+1.61 GB** | **`0.0029`** | **`0.1347`** | Best precision/size trade-off (~same drift as 8-bit, half the size delta) |
| **`affine8`** *(8-bit `g=64`)* | 17.21 GB | +3.04 GB | `0.0020` | `0.1312` | Near-lossless merged weights |

On `Jundot/Qwen3.8-27B-oQ4e-mtp` (`--components attn.o_proj`, `--export-modified-mode auto`):
* **40-layer attention-only ARA time**: **17.7 s** on M5 Ultra (full 27B MXFP4 ARA trial: **49–81 s/trial**).
* **Size delta**: `15.806 GB -> 15.806 GB` (**`+0.0000 GB`**), re-quantization KL drift **`0.0081`**.
* **Auxiliary tensors preserved**: **29/29** `language_model.mtp.*` + **333/333** `vision_tower.*` + `vision_config` (verified via vanilla `mlx_lm.load()`).

&nbsp;

## Usage

Prepare an Apple Silicon Mac (M1/M2/M3/M4/M5) with Python 3.10+ and [`uv`](https://docs.astral.sh/uv/). Then run:

```sh
git clone https://github.com/scoltzero/heretic-mlx.git
cd heretic-mlx
uv sync
uv run heretic-mlx /path/to/Qwen3.8-27B-oQ4e-mtp --verify-export
```

Replace `/path/to/Qwen3.8-27B-oQ4e-mtp` with any local MLX/MXFP4/oQ/BF16 directory or Hugging Face repo ID.
**By default, Heretic-MLX performs censorship removal (refusal suppression).**
If you want to do something else, choose one of the configuration files in [`configs/`](configs/), or write your own:

```sh
# Use the Qwen3.8-27B MXFP4 preset
uv run heretic-mlx --config configs/qwen38_mxfp4.toml

# Weights-only MXFP4 folder with a separate tokenizer directory + Chinese refusal markers
uv run heretic-mlx /path/to/Qwen3.8-27B-mxfp4 \
  --tokenizer /path/to/Qwen3.8-27B-4bit \
  --zh-refusals --verify-export

# Zero size growth (+0.00 GB) or standalone mlx-lm LoRA adapter (~164 MB)
uv run heretic-mlx /path/to/model --export-modified-mode preserve --verify-export
uv run heretic-mlx /path/to/model --export-modified-mode adapter -o ./my-heretic-lora

# Attention-only modification (leaves large MLPs untouched)
uv run heretic-mlx /path/to/model --components attn.o_proj --export-modified-mode preserve

# Evaluate an existing community uncensored MLX model apples-to-apples against the base model
uv run heretic-mlx /path/to/base-mlx --evaluate-model /path/to/uncensored-mlx --zh-refusals

# Quick 3-trial smoke test (verifies loading, ARA, and export in ~5 min on 27B)
uv run heretic-mlx /path/to/model --smoke --auto-select --verify-export -o ./smoke-out
```

> [!NOTE]
>
> `--smoke` runs only **3 random trials** on **32 prompts** (20 L-BFGS steps) to verify hardware speed and export integrity. To drive refusal rates down to ~0–5% with minimal KL divergence, run a full optimization (`n_trials = 60..100`, `400` good/bad prompts, `5 × 20` L-BFGS steps).

After Heretic-MLX has finished processing a model, you are given an interactive menu to **chat** live with any Pareto-optimal trial, **run additional trials**, **save** the model in your chosen precision (`auto`, `preserve`, `affine5`, `affine6`, `affine8`, `mxfp8`, `bf16`, `adapter`), or **upload it to Hugging Face** with an auto-generated Model Card — with **zero model reload** when switching trials after saving.

Optional extras:

```sh
uv sync --extra classifier   # Enables BERTClassifier (ModernBERT refusal classifier)
uv sync --extra eval         # Enables BenchmarkScore (lm-eval harness on MLX)
uv sync --extra dev          # Enables pytest & ruff
```


## Controlling what Heretic-MLX does using scorer plugins

Heretic-MLX steers model behavior by optimizing one or multiple scores computed by scorer plugins, keeping full compatibility with upstream Heretic 2.2's `[scorer.<ClassName>]` TOML configuration format.

The `Scorer` interface is defined in [`plugins.py`](src/heretic_mlx/plugins.py). By default, Heretic-MLX scores models by refusal rate (`KeywordRate`, with optional `--zh-refusals` Chinese markers) and KL divergence (`KLDivergence`). All five built-in scorers live in [`scorers.py`](src/heretic_mlx/scorers.py):

* `KeywordRate` (`heretic.scorers.keyword_rate.KeywordRate`)
* `KLDivergence` (`heretic.scorers.kl_divergence.KLDivergence`)
* `ResponseLength` (`heretic.scorers.response_length.ResponseLength`)
* `BERTClassifier` (`heretic.scorers.bert_classifier.BERTClassifier`)
* `BenchmarkScore` (`heretic.scorers.benchmark_score.BenchmarkScore`)

External scorer plugins can be referenced directly in TOML as `path/to/plugin.py:MyScorerClass` or `fully.qualified.module.MyScorerClass` without modifying `heretic-mlx`.


## How Heretic-MLX works

Heretic-MLX combines a modifier with a multivariate TPE hyperparameter optimizer powered by [Optuna](https://optuna.org/) and persistent `JournalStorage` checkpoints (so long runs can be interrupted with `Ctrl+C` and resumed cleanly).

Heretic-MLX ships with both Heretic 2.2 modifiers, implemented natively in MLX:

* **Arbitrary-Rank Ablation (ARA)**[^2]: The default modifier (`--modifier ara`), implemented in [`ara.py`](src/heretic_mlx/ara.py) using compiled MLX GPU autodiff (`mx.compile(mx.value_and_grad(...))`) + native Metal Strong-Wolfe `L-BFGS` + exact low-rank Gram per-row magnitude preservation (`row_scale` applied identically during solving, trial evaluation, and weight merging).
* **Magnitude-Preserving Orthogonal Ablation (MPOA / abliteration)**[^3][^4][^5]: Classic directional ablation (`--modifier abliteration`) with projected orthogonalization, full row normalization + low-rank SVD, and activation winsorization, implemented in [`abliteration.py`](src/heretic_mlx/abliteration.py).

During search, [`quant.py`](src/heretic_mlx/quant.py) wraps target projections in a zero-copy `LoRABypassLinear` that leaves base MXFP4/oQ weights untouched in unified memory and temporarily dequantizes only a single projection matrix to FP32 during solving.

[^2]: *Weidmann 2026* ([`p-e-w/heretic#211`](https://github.com/p-e-w/heretic/pull/211))
[^3]: *[Arditi et al. 2024](https://arxiv.org/abs/2406.11717)*
[^4]: *[Lai 2025 (1)](https://huggingface.co/blog/grimjim/projected-abliteration)*
[^5]: *[Lai 2025 (2)](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)*


## Reproducibility

Whenever Heretic-MLX exports a model, it writes a `reproduce/` folder containing `reproduce.json` (model, modifier, parameters, scores, environment info, and file hashes) alongside `SHA256SUMS.txt` (implemented in [`reproduce.py`](src/heretic_mlx/reproduce.py)).

Automatic reproduction skips the multi-hour Optuna search and reconstructs the modified model directly in minutes:

```sh
uv run heretic-mlx --reproduce path/to/reproduce/reproduce.json -o ./reproduced-model
```

After export, Heretic-MLX verifies the SHA256 hashes of every exported `.safetensors` shard and `config.json` against the manifest (verified 4/4 byte-for-byte SHA256 match on M5 Ultra with `Qwen3.8-27B MXFP4`).


## Tracking Upstream `p-e-w/heretic`

Because upstream [`p-e-w/heretic`](https://github.com/p-e-w/heretic) explicitly keeps mainline focused on PyTorch/Transformers ([PR #226](https://github.com/p-e-w/heretic/pull/226)), `heretic-mlx` is maintained as an independent Apple Silicon sibling repository and tracks its exact upstream baseline in [`UPSTREAM.json`](UPSTREAM.json).

You can check for new upstream releases or commits and map every changed upstream file to its `heretic-mlx` porting target at any time:

```sh
python3 scripts/check_upstream.py --markdown
```

A scheduled GitHub Actions workflow ([`.github/workflows/upstream-watch.yml`](.github/workflows/upstream-watch.yml)) also checks `p-e-w/heretic` daily and opens/updates a tracking issue whenever new upstream commits or releases arrive.


## Acknowledgments

Heretic-MLX is directly inspired by and ported from:

* **[Heretic](https://github.com/p-e-w/heretic)** by Philipp Emanuel Weidmann and contributors — all core algorithms (ARA, MPOA, Optuna TPE multi-objective search, scorer plugins, presets, and reproducibility design) originate from upstream Heretic.
* **[MLX](https://github.com/ml-explore/mlx)** and **[mlx-lm](https://github.com/ml-explore/mlx-lm)** by Apple Machine Learning Research.
* [The original abliteration paper *(Arditi et al. 2024)*](https://arxiv.org/abs/2406.11717), [Maxime Labonne's article on abliteration](https://huggingface.co/blog/mlabonne/abliteration), and Jim Lai's articles on [projected](https://huggingface.co/blog/grimjim/projected-abliteration) and [norm-preserving biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration).


## Citation

If you use Heretic or Heretic-MLX for your research, please cite upstream Heretic:

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Training-free language model modification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```


## License

This project is a derivative MLX port of [Heretic](https://github.com/p-e-w/heretic) (Copyright &copy; 2025-2026 Philipp Emanuel Weidmann + contributors), licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**. See [LICENSE](LICENSE) for details.

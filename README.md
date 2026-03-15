# Anima ex Machina

**Grounding Symbols Through Developmental Perception in a 228K-Parameter Model**

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.19037597)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What This Is

Every large language model knows that "fire" co-occurs with "hot." None has ever perceived heat.

This project builds a world small enough to be completely understood, places a 228K-parameter Transformer inside it, and guides it through a six-stage developmental pathway — from raw perceptual input to autonomous linguistic inference. The model learns what words mean by first learning what the world feels like: 103 entities, 13 sensory dimensions, 6 causal rules, and a developmental sequence modeled on how children acquire language.

Four findings define the system:

- **Developmental order is causally necessary.** Remove interaction training: causal reasoning vanishes (accuracy 0.001). Shuffle phase order: higher-order capacities collapse while lower-order ones survive.
- **The model generalizes through rules, not memorization.** Disambiguation ratio 5.0:1 — when per-dimension rules conflict with name-based retrieval, the model follows the rules.
- **Every step is mechanistically traceable.** Attention specialization, information arrival, and causal flow are localized layer by layer, head by head.
- **Once grounded, language operates autonomously.** With all perceptual input removed, causal reasoning remains at ceiling: 1.000 ± 0.000 across five seeds.

The bottleneck is developmental structure, not scale. Across a 50-fold parameter range (63K to 3.3M), the same capacities emerge.

**Paper:** [Anima ex Machina (Zenodo)](https://doi.org/10.5281/zenodo.19037597)

---

## Why This Matters

Current AI systems process language about fire without any representation of what fire is like. They learn the statistical shadow of meaning — the co-occurrence patterns left behind by beings who have already grounded their symbols in perception — without acquiring grounding itself.

This system takes a different path. It constructs the conditions under which grounding occurs, verifies that it has occurred, and makes every mechanism inspectable. The pathway from perception to representation to naming to grammar to causal reasoning is the functional infrastructure that genuine semantic understanding presupposes. We provide a method for constructing and verifying this infrastructure in a fully controlled environment where every variable can be isolated, every capacity can be ablated, and every internal mechanism can be inspected.

No prior system has occupied this intersection: full mechanistic decomposability combined with the complete perception-to-language-to-autonomous-reasoning pathway.

---

## Context

This research was conducted by a single independent researcher without institutional affiliation, academic supervision, funding, or GPU access. The entire experimental pipeline — five-seed baselines, phase ablations, extended permutations, scaling analysis, bidirectional comparisons, mechanistic decomposition — was run on an Apple M4 chip with 16 GB unified memory. Total compute time: approximately 24 hours.

The paper was developed in collaboration with Claude (Anthropic), which served as a research partner throughout the process — from experimental design discussion and code implementation to statistical analysis and manuscript drafting. The research direction, experimental philosophy, and core intellectual contributions are the author's; the implementation and writing were collaborative.

All code, data, trained model weights, and analysis scripts are fully open-source. Nothing is held back.

**I am currently seeking a stable research environment — a PhD position, visiting researcher role, or any form of collaboration — to continue this line of work. If you see value in what this project demonstrates, I welcome the opportunity to discuss next steps.**

Contact: mahpmiceliu@gmail.com

---

## Quick Start

**Requirements:** Python 3.10+, PyTorch 2.0+. No GPU required (runs on Apple M-series / CPU).

```bash
# Train the primary model (bidir30, seed=42, Phases 1–5)
cd code
python train_v6.py --seed 42 --reverse_ratio 0.3 --max_phase 5

# Run all evaluation experiments on a trained model
python run_all_exp1.py --results_dir ../results/baseline_5seed/seed_42
```

Training through Phase 5 takes ~17 minutes on Apple M4. Through Phase 6: ~22 minutes.

---

## Repository Structure

```
.
├── code/                   # All source code
├── data/                   # World definition
├── dim_ablation/           # Aggregated analysis results
├── paper/                  # Manuscript and figures
└── results/                # Raw experimental outputs (per-seed)
```

---

## code/

### Core System

| File | Purpose |
|------|---------|
| `model.py` | Transformer encoder architecture (4-layer, 4-head, pre-norm, MLM) |
| `train_core.py` | Training loop, masking logic, evaluation functions (P-recon, DESC accuracy, holdout probes, full-mask inference) |
| `train_phases.py` | Phase definitions: data composition, rehearsal multipliers, per-phase evaluation. Edit this file to change the developmental pathway |
| `train_v6.py` | Entry point. Parses args (seed, reverse_ratio, max_phase, d_model, n_layers), builds world, runs training |
| `v6_tool.py` | Data generation: entity encodings, sequence formatting (S1/S2/S3), distance attenuation, interaction computation |
| `v6_rules.py` | The 6 causal rules (heat, force, burn, mix, grow, decay): agent conditions and patient transformations |

### Experiment Scripts

| File | Paper section | What it does |
|------|--------------|-------------|
| `exp1_reverse.py` | §3.3 | Reverse inference: language→perception. Tests groups A–F (trained, novel, mismatch, fictitious, interaction, held-out) |
| `exp2_phase_transition.py` | §3.1.2 | Tracks P-token vs DESC-token ratio across Phase 2 training steps |
| `exp2a_gender.py` | §3.4 | Gender marker experiments: removal, swap, interaction-format ablation |
| `exp2b_phase_ablation.py` | §3.5 | Single-condition phase ablation (ΔPh1–ΔPh5, Shuffle) |
| `exp2b_shuffle_summary.py` | §3.5.6 | Aggregates extended permutation results (S-A, S-B, S-C, Δ34) |
| `exp2b_shuffle_supp.py` | §3.5.6 | Supplementary permutation analysis |
| `exp2c_dim_ablation.py` | §3.6 | Per-dimension ablation: masks one dimension, measures effect on all others |
| `exp2d_interp.py` | §3.7 | Mechanistic decomposition: attention specificity, cross-dim probing, information arrival, causal intervention |
| `exp3_confidence.py` | Not in paper | Confidence calibration analysis (noted in data handbook, §10) |
| `eval_ph6_summary.py` | §3.8 | Phase 6 summary: linguistic autonomy results across seeds |

### Multi-Seed Runners

| File | What it does |
|------|-------------|
| `run_multiseed.py` | General-purpose multi-seed trainer |
| `run_multiseed_baseline.py` | Trains 5-seed baseline (seeds 42, 123, 456, 789, 1024) with bidir30 |
| `run_multiseed_baselineP6.py` | Trains 5-seed baseline through Phase 6 |
| `run_multiseed_ablation_supplement.py` | Trains phase ablation variants (ΔPh1–5, Shuffle, S-A/B/C, Δ34) across 3 seeds |
| `run_bidir_comparison.py` | Trains 0%, 15%, 30% reverse-ratio models (5 seeds each) |
| `run_scale_ablation.py` | Trains scaling variants (d=32/128/256, 2-layer) |
| `run_all_exp1.py` | Runs all evaluation experiments on a trained model directory |
| `run_exp2c_multiseed.sh` | Shell script for multi-seed dimension ablation |

### Aggregation Scripts

| File | What it does |
|------|-------------|
| `aggregate_results.py` | Compiles per-seed results into summary statistics |
| `aggregate_exp2c.py` | Aggregates dimension ablation results across seeds |
| `exp_supplement_multiseed.py` | Generates the formatted multi-seed summary tables |
| `extract_scale_results.py` | Extracts scaling comparison data |

---

## data/

| File | Purpose |
|------|---------|
| `swadesh_v6_world.md` | Complete world definition: all 103 entities with their 13-dimensional perceptual encodings, the 6 causal rules, training/test interaction pairs, and novel entities |

---

## results/

All raw experimental outputs. Each trained model produces a directory containing `model.pt` (weights), `results.json` (training metrics), and experiment-specific JSON files.

### results/baseline_5seed/

The primary experimental models. 5 seeds (42, 123, 456, 789, 1024), bidir30, Phases 1–5.

```
seed_{id}/
├── model.pt                    # Trained weights
├── seed_results.json           # All metrics: baseline, exp1 reverse, exp2a gender,
│                               #   exp2c dim ablation, exp2d mechanistic probing
└── exp1_reverse_results.json   # Detailed reverse inference results
```

**Summary files** (at `baseline_5seed/` level):
- `all_baseline_results.json` — raw per-seed training metrics
- `exp_supplement_results.json` — formatted multi-seed summary (Tables 1–4 in paper)

### results/bidir_comparison/

Bidirectional training comparison. Three conditions × 5 seeds each.

```
bidir_comparison/
├── unidir_0pct/     # reverse_ratio = 0%   (seeds: 42, 137, 271, 314, 577)
├── bidir_15pct/     # reverse_ratio = 15%  (seeds: 42, 137, 271, 314, 577)
└── bidir_30pct/     # reverse_ratio = 30%  (seeds: 42, 137, 271, 314, 577)
    └── exp2c_aggregate.json   # Aggregated dim ablation for 30% condition
```

Each seed directory contains `model.pt`, `model_ph{1-5}.pt` (per-phase checkpoints), `results.json`, and `exp1_reverse_results.json`.

→ Paper Table 5 (resource competition).

### results/phase_ablation/

Phase ablation and permutation experiments.

```
phase_ablation/
├── phase_ablation/                    # Single-seed (seed=42) ablations
│   ├── results_2b_dph{1-5}/          # ΔPh1 through ΔPh5
│   └── results_2b_shuffle/           # Shuffle [5,4,1,3,2]
└── results_multiseed_ablation/        # 3-seed (42, 123, 456) ablations
    ├── dph{1-5}_seed_{id}/            # Phase ablations
    ├── shuffle_seed_{id}/             # Original shuffle
    ├── results_2b_shuf_12354_s{id}/   # S-A: [1,2,3,5,4]
    ├── results_2b_shuf_125_s{id}/     # Δ34: [1,2,5]
    ├── results_2b_shuf_31245_s{id}/   # S-B: [3,1,2,4,5]
    └── results_2b_shuf_52413_s{id}/   # S-C: [5,2,4,1,3]
```

→ Paper Tables 8, 8b (phase ablation and extended permutations).

### results/results_multiseed_ph6/

Phase 6 (linguistic autonomy) models. 5 seeds, full Phases 1–6.

```
results_multiseed_ph6/
├── results_v4_ph6_s42/    # Full run with per-phase checkpoints
│   ├── model_ph{1-6}.pt
│   └── results.json
├── seed_1024/model.pt     # Other seeds (final weights only)
├── seed_123/model.pt
├── seed_456/model.pt
└── seed_789/model.pt
```

→ Paper Table 15, §3.8.

### results/scaling/

Scaling analysis. All seed=42, bidir30.

```
scaling/
├── xs_d32_4L/    # d=32, 4 layers, 63K params
├── xs_d32_2L/    # d=32, 2 layers, 37K params
├── m_seed_42/    # d=128, 4 layers, 841K params
├── l_seed_42/    # d=256, 4 layers, 3.3M params
└── summary.json  # Comparative table
```

→ Paper Table 2, Supplementary Table S2.

---

## dim_ablation/

Aggregated analysis results (compiled from per-seed raw data). These are the files used to generate paper figures and tables.

| File | Content | Paper reference |
|------|---------|----------------|
| `multiseed_summary.json` | 5-seed baseline summary statistics | Tables 1, 3, 6, 7 |
| `multiseed_full.json` | Complete per-seed data for all experiments | Tables S3, S4 |
| `all_ablation_results.json` | Phase ablation results (ΔPh1–5, Shuffle) | Table 8 |
| `all_ablation_results_dph5-shuffle.json` | ΔPh5 and Shuffle multi-seed detail | Table 8 |
| `bidir_comparison_results.json` | 0%/15%/30% reverse-ratio comparison | Table 5 |
| `exp2c_dim_ablation_results.json` | 13×13 dimension ablation matrix | Table 9, Table S8 |
| `exp2d_interp_results.json` | Mechanistic probing: attention, cross-dim, causal intervention | Tables 10–13, Table S9 |
| `exp2b_shuffle_summary.json` | Extended permutation results (S-A, S-B, S-C, Δ34) | Table 8b |
| `exp2_phase_transition.csv` | Phase 2 transition curve data (P-token vs DESC ratio) | Figure 2 |
| `exp3_confidence.csv` | Confidence calibration data | Not in paper (§10 of data handbook) |
| `exp3_summary.csv` | Confidence summary statistics | Not in paper |
| `phase_transition.pdf` | Phase transition figure | Figure 2 |
| `plot_phase_transition.py` | Script to generate phase transition figure | — |

---

## paper/

| File | Purpose |
|------|---------|
| `Anima_ex_Machina_FINAL.md` | Full manuscript (Markdown source) |
| `Supplementary_Materials.md` | Supplementary tables S1–S10, Notes S1–S2 |

### paper/figures/

| File | Paper location |
|------|---------------|
| `Fig1_System_Architecture.{pdf,png}` | Figure 1: closed world + developmental pathway |
| `phase_transition.png` | Figure 2: Phase 2 perception→language transition |
| `Fig3_Three_Layer_Separation.{pdf,png}` | Figure 3: perception / identity / language layers |
| `Fig4_Mechanistic_Decomposition.{pdf,png}` | Figure 4: attention specialization + information flow |
| `Fig5_Ablation_Heatmap.{pdf,png}` | Figure 5: phase ablation results |

---

## Reproducing All Results

### Step 1: Train the 5-seed baseline

```bash
cd code
python run_multiseed_baseline.py
```

This trains 5 models (seeds 42, 123, 456, 789, 1024) with bidir30 through Phase 5. Output: `results/baseline_5seed/seed_{id}/`. ~85 minutes total.

### Step 2: Run all experiments on each seed

```bash
python exp_supplement_multiseed.py
```

Runs reverse inference, gender ablation, dimension ablation, and mechanistic probing on all 5 seeds. Output: per-seed `seed_results.json` and aggregated summaries.

### Step 3: Phase ablation

```bash
python run_multiseed_ablation_supplement.py
```

Trains ΔPh1–5, Shuffle, and extended permutations (S-A, S-B, S-C, Δ34) across 3 seeds. Output: `results/phase_ablation/`.

### Step 4: Bidirectional comparison

```bash
python run_bidir_comparison.py
```

Trains 0%, 15%, 30% reverse-ratio models (5 seeds each). Output: `results/bidir_comparison/`.

### Step 5: Scaling analysis

```bash
python run_scale_ablation.py
```

Trains d=32 (4L and 2L), d=128, d=256 variants. Output: `results/scaling/`.

### Step 6: Phase 6 (linguistic autonomy)

```bash
python run_multiseed_baselineP6.py
```

Trains 5-seed baseline through Phase 6. Output: `results/results_multiseed_ph6/`.

### Full pipeline

Steps 1–6 complete in approximately 24 hours on Apple M4 (16 GB). No GPU required. All results are deterministic given seed, though CUDA non-determinism may produce slight variation on GPU hardware (see Supplementary Note S2).

---

## Paper ↔ Data Mapping

| Paper table | Data source |
|-------------|-------------|
| Table 1 (forward pathway) | `results/baseline_5seed/seed_{id}/seed_results.json` → `baseline` |
| Table 2 (scaling) | `results/scaling/summary.json` |
| Table 3 (reverse inference) | `results/baseline_5seed/seed_{id}/seed_results.json` → `exp1_reverse` |
| Table 4 (C-group disambiguation) | `results/baseline_5seed/seed_{id}/exp1_reverse_results.json` |
| Table 5 (resource competition) | `dim_ablation/bidir_comparison_results.json` |
| Table 6 (gender swap) | `results/baseline_5seed/seed_{id}/seed_results.json` → `exp2a_gender` |
| Table 7 (gender in interaction) | same as above |
| Table 8 (phase ablation) | `results/phase_ablation/results_multiseed_ablation/` |
| Table 8b (extended permutations) | `dim_ablation/exp2b_shuffle_summary.json` |
| Table 9 (dimension ablation) | `dim_ablation/exp2c_dim_ablation_results.json` |
| Tables 10–13 (mechanistic) | `dim_ablation/exp2d_interp_results.json` |
| Table 15 (Phase 6) | `code/ph6_summary.json` |

---

## Hardware

All experiments were conducted on an Apple M4 chip with 16 GB unified memory. No discrete GPU is required. The entire experimental pipeline (training + evaluation + all ablations) completes in approximately 24 hours.

---

## License

MIT License. All code, data, trained weights, and analysis scripts are freely available for research and commercial use.

---

## Citation

```bibtex
@article{liu2026anima,
  title={Anima ex Machina: Grounding Symbols Through Developmental Perception in a 228K-Parameter Model},
  author={Liu, Zhiwei},
  year={2026},
  doi={10.5281/zenodo.19037597}
}
```

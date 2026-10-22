# CausalEEGNet — reproducibility package

Code accompanying the manuscript *"Temporal Predictive Alignment Between EEG
and Eye Movements for Gaze Prediction Using Cross-Attention"* (IEEE Access,
Access-2026-39609). The model is called CausalEEGNet in the manuscript and
`TempAlignEEGNet` in the code.

## Contents

| File | Purpose |
|---|---|
| `data_pipeline.py` | Loads the BALLADEER dataset, preprocesses EEG (detrend, 1–45 Hz band-pass, AF3/AF4 regression-based EOG correction), builds 2 s windows with 50-point gaze contexts, applies peak-to-peak quality control, and freezes the window cache. |
| `models_v2.py` | CausalEEGNet, its gaze-only and EEG-only variants, and the learned-mask network. |
| `baselines_v2.py` | LSTM, Transformer, LTC, and wavelet baselines with the same dual inputs and fusion head. |
| `metrics.py` | Pearson r, out-of-sample predictive R² (from prediction errors), per-participant metrics, participant bootstrap CIs, paired Wilcoxon and t-tests. |
| `run_main.py` | Nested participant-level 5-fold CV: outer folds for evaluation, inner 20% of training participants for early stopping. |
| `run_extra_models.py` | Masked LSTM and masked Transformer baselines (same EEG towers, CausalEEGNet's temporal mask). |
| `run_ablations.py` | No-mask, frontal-excluded, learned-mask, single-band, EEG-scrambled, target-shuffled, and parameter-matched variants. |
| `run_lagged.py` | Lagged directional analysis (16 Hz, four lags, blocked CV, circular-shift permutations), both directions. |
| `verify_mask.py` | Verifies that attention on temporally prohibited positions is exactly zero for both mask types, saves per-sample attention profiles for Fig. 6, and runs the synthetic delay-identifiability control. |
| `run_channel_zeroout.py` | Per-channel zero-out importance across folds. |
| `run_structure_history.py` | Records the learned delay, bandwidth, and temperature at every epoch of one representative run for Fig. 8(c,d). |
| `run_synth_extra.py` | Adds the two middle offsets (1.10, 1.20) to the synthetic delay-recovery controls. |
| `summarize_results.py` | Aggregates all runs into `results/summary.json`, including the masked baselines and the `LearnedMaskNet` alias. |
| `emit_numbers.py` | Emits `results/numbers.tex` from the registry; optionally audits a local manuscript tree if present. |
| `make_figures.py` | Generates `figures/Fig1.pdf`–`Fig13.pdf` from the registry. |

## Paths

All scripts resolve paths relative to the repository root:

```
cache/        generated window cache
results/      JSON registries and numbers.tex
predictions/  per-window held-out predictions
figures/      generated figures
DATA/         BALLADEER dataset (not included)
```

## Reproduction order

```bash
python data_pipeline.py
python run_main.py --models TempAlignEEGNet,GazeOnly,EEGOnly,LSTM,Transformer,LTC,WaveletTransformer
python run_main.py --models StatBaselines
python run_extra_models.py --models MaskedLSTM,MaskedTransformer --seeds 42,43,44
python run_ablations.py --variants NoMask,FrontalExcluded,TempStructNet,PermutedTarget --seeds 42,43,44
python run_ablations.py --variants PermutedEEG,BandTheta,BandAlpha,BandBeta,LTC_param_matched --seeds 42
python verify_mask.py
python run_lagged.py
python run_channel_zeroout.py
python run_structure_history.py
python summarize_results.py
python emit_numbers.py
python make_figures.py
```

Run the commands from the `code/` directory or use `python code/<script>.py`
from the repository root; both resolve paths identically.

GPU kernels may introduce minor nondeterminism; `torch.use_deterministic_algorithms(True)`
can be enabled if exact bitwise reproducibility is required.

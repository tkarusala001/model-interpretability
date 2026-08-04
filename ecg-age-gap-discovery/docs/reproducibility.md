# Reproducibility

## Environment

Python 3.12.6 on macOS (Darwin 24.0.0); `pyproject.toml` requires ≥3.11.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # ~3.5 minutes, 350 tests
```

Pinned lower bounds are in `pyproject.toml`. Every run directory records the
exact installed versions of torch, numpy, scipy, pandas, scikit-learn and wfdb,
along with the git commit and the resolved configuration, in `config.json`.

## Determinism

Every entrypoint calls `set_global_seed(seed)`, which seeds `random`, `numpy`,
`torch`, and sets `PYTHONHASHSEED`, cuDNN determinism and disables benchmark
autotuning. Data loaders take an explicitly seeded generator.

Verified by test: `test_training_is_reproducible` asserts identical test-set
predictions across two runs with the same seed, and
`test_parameters_are_fully_determined_by_the_seed` asserts model weights match
bit-for-bit — which is also how the no-pretrained-weights claim is checked.

**One hardware caveat.** Attribution defaults to CPU even when a GPU is present,
because Apple's MPS backend was observed returning *exactly zero* gradients for
18 of 20 recordings at batch size 20, silently. Integrated Gradients verifies
its own completeness axiom and warns if violated; keep that check enabled.

## Reproducing every result

```bash
bash scripts/run_full_pipeline.sh          # ~10-20 min, no data needed
FAST=1 bash scripts/run_full_pipeline.sh   # ~3 min smoke run
```

This needs nothing downloaded — the cohort is generated procedurally.

### Individual stages

| Result | Command | Runtime |
|---|---|---|
| All ground-truth checks | `pytest` | ~3 min |
| Age regressor + age gaps | `python scripts/train_age_regressor.py --epochs 60` | ~8 min |
| Attribution + outlier artifacts | `python scripts/run_attribution_analysis.py` | ~6 min |
| Decomposition | `python scripts/run_residual_decomposition.py` | ~7 min |
| Full discovery pipeline | `python scripts/run_discovery_experiment.py` | ~8 min |

Runtimes are for an Apple Silicon laptop CPU. Add `--n-recordings` / `--epochs`
to trade fidelity for speed.

## Headline numbers on synthetic data

Reproduced with `--n-recordings 1500 --epochs 35 --seed 0`:

| quantity | value |
|---|---|
| Age regressor test MAE | 4.36 years (R² 0.926) |
| Constant-prediction baseline MAE | 17.93 years |
| Age gap SD | 4.73 years |
| Explained by known intervals (beyond demographics) | **50.0%** |
| Unexplained | **32.8%** |
| Dominant known feature | `qrs_duration_ms` (univariate R² +0.405) |

Component accuracy against constructed ground truth:

| component | metric |
|---|---|
| R-peak detection @500 Hz | sensitivity 1.000, PPV 1.000, max error 2 ms (1 sample) |
| R-peak detection @100 Hz | sensitivity 1.000, PPV 1.000, max error 10 ms (1 sample) |
| QRS duration vs truth | r = 0.995 @500 Hz, r = 0.687 @100 Hz |
| PR interval vs truth | r = 0.985 @500 Hz |
| QT interval vs truth | r = 0.987 @500 Hz |

> **These are synthetic-data numbers.** They demonstrate the machinery is
> correct. They are not findings about real ECGs. See
> [limitations.md](limitations.md) §1.

## Real data

```bash
bash scripts/download_data.sh                             # PTB-XL 1.0.3, ~1.7 GB
python -c "from ecg_discovery.data.ptbxl_dataset import summarise_ptbxl; \
           print(summarise_ptbxl('data/ptbxl'))"          # verify the download
python scripts/run_discovery_experiment.py --data ptbxl
```

PTB-XL is **CC BY 4.0** and fully open — no credentialing, data use agreement,
CITI training or account required (verified 2026-07-28). Attribution is required;
see `scripts/download_data.sh` for the citation.

### Verified download output (2026-07-29)

```
PTB-XL at data/ptbxl
  raw:      21,799 recordings, 18,869 patients
            293 with the age-300 sentinel (>89 years, dropped)
            age range 2-89
  filtered: 21,373 recordings, 18,495 patients
            age 18-89 (mean 59.8), 48% female
            repeat patients: 2,878 extra recordings
  diagnostic superclasses: {'NORM': 9375, 'MI': 5346, 'STTC': 5092,
                            'CD': 4732, 'HYP': 2588}
            403 recordings carry no diagnostic superclass
  official folds: train 17,090, val 2,132, test 2,151
```

Matching this exactly is the check that the download is complete and behaves as
documented. Note the 2,878 repeat recordings: patient-level splitting is a real
hazard on this dataset, not a hypothetical one.

### Headline numbers on PTB-XL

`python scripts/run_discovery_experiment.py --data ptbxl --official-split --epochs 60`
(~35 min; the 500 Hz view is loaded lazily for the test split only, so peak
memory stays near 2 GB rather than 6 GB)

| quantity | value |
|---|---|
| Age regressor test MAE | **7.32 years** (R² 0.694) |
| Test fold | 2,151 recordings (official fold 10) |
| Age gap SD | 9.40 years |
| Known features | 15 (timing, amplitude, axis, morphology) |
| Explained by demographics | 20.4% |
| Attributable to known measurements | **14.6%** |
| Unexplained | **65.0%** |
| Largest diagnostic ΔAUC | +0.002 (NORM), below the 0.020 threshold |

Sensitivity of the result to analysis choices — the project's central empirical
finding, and reproducible by varying only the flags shown:

| | 5 features, `patient_level_split`, 40 ep | 5 features, `--official-split`, 60 ep | 15 features, `--official-split`, 60 ep |
|---|---|---|---|
| attributable | 3.9% | 3.5% | 14.6% |
| unexplained | 66.8% | 75.8% | 65.0% |
| ΔAUC MI | +0.015 (sig.) | +0.007 (ns) | +0.001 (ns) |

The 5-feature configuration is recoverable by setting `known_features` in
`configs/validation_framework.yaml` back to heart rate, PR, QRS, QT and QTc.

### Attribution results on PTB-XL

```bash
# attribution with all three null controls        (~50 min)
python scripts/run_attribution_analysis.py --data ptbxl --official-split \
    --epochs 60 --with-controls --shuffled-control --n-attribution 400

# replication across seeds, per-lead, classical-P test   (~90 min)
python scripts/probe_p_wave_finding.py --seeds 0 1 2 --epochs 60 --ig-steps 128

# causal occlusion with width-matched controls            (~60 min)
python scripts/run_occlusion_test.py --seeds 0 1 2 --epochs 60 --n-recordings 1200
```

| quantity | value |
|---|---|
| P-wave attribution above amplitude null | **+0.041** (range +0.033 to +0.050), 3/3 seeds |
| Amplitude-null correlation | r = 0.929 (segment ranking is amplitude-driven) |
| Shuffled-label control MAE | 12.96 y vs 13.21 y baseline (learned nothing) |
| Classical P features explain | **0.7%** of the age gap |
| P-wave occlusion vs matched control | **+0.582 y** (range +0.468 to +0.733), 3/3 |
| QRS occlusion vs matched control | +7.348 y - inflated by distribution shift |
| T-wave occlusion vs matched control | +0.617 y |

Full interpretation and caveats in
[attribution_findings.md](attribution_findings.md).

## External replication cohort

```bash
bash scripts/download_chapman.sh          # or scripts/download_chapman.py for a subset
python -c "from ecg_discovery.data.chapman_dataset import summarise_chapman; \
           print(summarise_chapman('data/chapman'))"
python scripts/run_chapman_replication.py --limit 24000 --seeds 0 1 2 \
    --epochs 60 --ig-steps 128 \
    --ptbxl-checkpoint runs/attribution_analysis/<ts>/artifacts/model.pt
```

Chapman-Shaoxing-Ningbo is **CC BY 4.0**, fully open (verified 2026-07-30).
Note: PhysioNet's `get-zip` endpoint was measured at **4.4 kB/s** (6.5 days for
2.5 GB); the browser download or the per-file endpoint used by
`download_chapman.py` are far faster.

Verified summary:

```
  raw:      45,152 header files
  filtered: 42,997 recordings with a usable age and sex,
            age 18-89 (mean 60.7), 44% female
  native rate: 500 Hz, 10 s, 12 leads
  note: one recording per patient; no repeat visits
```

| quantity | PTB-XL | Chapman |
|---|---|---|
| model MAE | 7.32 y | 7.72-7.77 y |
| P attribution above amplitude null | +0.041 | **+0.059** |
| P occlusion vs matched control | +0.582 y | **+1.043 y** |
| seeds with CI excluding zero | 3/3 | 3/3 |

Transfer arm (PTB-XL model applied unchanged to Chapman): MAE 8.70, R² 0.456 —
substantial domain shift — yet P attribution +0.0291 [+0.0269, +0.0313] and
P occlusion +0.510 both remain significant.

## Configuration

All settings live in `configs/*.yaml` and are parsed into validated frozen
dataclasses. Unknown keys are rejected rather than ignored, so a typo cannot
silently fall back to a default. Values that encode physiology carry their
justification in comments.

Deliberate defaults worth knowing about:

- `sampling_rate_hz: 100` for the model, `interval_sampling_rate_hz: 500` for
  interval measurement. Measuring intervals at 100 Hz would understate what
  known features explain and bias toward a false discovery claim.
- `powerline_notch_hz: null` at 100 Hz — a 50 Hz notch is not representable when
  Nyquist *is* 50 Hz, and the 0.5–40 Hz bandpass already excludes that band.
- `t_wave_slope_fraction: 0.7`, raised from Pan–Tompkins' 0.5 on the evidence of
  a sweep across 45–180 bpm; costs no sensitivity anywhere tested.
- `edge_guard_ms: 50.0`, the exact point at which sensitivity and PPV both
  reach 1.000 at both sampling rates.

## Run directories

Each run writes `runs/<experiment>/<UTC timestamp>/` containing `config.json`
(resolved config, seed, git commit, package versions, argv), `metrics.jsonl`
(one record per epoch, flushed immediately so a killed run keeps partial
results), and `artifacts/` (checkpoint, per-split predictions with record ids,
splits, decomposition, diagnostic-link report, figures).

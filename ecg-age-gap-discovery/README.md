# ECG Age-Gap Discovery

**What does a medical AI actually know?** This repository trains a small 1D CNN
from scratch to predict chronological age from a 12-lead resting ECG, then asks
a harder question than "what did the model look at": *is the model's extra
signal anything cardiology does not already measure?*

Attribution can show that a model attends to the QRS complex. It cannot tell you
whether the model learned something new or quietly rediscovered that wide QRS
complexes mean older — a relationship cardiology has recorded for decades. This
project is built around closing that gap.

```bash
bash scripts/run_full_pipeline.sh     # every phase, end to end, no data needed
```

> ### Status
> **Phases 1–11 complete; 333 tests passing.** Everything is validated against a
> synthetic cohort with constructed ground truth.
>
> **No result here has been produced on real ECGs yet.** PTB-XL is not
> downloaded, so the real-data path — written and unit-tested against a
> miniature dataset with PTB-XL's schema — has not been run. That is the largest
> gap between this code and a paper, and it is tracked as
> [limitations.md §1](docs/limitations.md).

## What is and is not claimed

Predicting age from an ECG is **not new**, and this project does not claim it
is. The ECG age gap — predicted minus chronological age — is an active topic in
cardiology ML. Everything claimed as a contribution sits in the interpretability
and validation layer:

**1. Fiducial-segment attribution.** Raw per-sample saliency over a 12-lead ECG
is 12,000 numbers no clinician can agree or disagree with. Attribution is
aggregated into named cardiac segments (P wave, QRS complex, T wave, isoelectric
remainder) per lead, using a from-scratch delineation pipeline. This is an
*adaptation* of standard attribution to ECG's structure, not a new algorithm.

The care that makes it honest is the **width confound**: segments have very
different durations, so comparing summed attribution reports the T wave as
dominant in nearly every recording as an artefact of arithmetic. Both a
width-confounded `share` and a width-corrected `density` are always reported.

**2. A rediscovery-vs-discovery validation framework.** The age-gap residual is
regressed against independently computed classical intervals. Only the variance
those known measurements *fail* to explain is eligible to be called a discovery
candidate. The generalisable template — *regress the novel signal against
everything already known; only the residual is a candidate* — is the part
intended to outlive this dataset. See
[validation_methodology.md](docs/validation_methodology.md).

**3. Reporting whichever outcome occurs.** A null result is reported in the same
format as a positive one, enforced in code rather than promised in prose.

## The error asymmetry

Every weakness in this pipeline pushes the same way: **towards claiming a
discovery.** Noisy interval measurement, a small known-feature set, an explainer
too rigid to fit a real relationship, or an overfitted age model all shrink the
explained share and inflate the residual. Nothing produces a false negative.

So "fully explained" is strong evidence and "largely unexplained" is weak
evidence, and the honest reading of a large residual is *"we have not shown this
is known"* — never *"we have shown this is new"*. Two design decisions follow
directly: intervals are measured at **500 Hz** while the model trains at 100 Hz
(at 100 Hz, QRS duration correlates with truth at r = 0.69 versus 0.995), and
the model is deliberately undersized at 326,497 parameters so its age gap is
signal rather than memorisation.

## Provenance constraints

- **No pretrained weights, anywhere.** The regressor is randomly initialised and
  trained only on data in this repository. Verified by test: model parameters are
  bit-for-bit determined by the seed alone.
- **Attribution is hand-implemented.** Input × gradient and Integrated Gradients
  are written directly against PyTorch autograd, not imported from an
  interpretability library, because the aggregation step is claimed as a
  contribution and a claim about aggregation is weaker if the underlying numbers
  arrive from an opaque dependency.

## Results on synthetic data

| | |
|---|---|
| Age regressor test MAE | 4.36 years (R² 0.926), baseline 17.93 |
| Explained by known intervals | **50.0%** |
| Unexplained | **32.8%** |
| R-peak detection | sensitivity 1.000, PPV 1.000, error ≤ 1 sample |

Validated in **both** directions: a cohort whose age signal runs entirely
through QRS duration is correctly attributed to known features, and one carried
by T-wave morphology correctly survives as a large unexplained residual.

Again: these demonstrate the machinery is correct. They are not findings about
hearts.

## Data

[PTB-XL](https://physionet.org/content/ptb-xl/) 1.0.3 — 21,799 clinical 12-lead
ECGs from 18,869 patients, with age, sex and cardiologist-assigned SCP-ECG
statements. **Licence CC BY 4.0** and fully open: no credentialing, data use
agreement, CITI training or account required (verified 2026-07-28). ~1.7 GB.

```bash
bash scripts/download_data.sh
python scripts/run_discovery_experiment.py --data ptbxl
```

The full pipeline also runs on a **procedurally generated synthetic cohort**
with known fiducial points and known injected age effects, so every component is
validated against constructed ground truth before real data is involved.

## Layout

```
ecg_discovery/
  data/               synthetic generator, PTB-XL loader, preprocessing, splits
  signal_processing/  QRS detection, wave delineation, interval features
  models/             from-scratch 1D CNN age regressor
  training/           training loop, patient-level splits
  interpretability/   hand-implemented IG, fiducial-segment aggregation
  validation/         residual decomposition          <- core contribution
  discovery_experiments/  diagnostic-link test
  analysis/           figures
```

| Phase | Component | Status |
|---|---|---|
| 1 | Scaffolding, config system | done |
| 1.5 | Synthetic ECG generator | done |
| 2 | QRS detection | done |
| 3 | Wave delineation, interval features | done |
| 4 | From-scratch age regressor | done |
| 5 | Training loop, patient-level splits | done |
| 6 | Fiducial-segment attribution | done |
| 7 | Attribution analysis | code done, **not run on PTB-XL** |
| 8 | Residual decomposition | done |
| 9 | Discovery experiment | done, **synthetic only** |
| 10 | Visualization | done |
| 11 | Packaging and docs | done |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Documentation

- [validation_methodology.md](docs/validation_methodology.md) — the framework,
  in full; intended as the paper's methods section
- [limitations.md](docs/limitations.md) — what is not established, ordered by
  how much it constrains conclusions
- [related_work.md](docs/related_work.md) — what is prior art and what is
  claimed
- [reproducibility.md](docs/reproducibility.md) — exact commands, seeds,
  runtimes, expected numbers

## License

Apache-2.0. See [LICENSE](LICENSE). PTB-XL itself is CC BY 4.0 and is not
redistributed here.

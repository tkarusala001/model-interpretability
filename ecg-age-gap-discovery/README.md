# ECG Age-Gap Discovery

**What does a medical AI actually know?** This repository trains a small 1D CNN
from scratch to predict chronological age from a 12-lead resting ECG, and then
asks a harder question than "what did the model look at": *is the model's extra
signal anything cardiology does not already measure?*

> **Status: under construction.** Phase 1 (scaffolding + config system) is
> complete. This README is a skeleton and will be filled out in Phase 11 with
> install instructions, quickstart, and accurate result reporting. See
> "Build progress" below for what does and does not exist yet.

## What is and is not claimed

Predicting age from an ECG is **not new**, and this project does not claim it
is. The ECG age gap - the difference between predicted and chronological age -
is an active topic in cardiology ML. Everything claimed as a contribution here
sits in the interpretability and validation layer:

1. **Fiducial-segment attribution.** Raw per-sample saliency over a 12-lead ECG
   is not interpretable to a clinician. Attribution is aggregated into named
   cardiac segments (P wave, QRS complex, T wave, isoelectric remainder) per
   lead, using a from-scratch delineation pipeline. This is an *adaptation* of
   standard attribution to ECG's periodic multi-lead structure, not a new
   attribution algorithm.
2. **A rediscovery-vs-discovery validation framework.** The model's age-gap
   residual is regressed against independently computed classical ECG intervals.
   Only the variance those known measurements *fail* to explain is eligible to
   be called a discovery candidate. The generalisable template - *regress the
   novel signal against everything already known; only the residual is a
   candidate* - is the part intended to outlive this dataset.
3. **Reporting whichever outcome occurs.** If the age-gap signal turns out to be
   fully explained by known intervals, that is reported as a negative result,
   not reframed.

## Provenance constraints

Two constraints are enforced throughout, because the paper's claims depend on
them:

- **No pretrained weights, anywhere.** The age regressor is randomly
  initialised and trained only on the data in this repository. No
  `pretrained=True`, no checkpoint downloads, no ECG foundation models. Every
  number is attributable to this codebase.
- **Attribution is hand-implemented.** Input x gradient and Integrated
  Gradients are written directly against PyTorch autograd rather than imported
  from an interpretability library, because the fiducial-segment aggregation
  step is claimed as a methodological adaptation and needs to be inspectable end
  to end.

## Data

[PTB-XL](https://physionet.org/content/ptb-xl/) - 21,799 clinical 12-lead ECGs
from 18,869 patients, with age, sex and cardiologist-assigned SCP-ECG diagnostic
statements. Access mechanics and license terms are verified at build time in
`scripts/download_data.sh`; see that script and `docs/reproducibility.md`.

The full pipeline also runs end-to-end on a **procedurally generated synthetic
cohort** with known fiducial points and known injected age effects, so every
component can be validated against a constructed ground truth before real data
is involved.

## Build progress

| Phase | Component | Status |
|---|---|---|
| 1 | Scaffolding, config system | done |
| 1.5 | Synthetic ECG generator | done |
| 2 | QRS detection | done |
| 3 | Wave delineation, interval features | done |
| 4 | From-scratch age regressor | pending |
| 5 | Training loop, patient-level splits | pending |
| 6 | Fiducial-segment attribution | pending |
| 7 | Attribution analysis on PTB-XL | pending |
| 8 | Residual decomposition | pending |
| 9 | Discovery experiment | pending |
| 10 | Visualization | pending |
| 11 | Packaging and docs | pending |

## Development install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).

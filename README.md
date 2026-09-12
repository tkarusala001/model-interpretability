# ECG Age-Gap Discovery Validation

Code for "Unexplained Variance Is Not Evidence of Discovery: Validating
Interpretability-Derived Claims in ECG Age Models."

This repository implements a validation protocol for distinguishing genuine
discovery from rediscovery when interpretability is used to argue that a
model has learned something not already captured by known measurements.
It is applied here to an ECG-based age-gap model, but the protocol itself
(demographic adjustment, out-of-fold evaluation, rigid/flexible dual
explainers, and null-controlled attribution) is not specific to ECG.

## What's here

- **Model training**: from-scratch 1D residual CNN age regressor, no
  pretrained weights, cross-validated ensemble scoring.
- **Validation protocol**: regression of the age-gap residual against
  enumerated known covariates, with demographic adjustment and grouped
  out-of-fold evaluation.
- **Attribution analysis**: Integrated Gradients with completeness
  verification, three null baselines (amplitude, untrained model,
  shuffled labels), and causal occlusion testing.
- **Synthetic validation**: procedurally generated ECGs with known
  ground truth, used to test the protocol and every custom signal
  measurement before applying them to real data.

## Reproducing without data downloads

The full pipeline runs end to end on procedurally generated synthetic
data with no external downloads required. The test suite (366 tests)
verifies every ground-truth claim reported in the paper, including
signal-processing accuracy against constructed ground truth and the
protocol's behavior on synthetic cohorts with known answers.

```bash
pip install -r requirements.txt
pytest tests/
```

## Real-data experiments

Reproducing the PTB-XL and Chapman-Shaoxing-Ningbo results requires
downloading those datasets separately (both public, CC BY 4.0):

- PTB-XL: https://physionet.org/content/ptb-xl/
- Chapman-Shaoxing-Ningbo: https://physionet.org/content/ecg-arrhythmia/

See `ecg-age-gap-discovery/` for the main pipeline.

## License

Code released under CC BY 4.0, matching the datasets it processes.

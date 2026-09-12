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
> **Phases 1–11 complete; 350 tests passing.** Validated against a synthetic
> cohort with constructed ground truth, **and run end to end on PTB-XL**
> (21,373 recordings, 18,495 patients, official folds).
>
> Two results, pointing opposite ways, both replicated across independently
> trained models in **two cohorts on different continents** (PTB-XL, Germany;
> Chapman-Shaoxing-Ningbo, China):
> **(a)** the age-gap residual carries no clinically meaningful information
> beyond classical measurement, and an apparently significant finding dissolved
> under scrutiny;
> **(b)** the model nonetheless *causally depends* on atrial information that
> classical P-wave measurement does not capture — replicated across 6 models,
> 2 cohorts, and surviving model transfer between them. A falsifiable pointer,
> not a validated marker.
> See [attribution_findings.md](docs/attribution_findings.md).

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
regressed against independently computed classical measurements — 15 of them,
spanning timing, amplitude, axis and morphology. Only the variance they *fail*
to explain is eligible to be called a discovery candidate. The generalisable
template — *regress the novel signal against everything already known; only the
residual is a candidate* — is the part intended to outlive this dataset. See
[validation_methodology.md](docs/validation_methodology.md).

**2b. Null controls for segment attribution.** A segment attribution profile
can simply restate where the signal is largest. Three controls — signal
amplitude, an untrained model, and a model trained on permuted labels — separate
"what the model attends to" from "where the ECG is big". On synthetic data with
known ground truth, density-based attribution pointed at the *wrong wave*
without them.

**2c. A causal occlusion protocol.** Attribution shows where a model looks, not
what it needs. Each cardiac segment is replaced with its isoelectric baseline
and compared against a **width-matched isoelectric control**, so degradation
from "removing this structure" is separated from degradation from "removing this
many samples".

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

## Results on PTB-XL

Age regressor on the official test fold (2,151 human-over-read recordings):
**MAE 7.32 years, R² 0.694** — comparable to published ECG-age models, from
326,497 parameters trained from scratch on 17,090 recordings.

The headline result is what happens to the "discovery" as the analysis gets
more rigorous. Three runs, each stricter than the last:

| | A: 5 features<br>own split | B: 5 features<br>official folds | C: **15 features**<br>official folds |
|---|---|---|---|
| test MAE | 7.53 | 7.32 | 7.32 |
| attributable to known | 3.9% | 3.5% | **14.6%** |
| unexplained | 66.8% | 75.8% | **65.0%** |
| ΔAUC, NORM | +0.008 * | +0.012 * | **+0.002** * |
| ΔAUC, MI | **+0.015 \*** | +0.007 ns | **+0.001** ns |

**Every apparent effect shrank under scrutiny.** The MI association would have
been the headline of run A — "the unexplained ECG age gap improves myocardial
infarction detection". It failed to replicate on a different split (B), and
collapsed to +0.001 once the known-feature set included the amplitude, axis and
ST measurements a cardiologist actually uses (C).

Note what produced that collapse: **nothing about the model changed** between B
and C. Same architecture, same seed, same 7.32 MAE. Only the definition of
"already known" got more complete — which quadrupled the attributable share and
took the residual's apparent value to zero.

This is a direct empirical demonstration of the error asymmetry above: a thin
definition of "already known" inflates both the unexplained residual *and* the
apparent value of that residual. Every remaining effect is below the 0.020 AUC
clinical-relevance threshold fixed before any data was seen.

## The attribution result

The model places more attribution on the **P wave** than the ECG's own amplitude
structure predicts — **+0.041** share (range +0.033 to +0.050), **3/3 seeds**,
intervals excluding zero. It departs from an untrained model and from one trained
on permuted labels. It localises to **aVF, aVR, V1 and the limb leads**, where
P-wave morphology is clinically read, and is *negative* in V2/V3.

Classical P-wave measurement (P duration, P amplitude, PR) explains **0.7%** of
the age gap. And occlusion shows the dependence is causal: blanking the P wave
costs **+0.582 years** of MAE beyond a width-matched isoelectric control, 3/3
seeds.

> **The claim:** the atrial complex carries age information the model causally
> depends on, which P duration, P amplitude and PR interval do not capture. The
> QRS remains the dominant driver by roughly an order of magnitude.

**Replicated on an independent cohort.** On 24,000 recordings from three Chinese
hospitals sharing no data with PTB-XL, three freshly trained models give
attribution **+0.059** and occlusion **+1.043 y** — *larger* than PTB-XL's
+0.041 and +0.582. A PTB-XL model transferred unchanged degrades substantially
(MAE 7.32 → 8.70) yet **still shows the effect**, so it is a property of ECGs
rather than of one model or institution.

Falsifiable, and that is the point: measure P terminal force, P-wave dispersion,
notching or P area, and see whether the 0.7% closes.

Caveats travel with it — the gross segment ranking is amplitude-driven
(r = 0.929), P and T are causally comparable, and the QRS occlusion figure is
inflated by distribution shift. Full detail and every counter-argument in
[attribution_findings.md](docs/attribution_findings.md).

## Results on synthetic data

The machinery is validated separately against constructed ground truth, where
the answer is known:

| | |
|---|---|
| Age regressor test MAE | 4.36 years (R² 0.926), baseline 17.93 |
| Explained by known intervals | **50.0%** |
| Unexplained | **32.8%** |
| R-peak detection | sensitivity 1.000, PPV 1.000, error ≤ 1 sample |

Validated in **both** directions: a cohort whose age signal runs entirely
through QRS duration is correctly attributed to known features, and one carried
by T-wave morphology correctly survives as a large unexplained residual. So the
3.5% found on PTB-XL is not the framework failing to detect things — it detects
them when they are there.

## Data

Two independent cohorts, both CC BY 4.0 and fully open.

**Chapman-Shaoxing-Ningbo** ([PhysioNet](https://physionet.org/content/ecg-arrhythmia/1.0.0/))
— 45,152 twelve-lead ECGs from three Chinese hospitals, 10 s at 500 Hz, used as
the external replication cohort (`bash scripts/download_chapman.sh`).

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
| 7 | Attribution analysis + null controls | done, **run on PTB-XL** |
| 7b | Replication, per-lead, causal occlusion | done, **run on PTB-XL** |
| 7c | External replication, 2nd cohort | done, **Chapman-Shaoxing-Ningbo** |
| 8 | Residual decomposition | done, **run on PTB-XL** |
| 9 | Discovery experiment | done, **run on PTB-XL** |
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
- [attribution_findings.md](docs/attribution_findings.md) — the atrial result,
  its five lines of evidence, and everything it does not support
- [limitations.md](docs/limitations.md) — what is not established, ordered by
  how much it constrains conclusions
- [related_work.md](docs/related_work.md) — what is prior art and what is
  claimed
- [reproducibility.md](docs/reproducibility.md) — exact commands, seeds,
  runtimes, expected numbers


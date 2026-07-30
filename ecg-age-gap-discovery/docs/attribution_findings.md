# What the model attends to, and what it needs

*The Phase 7 attribution result on PTB-XL, the controls it survived, and the
claims it does and does not support.*

## Summary

A from-scratch ECG-age model (326,497 parameters, test MAE 7.32 years on
PTB-XL's official test fold) places more attribution on the **P wave** — atrial
depolarisation — than the ECG's own amplitude structure predicts. The effect
replicates across three independently trained models, localises to the leads
where P-wave morphology is clinically read, and survives a causal occlusion
test. Classical P-wave measurement explains 0.7% of the model's age gap.

**The claim:** the atrial complex carries age information the model causally
depends on, which P duration, P amplitude and PR interval do not capture. The
QRS complex remains the dominant driver by roughly an order of magnitude.

**What to do with it:** measure P terminal force, P-wave dispersion, notching,
or P-wave area, and test whether they close the 0.7%. That is a falsifiable
instruction, and it is the entire point of the result.

## Evidence

Five independent lines, each replicated across seeds 0, 1 and 2.

### 1. Attribution exceeds the amplitude null

Share of total absolute Integrated Gradients attribution falling in the P wave,
against the share of total signal *energy* in the P wave:

| seed | test MAE | P attribution | P amplitude | delta | 95% CI |
|---|---|---|---|---|---|
| 0 | 7.32 | 0.119 | 0.086 | +0.0334 | [+0.0310, +0.0357] |
| 1 | 7.46 | 0.136 | 0.086 | +0.0499 | [+0.0470, +0.0528] |
| 2 | 7.40 | 0.125 | 0.086 | +0.0389 | [+0.0360, +0.0418] |

Mean +0.0407, range [+0.0334, +0.0499]. **3/3 with intervals excluding zero.**

Computed at 128 integration steps. The Phase 7 exploratory run used 32 steps and
failed its completeness check on roughly 1 recording in 32; quadrupling the
resolution changed seed 0's estimate from +0.0334 to +0.0334, so the effect is
not an artefact of under-integration.

### 2. It departs from an untrained and a shuffled-label model

| segment | trained | amplitude | untrained | shuffled |
|---|---|---|---|---|
| P | **0.119** | 0.086 | 0.081 | 0.056 |
| QRS | 0.341 | 0.320 | 0.358 | 0.578 |
| T | 0.178 | 0.225 | 0.213 | 0.110 |
| other | 0.362 | 0.369 | 0.348 | 0.256 |

The shuffled-label control was trained on permuted ages and reached a test MAE
of 12.96 years against a constant-prediction baseline of 13.21 — it learned
nothing, confirming the control is valid rather than accidentally informative.

### 3. It localises to the leads where the P wave is read

Difference from the amplitude null, per lead:

| lead | delta | | lead | delta |
|---|---|---|---|---|
| **aVF** | **+0.0088** | | V5 | +0.0014 |
| **aVR** | **+0.0062** | | V4 | +0.0009 |
| **V1** | **+0.0052** | | V6 | −0.0006 |
| **aVL** | **+0.0047** | | **V2** | **−0.0018** |
| I | +0.0039 | | **V3** | **−0.0024** |
| II | +0.0037 | | | |
| III | +0.0033 | | | |

The elevation sits in the limb leads and V1 — where P-wave morphology is
clinically assessed (V1 for biphasic P and left atrial abnormality; II, III and
aVF for P-wave axis) — and is *negative* in V2 and V3, the mid-precordial leads
where the P wave is small and the QRS dominates. A uniform smear across all
twelve leads would have suggested an artefact; this pattern does not.

### 4. Classical P-wave measurement does not explain the age gap

Regressing the age gap on P duration, P amplitude and PR interval, out of fold,
above demographics:

```
incremental R²: 0.7%
  pr_interval_ms  +0.0179
  p_amplitude_mv  +0.0153
  p_duration_ms   +0.0126
```

Yet the model's per-recording P-wave attention **does** track those
measurements — r = +0.473 with P duration, r = +0.432 with P amplitude. So the
model is genuinely reading the atrial complex, not attending to noise that
happens to fall in that window, while extracting something those three numbers
do not represent.

### 5. Occlusion: the model needs it, not merely looks at it

Each segment replaced with its per-lead isoelectric baseline in every beat, on
1,200 test recordings, against a **width-matched isoelectric control** — the
same number of samples blanked from the electrically silent stretches. The
control matters because blanking any part of an ECG perturbs the input, and a
network may degrade simply because its input became unfamiliar.

| segment | mean Δ MAE vs control | range | per 100 samples | seeds positive |
|---|---|---|---|---|
| P | **+0.582 y** | [+0.468, +0.733] | +0.418 | **3/3** |
| QRS | +7.348 y | [+5.715, +8.471] | +5.460 | 3/3 |
| T | +0.617 y | [+0.504, +0.760] | +0.295 | 3/3 |

Occluding the P wave costs more than removing an equal number of isoelectric
samples, in every seed. The dependence is causal, not merely correlational.

## What this does not support

**The P wave is not the model's principal feature.** Per sample, the QRS is
worth ten times more. Attribution's *relative* elevation of the P wave above
the amplitude null does not make the P wave important in absolute terms, and any
write-up that elides this is overclaiming.

**P and T are causally comparable.** +0.418 versus +0.295 per 100 samples, and
they swap order between seeds (P 0.336 / T 0.364 in seed 0; P 0.527 / T 0.241 in
seed 1). The P wave is a real but secondary contributor, not a uniquely
important one.

**The QRS occlusion figure is inflated by distribution shift.** An ECG with its
QRS complexes blanked is wildly out of distribution — seed 1's MAE reached 16.5
years, *worse than guessing the mean* (13.21). That is "the input was
destroyed", not a clean effect size. The width-matched control handles removing
*samples*; it cannot handle removing the signal's most salient structure. The
P-wave figure is the trustworthy one precisely because it is a gentler
perturbation that leaves a recognisable ECG behind.

**The gross segment ranking is amplitude-driven.** Correlation between the
trained profile and the amplitude null is r = 0.929. The *departures* from that
null are the finding; the ranking of segments by attribution is not.

**Nothing here is clinical.** The result identifies where to look. It does not
establish causation in the biological sense, clinical utility, or replication in
another cohort. Single-dataset, single-institution, adults 18-89 only.

## Why this is a discovery *candidate* rather than a discovery

The framework in [validation_methodology.md](validation_methodology.md) exists
to stop exactly the inference this result invites. What has been shown is that a
model depends on atrial information not captured by three classical P-wave
measurements. What has *not* been shown is that no existing measurement captures
it — only that these three do not, and the known-feature set is demonstrably
incomplete (expanding it from 5 to 15 features quadrupled explained variance).

The honest status is: **a specific, falsifiable, cardiologist-testable pointer**,
which is the most that interpretability plus validation can deliver on its own.
Confirming it requires someone to measure the atrial features this project does
not compute.

## Reproducing

```bash
# attribution with all three null controls
python scripts/run_attribution_analysis.py --data ptbxl --official-split \
    --epochs 60 --with-controls --shuffled-control --n-attribution 400

# replication across seeds, per-lead breakdown, classical-P test
python scripts/probe_p_wave_finding.py --seeds 0 1 2 --epochs 60 --ig-steps 128

# causal occlusion with width-matched controls
python scripts/run_occlusion_test.py --seeds 0 1 2 --epochs 60 --n-recordings 1200
```

Roughly 50, 90 and 60 minutes respectively on a laptop CPU.

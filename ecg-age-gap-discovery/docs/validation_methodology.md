# Separating rediscovery from discovery

*How this project decides whether a model's signal is new — and what that
decision can and cannot support.*

## The problem with "the model found something"

A neural network predicts age from an ECG and is wrong by some amount. That
error — the **age gap** — is routinely interpreted as the heart being
biologically older or younger than the patient. Attribution methods can then
show *where in the signal* the model looked, and a figure showing concentrated
attention on the QRS complex reads as a finding.

None of that establishes anything new. The model may have learned nothing more
than "wide QRS complexes mean older", a relationship cardiology has recorded for
decades. It would still produce a perfectly good age gap; attribution would
still highlight the QRS complex; the figure would look identical. Interpretability
answers *where the model looked*. It is structurally incapable of answering
*whether what it saw was already known*.

This gap between the two questions is where claims of AI-discovered biomarkers
tend to live.

## The test

Regress the novel signal against everything already measurable, and keep only
what is left over.

```
age gap  ~  demographics                      →  R²_baseline
age gap  ~  demographics + known measurements →  R²_full

attributable to known measurements = R²_full − R²_baseline
unexplained                        = 1 − R²_full
```

Only the unexplained part is eligible to be called a discovery candidate. Even
then it is a *candidate*: it means "we have not shown this is already known",
which is a much weaker statement than "this is new".

For this project the known measurements are fifteen classical ECG quantities —
heart rate and its variability, P duration, PR, QRS duration, QT and QTc, the
P/R/T amplitudes, ST deviation, QRS and T axes, Sokolow-Lyon voltage, and
R-wave progression — computed by signal processing with no learning of any
kind, in `ecg_discovery/signal_processing/interval_features.py`.

The size of that list is not incidental. Expanding it from five to fifteen
quadrupled the share attributable to existing knowledge on real data, without
changing the model at all. Anyone applying this framework should treat the
known-feature set as a variable to be pushed, not a fixed input — see
"What it found on real data" below.

## Four things that make this a real test rather than a formality

### 1. Demographics are regressed out first

An age gap is *mechanically* correlated with age. Regression models pull towards
the mean, over-predicting the young and under-predicting the old, so the gap
carries an age trend that is an artefact of regression rather than a fact about
hearts. Known intervals also drift with age. Without adjustment, that shared age
dependence gets credited to the intervals.

The reported figure is therefore **incremental** R² over demographics alone.
`test_demographic_adjustment_prevents_crediting_a_regression_artefact` constructs
the failure directly: an age gap depending on age alone, with an interval that
is a pure function of age and carries no extra information. Unadjusted, the
interval appears to explain over 60% of the gap. Adjusted, it explains under 10%.

### 2. Everything is evaluated out of fold

In-sample R² rises with model flexibility whether or not any relationship
exists. A gradient-boosted model will "explain" pure noise given enough trees,
which would let anyone define a discovery out of existence by fitting harder.
All fits are cross-validated, grouped by patient where patient identifiers exist,
and the unexplained residual passed to Phase 9 is built from out-of-fold
predictions so it is not contaminated by its own fit.

A related trap surfaced during development and is worth recording: an
*overfitting baseline* is as dangerous as an overfitting full model. Fitting
several hundred trees to two demographic features on ~100 recordings produced an
out-of-fold R² of **−31%**, and subtracting that negative number inflated the
"attributable" figure to a meaningless 84%. Two fixes followed — internal early
stopping for the boosted explainer, and a floor of zero on the baseline, since a
model that loses to the mean explains *nothing*, not a negative amount.

### 3. Both a linear and a non-linear explainer, always reported

The linear model is interpretable. The boosted model catches curved or
interacting relationships a linear fit would miss — and a real but non-linear
dependence on a known interval, missed by a linear explainer, would sit in the
residual looking exactly like a discovery.

Both are always fitted and both are always reported, so there is no opportunity
to pick whichever number reads better after seeing results. The headline uses
whichever explains **most**, because the smallest credible unexplained residual
is the conservative claim.

### 4. A measurement can see what its definition cannot

This one emerged from validating the framework and changed how we read the
known-feature set.

In the synthetic cohort, T-wave **skew** is constructed so that it leaves the
true QT interval exactly unchanged — onset, offset, duration and amplitude are
all preserved, and `test_synthetic_ecg.py` verifies it. It should be perfectly
invisible to QT.

It is not. Measured against the latent factor driving skew, after removing the
shared age trend:

| quantity | true value | measured value |
|---|---|---|
| QT interval | −0.001 | **+0.330** |
| QRS duration | +0.000 | +0.004 |

QT is not measured by consulting its definition. It is measured by the tangent
method, which reads the slope of the T wave's descending limb — and skew changes
that limb's shape. So a morphology change invisible to the *definition* of a
known interval is partly visible to its *measurement*.

The framework consequently attributes part of that signal to rediscovery, which
is **correct**: a clinician measuring QT would partly detect the change too. The
lesson is a caution against a tempting shortcut in this kind of validation —
arguing "the known quantity is unchanged in principle, therefore this is new".
Only the measurement can be regressed out, and only the measurement is what a
clinician actually has.

## Which way the errors point

Every weakness in this pipeline pushes in the same direction: **towards claiming
a discovery.**

- Noisy interval measurement → known features explain less → unexplained share grows.
- A small known-feature set → less is attributable → unexplained share grows.
- An explainer too rigid to fit a real relationship → same.
- A model that overfits → its age gap is partly memorisation noise, which no ECG
  measurement can explain → same.

There is no corresponding mechanism that manufactures a false *negative*. A
finding of "fully explained" is therefore strong evidence, while "largely
unexplained" is weak evidence, and the asymmetry should govern how the result is
written up.

Two design decisions follow directly. Intervals are measured at **500 Hz** even
though the model trains at 100 Hz, because at 100 Hz one sample is 10 ms and
QRS duration correlates with truth at only r = 0.69 against r = 0.995 at 500 Hz
— and since explained variance goes as r², measuring coarsely would leave half
of a *genuinely known* effect sitting in the residual. And the model is
deliberately undersized (326,497 parameters) so its age gap is signal rather
than memorisation.

## What it found on real data

Applied to PTB-XL (2,151 held-out recordings, official folds), the framework
returned a negative result — and the *way* it got there is the point.

| | 5 known features | 5 features, different split | **15 known features** |
|---|---|---|---|
| attributable to known | 3.9% | 3.5% | **14.6%** |
| unexplained | 66.8% | 75.8% | **65.0%** |
| ΔAUC, MI | **+0.015, significant** | +0.007, ns | +0.001, ns |
| ΔAUC, NORM | +0.008, sig. | +0.012, sig. | +0.002, sig. |

The first column is a publishable positive finding: after Bonferroni correction
across five diagnostic superclasses, the unexplained ECG age-gap residual
improved myocardial-infarction detection. Reported on its own, it would read as
a discovered biomarker.

It survives neither a change of split nor a more complete definition of "already
known". Between the second and third columns **nothing about the model changed**
— same architecture, same seed, same 7.32-year test MAE. Only the known-feature
set grew, from five timing intervals to fifteen measurements including
amplitude, axis, ST deviation and R-wave progression. That quadrupled the
attributable share and took the residual's incremental value to +0.001 AUC.

Two conclusions follow, and the second is the more general one:

1. For this model and this dataset, the age gap carries no clinically
   meaningful information beyond classical ECG measurement.
2. **The apparent size of a "discovery" is a function of how thoroughly the
   analyst enumerated what was already known.** That is not a property of the
   biology. It is a property of the analysis, and it is invisible unless the
   known-feature set is varied deliberately, which is why doing so should be
   standard practice rather than an extra.

## How we know the framework works

It is validated against constructed ground truth, in both directions, because a
method that always reported "explained" would be useless and one that always
reported "unexplained" would manufacture discoveries.

| constructed cohort | required behaviour | result |
|---|---|---|
| Age signal carried **entirely by QRS duration** | attribute it to known features | recovers >70% of the explainable ceiling |
| Age signal carried **entirely by T-wave skew** | leave it unexplained | >50% unexplained |
| **Both** channels, equal variance | split roughly in half | 50.0% attributable, 32.8% unexplained |

The middle row is the one that matters most: it is the case where a naive method
would declare a discovery, and the case where a method that explains everything
would destroy a real one.

**The ceiling matters.** An age gap is not purely the injected signal — it is
the injected signal *plus the model's own prediction error*, and no ECG
measurement can explain a neural network's noise. Demanding 100% would be
demanding an explanation for something that is not there. The tests therefore
compare against what the ground-truth latent factor itself explains, which
separates "the framework works" from "the model trained well today".

## Generality

Nothing here is specific to ECG, and the ECG instantiation is best read as a
worked example. The template is:

> **Regress the novel signal against everything already known, evaluate out of
> fold, adjust for the covariates that mechanically induce correlation, use
> both a rigid and a flexible explainer, and treat only the residual as a
> discovery candidate — while recognising that every weakness in the procedure
> inflates that residual.**

It applies wherever a model is claimed to have found something new in a domain
that already has an established measurement toolkit: imaging biomarkers against
radiological features, genomic predictors against known variants, materials
models against established descriptors. The requirement is only that the domain
can enumerate what it already measures — which is exactly what makes a mature
domain a hard place to claim a discovery, and exactly what makes such a claim
worth something when it survives.

## What this framework does not do

- It does not prove novelty. It bounds it. An unexplained residual is the
  *ceiling* on a discovery claim, not the claim itself.
- It is only as good as the known-feature list, which is configuration, not
  physics. A richer list could only ever explain more. See
  [limitations.md](limitations.md).
- It says nothing about *causation*, clinical utility, or whether an
  unexplained signal is even stable across datasets.

Phase 9 takes the next step — asking whether the unexplained residual predicts
independently assigned diagnostic labels better than the known intervals alone —
and reports whichever answer it gets.

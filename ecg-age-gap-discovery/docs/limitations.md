# Limitations

Ordered by how much they constrain the conclusions, not by how comfortable they
are to state.

## 1. The unexplained residual is a ceiling, not a discovery

Phases 8 and 9 have now run on PTB-XL. Known measurements explain **14.6%** of
the age-gap residual beyond demographics, leaving **65.0%** unexplained. That
65% must not be read as a discovery, for three reasons that are all visible in
the data.

**The model's own error is in there.** Test MAE is 7.32 years and the age gap
has SD 9.40, so a substantial share of the "residual" is simply the model being
wrong. No ECG measurement can explain a neural network's error, and on real
data there is no ground truth that would let us separate the two.

**The number moves with things that have nothing to do with discovery.** A
better-trained model was *less* regressive, so demographics explained less of
its gap (29.3% → 20.7%) and the unexplained share went *up* (66.8% → 75.8%)
while the attributable share stayed flat. Unexplained fraction is not a
discovery metric.

**It shrinks as the known-feature set grows.** Going from 5 to 15 classical
measurements quadrupled the attributable share, 3.5% → 14.6%, with no change to
the model. There is no reason to think 15 is the ceiling; a full clinical
feature set would explain more still. See §3.

### What remains untested on real data

- **Delineation on pathological rhythms.** Atrial fibrillation, bundle branch
  block and paced rhythms get no special handling and are unvalidated. One
  real-data failure has already been found and fixed (see §6). This matters
  more than usual for the atrial finding in §1c: absent or disorganised P waves
  are exactly the case where P-wave delineation is least reliable.
- **Replication in another cohort.** Everything is PTB-XL, one institution.

## 1b. The diagnostic association did not replicate — and that is the finding

Across three progressively stricter analyses, every apparent link between the
unexplained residual and cardiologist labels shrank toward zero:

| | 5 features, own split | 5 features, official folds | 15 features, official folds |
|---|---|---|---|
| ΔAUC, MI | **+0.015 \*** | +0.007 ns | +0.001 ns |
| ΔAUC, NORM | +0.008 * | +0.012 * | +0.002 * |

The MI association was statistically significant after Bonferroni correction in
the first analysis. It would have been a reportable positive finding. It did not
survive a change of split, and did not survive a more complete definition of
"already known".

NORM remains nominally significant at +0.002 AUC, which is below the 0.020
clinical-relevance threshold fixed before any data was examined. With 2,129
recordings, statistical detectability at this effect size carries no clinical
meaning, and the reporting code flags it automatically.

**The honest conclusion is that this model's age gap contains no clinically
meaningful, independently verifiable information beyond what classical ECG
measurement already provides** — and that an analysis stopping one step earlier
would have concluded otherwise.

## 1c. Caveats on the atrial finding

The P-wave result ([attribution_findings.md](attribution_findings.md)) survived
five independent checks. Four things it still does not support:

- **The P wave is not the model's principal feature.** Per sample occluded, the
  QRS is worth ten times more (+5.46 vs +0.42 years per 100 samples).
  Attribution's *relative* elevation of the P wave above the amplitude null does
  not make it important in absolute terms.
- **P and T are causally comparable** (+0.418 vs +0.295 per 100 samples) and swap
  order between seeds. The P wave is a real but secondary contributor, not a
  uniquely important one.
- **The QRS occlusion figure is inflated by distribution shift.** An ECG with its
  QRS blanked is wildly out of distribution - seed 1 reached MAE 16.5 years,
  worse than guessing the mean. The width-matched control handles removing
  *samples*, not removing the signal's most salient structure. The P-wave figure
  is the trustworthy one because its perturbation is gentler.
- **The gross segment ranking is amplitude-driven** (r = 0.929 with the amplitude
  null). Only the *departures* are the finding.

And the framework's own caveat applies to it: we have shown three classical
P-wave measurements do not capture the dependence, not that none does. The
known-feature set is demonstrably incomplete (see §3).

## 2. Single-lead delineation, and a test fixture that cannot detect the problem

Clinical QRS duration is measured from the *earliest* onset in any lead to the
*latest* offset in any lead, because depolarisation reaches different parts of
the heart at different times. This project delineates on one lead (or a
multi-lead vector magnitude), which is biased **short**.

The synthetic cohort cannot detect this at all: its twelve leads are exact
linear projections of three shared components, so inter-lead timing dispersion
is **zero by construction**. Every delineation accuracy figure quoted here is
therefore silent on the single-lead bias — a limitation of the test fixture, not
evidence that the bias is absent. It must be re-checked on PTB-XL.

## 3. The known-feature set is still incomplete, and that inflates the residual

The decomposition now uses **15** classical measurements — heart rate, HRV, PR,
P duration, QRS duration, QT, QTc, R/T/P amplitudes, ST deviation, QRS and T
axes, Sokolow-Lyon voltage, and R-wave progression. That is far more than the
original five, and the effect of expanding it was large: attributable variance
went from 3.5% to 14.6% with no change to the model.

**That expansion is direct evidence this list is still not the ceiling.** A
full clinical feature set would include T-wave symmetry and notching, QRS
fragmentation, P-wave dispersion, ST slope morphology, late potentials, and
lead-specific patterns this project does not compute. Each would explain more,
and the unexplained share would fall further.

So the reported 65% is an **upper bound on what could be new**, not an
estimate of it — which is why the write-up says "we have not shown this is
already known" rather than "this is new".

## 4. Every error in the pipeline points toward claiming a discovery

There is no mechanism here that manufactures a false negative:

| weakness | effect | direction |
|---|---|---|
| noisy interval measurement | known features explain less | ⬆ unexplained |
| small known-feature set | less attributable | ⬆ unexplained |
| explainer too rigid | real relationships missed | ⬆ unexplained |
| overfitted age model | gap is partly memorisation noise | ⬆ unexplained |

"Fully explained" is therefore strong evidence; "largely unexplained" is weak
evidence. See [validation_methodology.md](validation_methodology.md).

## 5. A measurement can see what its definition cannot

T-wave skew is constructed to leave the true QT interval *exactly* unchanged.
Measured QT nonetheless correlates +0.330 with it (true QT: −0.001), because QT
is measured by the tangent method, which reads the shape of the T wave's
descending limb.

The framework handles this correctly — it credits the detectable part to
rediscovery, as it should. The lesson is that "the known quantity is unchanged
in principle" is **not** a valid argument for novelty. Only the measurement can
be regressed out, and only the measurement is what a clinician has.

## 6. Signal-processing limitations, individually measured

**One failure was found only on real data.** P-wave boundary search terminated
when the signal fell below 20% of P-wave height. A real P wave is ~0.1 mV, so
that threshold is ~0.02 mV — comparable to baseline noise. On clean synthetic
data the search terminated correctly; on PTB-XL it ran to the edge of the search
window in 22% of beats, producing P waves over 200 ms wide (physiologically
impossible) in 19.5% of recordings and PR intervals pinned at the window ceiling
in 16%. Since PR is a known feature, this had been inflating the unexplained
residual. Fixed with a sustained quiet-run requirement plus a physiological cap,
and pinned by a test that reproduces the failure synthetically at a raised noise
floor. It is recorded here because it is the clearest example in the project of
synthetic validation being necessary but not sufficient.

- **T-wave onset is the least accurate fiducial**, around 24 ms late. It feeds no
  known interval (QT uses T-*offset*) but shifts the Phase 6 T-segment boundary,
  moving a little early-T attribution into the isoelectric bucket. The synthetic
  T wave starts with zero amplitude *and* zero slope, making its onset
  mathematically undetectable, so this is a pessimistic estimate.
- **Very tall T waves are still mis-detected.** At ~83% of R-wave amplitude a
  synthetic T wave is nearly as steep as the QRS, and slope-plus-amplitude
  thresholding cannot separate two waves differing in neither: positive
  predictive value plateaus near 0.97 whatever `t_wave_slope_fraction` is set to.
- **No pathology handling.** Absent P waves (atrial fibrillation), fused P and T
  waves at high rates, bundle branch block and paced rhythms get no special
  treatment. Waves that cannot be found are reported missing rather than guessed.
- **Interval measurement at 100 Hz is inadequate** and the project does not do
  it: QRS duration correlates with truth at r = 0.69 at 100 Hz against r = 0.995
  at 500 Hz.

## 7. Interpretability limitations

- **Attribution says where the model looked, not whether it was right.** A large
  share on a segment is not evidence that the segment carries real information;
  that is what Phases 8 and 9 exist to test.
- **The Integrated Gradients baseline is a choice that changes the numbers.**
  The default (zero, which equals the per-lead training mean on standardised
  input) is argued for rather than derived. There is no canonical neutral ECG.
- **Segment boundaries are shared across all twelve leads**, so a lead whose QRS
  begins slightly earlier has a few samples counted as isoelectric.
- **Apple MPS silently returns zeroed gradients** at some batch sizes — 18 of 20
  recordings in one observed case, with no error raised. Attribution avoids MPS
  by default and verifies the completeness axiom. Anyone reusing this code on
  other hardware should keep that check enabled.

## 8. Scope of the claim

Even a positive Phase 9 result on real data would establish only that the
unexplained residual correlates with cardiologist labels **in PTB-XL**. It would
not establish causation, clinical utility, generalisation to another cohort or
recording device, or that the signal is stable over time. Single-dataset
generalisability is untested, and PTB-XL is one institution's data.

## 9. Cohort restrictions

Adults 18–89 only. PTB-XL records ages above 89 as the sentinel 300 (dropped
rather than clipped, since a fabricated age would corrupt the regression target
and every residual), and includes children from age 0, who are excluded because
an age model spanning paediatric and adult ECGs is modelling growth as much as
ageing. Conclusions do not transfer to either excluded group.

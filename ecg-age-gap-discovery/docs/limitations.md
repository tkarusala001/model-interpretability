# Limitations

Ordered by how much they constrain the conclusions, not by how comfortable they
are to state.

## 1. Nothing has been validated on real ECGs yet

**Every number produced by this repository so far comes from a synthetic
cohort whose answers we constructed ourselves.** PTB-XL has not been
downloaded, so the real-data path — written and unit-tested against a
miniature dataset with PTB-XL's schema — has never actually run.

This is the largest gap between what the code does and what a paper would
claim. Synthetic validation establishes that the machinery is correct: that
R peaks are found to within one sample, that measured intervals track
constructed ones, that attribution localises a known injected effect, and that
the decomposition behaves correctly in both directions. It establishes nothing
whatsoever about hearts.

Specifically pending real data:

- The Phase 9 result. On synthetic data the diagnostic labels are linked to the
  unexplained channel **by construction**, so the positive result there confirms
  only that the experiment can detect a link when one exists.
- Whether the QRS-detection and delineation parameters, tuned on smooth
  synthetic waveforms, hold up on clinical recordings with pathology, artefact
  and pacing.
- The headline decomposition number itself.

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

## 3. The known-feature set is small, and that inflates the discovery claim

The decomposition regresses the age gap against five interval measurements:
heart rate, PR, QRS duration, QT and QTc. A cardiologist has more —
amplitudes, axis, ST deviation, T-wave morphology, R-wave progression,
QRS fragmentation.

**A richer known-feature set could only ever explain more**, so the reported
unexplained share is an upper bound rather than an estimate. This is the single
most important caveat on the central result, and it is why the write-up says
"we have not shown this is already known" rather than "this is new".

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

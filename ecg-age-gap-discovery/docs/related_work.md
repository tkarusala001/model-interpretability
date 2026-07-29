# Related work, and what is and is not novel here

## What is not novel

**ECG age prediction.** Training a neural network to predict chronological age
from a 12-lead ECG is an established task. The best-known demonstration is
Attia et al., *Age and Sex Estimation Using Artificial Intelligence From
Standard 12-Lead ECGs* (Circulation: Arrhythmia and Electrophysiology, 2019),
which trained on several hundred thousand recordings at the Mayo Clinic and
reported a mean absolute error of roughly 7 years. Lima et al., *Deep neural
network-estimated electrocardiographic age as a mortality predictor* (Nature
Communications, 2021), reported that a large predicted-minus-chronological age
gap was associated with higher mortality. The model in this repository is much
smaller, trained from scratch on a much smaller dataset, and is not competitive
with either — nor is it meant to be. It exists so that its *errors* can be
interrogated.

**The "ECG age gap" as a concept.** The idea that the difference between
predicted and chronological age reflects cardiovascular ageing is an active
area, not something introduced here.

**Integrated Gradients.** The attribution method is Sundararajan, Taly & Yan,
*Axiomatic Attribution for Deep Networks* (ICML 2017). It is implemented
directly against PyTorch autograd rather than imported, but the algorithm is
theirs. Input × gradient is older still and standard. The *completeness* axiom
this project leans on as a correctness check comes from that paper.

**Pan–Tompkins QRS detection.** Pan & Tompkins, *A Real-Time QRS Detection
Algorithm* (IEEE Transactions on Biomedical Engineering, 1985). Implemented
from scratch here, with two documented deviations for offline use (zero-phase
filtering and a sampling-rate-agnostic derivative), but the algorithm is theirs.

**The tangent method for T-wave offset.** A conventional manual technique in
electrocardiography, generally attributed to Lepeschkin and Surawicz (1952).

**PTB-XL.** Wagner et al., *PTB-XL, a large publicly available
electrocardiography dataset* (Scientific Data, 2020), doi:10.1038/s41597-020-0495-6;
PhysioNet resource doi:10.13026/kfzx-aw45. Licensed **CC BY 4.0** (verified
2026-07-28 — note that some secondary sources, including this project's own
original brief, describe it as ODC-BY). The five diagnostic superclasses
(NORM, MI, STTC, CD, HYP) and the `strat_fold` split are the dataset's own.

## What is claimed as a contribution

### 1. Fiducial-segment attribution

An **adaptation** of standard attribution to the periodic, multi-lead structure
of the ECG — not a new attribution algorithm. Raw per-sample saliency over a
12-lead recording is 12,000 numbers that no clinician can agree or disagree
with. Aggregating into P wave, QRS complex, T wave and isoelectric remainder,
per lead, using a from-scratch delineation pipeline, turns it into a claim a
cardiologist can evaluate.

The specific methodological care claimed is the **width confound**: segments
have very different durations, so summing attribution within each and comparing
totals reports the T wave as dominant in almost every recording as an artefact
of arithmetic. Both a width-confounded `share` and a width-corrected `density`
are always reported, since neither answers the question alone.

Related in spirit: work on concept-level and region-level attribution in
medical imaging, which similarly projects pixel attributions onto
clinically-named structures. The ECG instantiation and the width correction are
what is offered here.

### 2. The rediscovery-vs-discovery validation framework

The part intended to outlive this dataset. Regress the model's novel signal
against everything the domain already measures, evaluate out of fold, adjust
for covariates that mechanically induce correlation, use both a rigid and a
flexible explainer, and treat only the residual as a discovery candidate.

We are not aware of this being applied as an explicit, testable protocol to
attribution-derived claims in ECG. Its components are individually standard —
incremental R², nested model comparison, cross-validation — and that is a
feature: the contribution is the *composition* and the argument for why each
piece is load-bearing, not new statistics.

Two findings from building it are contributions in their own right:

- **Error asymmetry.** Every weakness in such a pipeline inflates the
  unexplained residual. There is no mechanism producing a false negative, so
  "unexplained" is weak evidence while "explained" is strong.
- **A measurement can see what its definition cannot.** A change constructed to
  leave the true QT interval exactly unchanged nonetheless moves *measured* QT
  (+0.330 against −0.001), because QT is measured by reading the T wave's
  descending limb. "The known quantity is unchanged in principle" is not a valid
  argument for novelty.

### 3. Honest reporting of whichever outcome occurs

Phase 9 reports a null result in the same format as a positive one, and the
figure caption states the verdict either way. This is a commitment enforced in
code rather than a promise: `DiagnosticLinkReport.summary_text` has no path that
presents a null as anything other than a null, and the test suite checks both
branches.

## Positioning for Interp4Discovery

- **Topic 1 (methods for unfamiliar modalities):** fiducial-segment attribution.
- **Topic 2 (case studies surfacing verifiable knowledge):** the decomposition
  and the diagnostic-link experiment.
- **Topic 3 (validation and epistemology):** the framework itself, the error
  asymmetry, and the measurement-versus-definition distinction.
- **Topic 4 (failure cases and negative results):** whatever Phase 9 returns on
  real data, plus the documented failures in
  [limitations.md](limitations.md) — the T-wave discrimination plateau, the
  test fixture that structurally cannot detect single-lead bias, and the silent
  MPS gradient corruption.

## Bibliographic caution

Citations above are given by author, title, venue and year as understood at
time of writing. **Page numbers, volume numbers and exact DOIs beyond the two
PTB-XL identifiers verified against PhysioNet have not been checked**, and
should be confirmed against the publisher record before submission rather than
copied from here.

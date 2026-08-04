# Paper draft — structure, prose, and tables

**Unexplained Variance Is Not Novel Signal: Validating Interpretability-Derived
Discovery in ECG Age Models**

Target: NeurIPS 2026 Workshop on Interpretability for Discovery. 5 pages plus
unlimited appendix.

All numbers are final and reproducible from `docs/reproducibility.md`.

---

## Abstract (~180 words)

> Interpretability is increasingly used to argue that a model has discovered
> something. A neural network predicts a clinical variable, its residual error
> is found to be "unexplained" by standard measurements, attribution localises
> that residual to an interpretable structure, and a discovery is claimed. We
> show this inference is unsafe, and give a protocol that makes it testable.
>
> Training a 12-lead ECG age regressor from scratch on PTB-XL, we find that an
> apparently significant association between its age-gap residual and
> myocardial infarction (+0.015 AUC, significant after correction) fails to
> replicate across data splits and disappears entirely when the set of
> "already known" measurements is expanded from 5 to 15 — **with no change to
> the model**. Expanding that set quadrupled the variance attributable to
> existing knowledge, from 3.5% to 14.6%. The apparent magnitude of a discovery
> is therefore partly a property of the analyst's diligence, not the data.
>
> Applying the same protocol to attribution, we identify one finding that
> survives. The model places more attribution on the P wave than signal
> amplitude, an untrained model, or a shuffled-label model predicts; occluding
> the P wave degrades prediction beyond a width-matched control; and classical
> P-wave measurement explains 0.7% of the age gap. This replicates across six
> independently trained models in **two cohorts on different continents**, and
> survives transfer of a model between them. We report it as a falsifiable
> pointer - measure P terminal force or dispersion and test whether the gap
> closes - not as a validated marker.

---

## 1. Introduction (~0.6 pages)

**Paragraph 1 — the pattern.** Models match experts across domains and appear to
have learned regularities humans have not articulated. Interpretability offers
to read those out. In clinical ML the standard move is: model residual →
"unexplained by known measurements" → attribution → discovery claim.

**Paragraph 2 — the gap.** Attribution answers *where the model looked*. It is
structurally incapable of answering *whether what it saw was already known*. A
model that learned "wide QRS means older" — recorded in cardiology for decades —
produces an identical figure to one that found something new.

**Paragraph 3 — contributions.**
1. A validation protocol separating rediscovery from discovery: regress the
   novel signal against enumerated prior knowledge, out of fold, adjusting for
   covariates that mechanically induce correlation.
2. Null controls and a causal occlusion test for segment attribution on
   physiological signals.
3. An empirical demonstration that a significant finding dissolves under both,
   and a quantification of how much the conclusion depends on the analyst.
4. One finding that survives every check, stated as a falsifiable pointer.

---

## 2. Setup (~0.5 pages)

Keep short. PTB-XL 1.0.3 (CC BY 4.0), official folds, 17,090 / 2,132 / 2,151.
Adults 18–89; the age-300 anonymisation sentinel is dropped, not clipped.
A 326,497-parameter 1D CNN trained from scratch — **no pretrained weights** —
reaching **MAE 7.32 years, R² 0.694**, comparable to published ECG-age models.
Deliberately small so its residual is signal rather than memorisation.

One design point earns its space because it changes conclusions: intervals are
measured at **500 Hz** while the model trains at 100 Hz. At 100 Hz, QRS duration
correlates with ground truth at r = 0.687 against 0.995 at 500 Hz; since
explained variance goes as r², coarse measurement would leave half a *genuinely
known* effect in the residual.

---

## 3. The protocol (~1 page) — **the core contribution**

```
age gap ~ demographics                      → R²_baseline
age gap ~ demographics + known measurements → R²_full
attributable = R²_full − R²_baseline    unexplained = 1 − R²_full
```

Four components, each with a stated failure mode it prevents:

1. **Demographic adjustment.** An age gap is mechanically anti-correlated with
   age (regression to the mean). Unadjusted, an interval that is a pure function
   of age appears to explain >60% of the gap; adjusted, <10%.
2. **Out-of-fold evaluation, grouped by patient.** A boosted model explains
   noise in-sample. An *overfitting baseline* is equally dangerous: 300 trees on
   two features gave out-of-fold R² = −31%, inflating "attributable" to a
   meaningless 84%. Baseline R² is floored at zero — a model losing to the mean
   explains nothing, not a negative amount.
3. **A rigid and a flexible explainer, both always reported.** A real but
   non-linear dependence missed by a linear fit would sit in the residual
   looking exactly like a discovery.
4. **Measurements, not definitions.** T-wave skew constructed to leave the true
   QT interval *exactly* unchanged (r = −0.001) nonetheless moves *measured* QT
   (r = **+0.330**), because QT is read from the T wave's descending limb.
   "The known quantity is unchanged in principle" is not an argument for
   novelty; only the measurement can be regressed out, and only the measurement
   is what a clinician has.

**The error asymmetry** (highlight this — it is the paper's conceptual core).
Every weakness — measurement noise, a thin feature set, a rigid explainer, an
overfitted model — inflates the *unexplained* residual. Nothing produces a false
negative. Therefore "fully explained" is strong evidence and "largely
unexplained" is weak evidence, and most published inference has this backwards.

*Validation:* on a synthetic cohort with constructed ground truth the protocol
correctly attributes a QRS-mediated effect to known features and correctly
preserves a morphology effect invisible to timing (appendix).

---

## 4. Result 1: the discovery dissolves (~0.85 pages) — **Table 1**

| | 5 features, own split | 5 features, official folds | **15 features, official folds** |
|---|---|---|---|
| test MAE | 7.53 | 7.32 | 7.32 |
| attributable to known | 3.9% | 3.5% | **14.6%** |
| unexplained | 66.8% | 75.8% | **65.0%** |
| ΔAUC, MI | **+0.015 (sig.)** | +0.007 (ns) | **+0.001 (ns)** |
| ΔAUC, NORM | +0.008 (sig.) | +0.012 (sig.) | +0.002 (sig.) |

Column 1 is a publishable positive finding. It survives neither a change of
split nor a fuller enumeration of prior knowledge. **Between columns 2 and 3
nothing about the model changed** — same architecture, seed and test MAE; only
the known-feature set grew, from timing intervals to include amplitude, axis,
ST deviation and R-wave progression.

Effect-size threshold (0.020 AUC) fixed **before** any data was examined; the
reporting code flags sub-threshold results automatically. All remaining effects
fall below it.

*Two consequences.* For this model and dataset, the age gap carries no
clinically meaningful information beyond classical measurement. More generally,
**the apparent size of a discovery is a function of how thoroughly prior
knowledge was enumerated** — a property of the analysis, not the biology,
invisible unless the feature set is deliberately varied.

---

## 5. Result 2: a finding that survives (~1.25 pages) — **Tables 2 and 3**

Attribution requires its own controls: the QRS is the largest deflection on an
ECG, so gradients there are large regardless of what was learned. On synthetic
ground-truth data, density-based attribution pointed at the **QRS** when the
model's information was entirely in the T wave — only an amplitude null exposed
it.

**Table 2 — segment attribution against three nulls** (share of total)

| segment | trained | amplitude | untrained | shuffled labels |
|---|---|---|---|---|
| P | **0.119** | 0.086 | 0.081 | 0.056 |
| QRS | 0.341 | 0.320 | 0.358 | 0.578 |
| T | 0.178 | 0.225 | 0.213 | 0.110 |

Shuffled-label control reached MAE 12.96 against a 13.21 constant baseline — it
learned nothing, confirming the control is valid.

P-wave elevation above the amplitude null: **+0.041** (range +0.033 to +0.050),
**3/3 independently trained models**, all CIs excluding zero, unchanged at 4×
integration resolution. It localises to **aVF, aVR, V1 and the limb leads** —
where P-wave morphology is clinically read — and is *negative* in V2/V3.

**Table 3 — causal occlusion** (Δ MAE vs width-matched isoelectric control)

| segment | PTB-XL | seeds | Chapman | seeds |
|---|---|---|---|---|
| P | **+0.582 y** | 3/3 | **+1.043 y** | 3/3 |
| QRS | +7.348 y | 3/3 | +5.010 y | 3/3 |
| T | +0.617 y | 3/3 | +1.006 y | 3/3 |

**Table 4 — external replication** (Chapman-Shaoxing-Ningbo, 3 Chinese hospitals,
24,000 recordings, no data shared with PTB-XL)

| arm | MAE | P attribution vs amplitude null | P occlusion |
|---|---|---|---|
| fresh, seed 0 | 7.77 | +0.0493 [+0.0461, +0.0525] | +0.651 [+0.383, +0.918] |
| fresh, seed 1 | 7.76 | +0.0788 [+0.0741, +0.0835] | +1.149 [+0.818, +1.480] |
| fresh, seed 2 | 7.72 | +0.0486 [+0.0455, +0.0518] | +1.328 [+1.038, +1.618] |
| **transfer** (PTB-XL model, unchanged) | 8.70 | +0.0291 [+0.0269, +0.0313] | +0.510 [+0.040, ...] |

Every interval excludes zero. The effect is **larger** in the second cohort than
the first, which is the opposite of the cohort-specific-artefact pattern, and it
occurs in an arrhythmia-enriched population where P waves are *harder* to detect
(94.2% vs 98.0% of beats).

The transfer arm separates two things that are otherwise confounded. Applied
unchanged to Chinese recordings, the PTB-XL model degrades substantially
(MAE 7.32 -> 8.70, R² 0.694 -> 0.456) - a real domain shift. **The atrial
dependence survives it anyway.** The finding is therefore a property of ECGs,
not of one model or one institution's data: the model generalises poorly while
the finding generalises well.

Classical P-wave measurement (P duration, P amplitude, PR) explains **0.7%** of
the age gap — while the model's P-wave attention correlates r ≈ 0.45 with those
same measurements. It reads the atria, and extracts something those numbers do
not represent.

> **Claim.** The atrial complex carries age information the model causally
> depends on, not captured by P duration, P amplitude or PR interval. The QRS
> remains dominant by roughly an order of magnitude.

**Falsifiable, and that is the point:** measure P terminal force, P-wave
dispersion, notching or P area, and test whether the 0.7% closes.

*State the caveats here, not in a footnote:* the gross segment ranking is
amplitude-driven (r = 0.929) — only the departures are the finding; P and T are
causally comparable and swap order across seeds; the QRS occlusion figure is
inflated by distribution shift (an ECG without QRS complexes is far out of
distribution, and one seed fell below the mean predictor).

---

## 6. Limitations and conclusion (~0.5 pages)

Two cohorts, both single-institution-family; adults only; no clinical outcome
data. The atrial result shows *three* classical measurements do not capture the
dependence, not that none does — and the 5→15 experiment proves the
known-feature set is still incomplete. Attribution is not causation even with
occlusion: it establishes the model needs the information, not that the
information is biologically causal.

**Closing.** Interpretability tells you where a model looked. Deciding whether
that constitutes knowledge requires enumerating what was already known, and the
answer is sensitive to how well you do it. We give a protocol, demonstrate a
finding dissolving under it, and one surviving.

---

## Appendix

- Synthetic cohort with constructed ground truth; protocol validated in both
  directions (100% explained / large unexplained by construction)
- Signal-processing accuracy: R-peak detection sensitivity 1.000, PPV 1.000,
  ≤1 sample error at 100 and 500 Hz; interval correlations vs ground truth
- Per-lead attribution breakdown
- Apple MPS silently returning exactly-zero gradients for 18/20 recordings,
  caught only by the Integrated Gradients completeness axiom
- A P-wave delineation failure visible only on real data (22% of onsets pinned
  to the search-window edge, physiologically impossible 200 ms P waves)
- All 350 tests; every command in `docs/reproducibility.md`

---

## Writing notes

**Lead with Table 1.** It is the most quotable result and the one that changes
how the audience reads their own work.

**Do not lead with the P wave.** It is a pointer, not a validated discovery.
Putting it in the title or abstract's first line invites "that's not
established" as the reviewer's first reaction.

**Figures — two only, at this page count.** (1) the stacked variance
decomposition across the three configurations; (2) occlusion Δ MAE per segment
with CIs, both cohorts side by side. Both already generated by
`ecg_discovery/analysis/visualization.py`.

**Every caveat in the main text, none demoted to appendix.** The paper's
credibility is its central asset; naming a weakness before a reviewer does is
worth more than the space it costs.

---

## Responsible-use statement (MANDATORY — omission is grounds for desk rejection)

Place after the references; it does not count toward the five-page main-text
limit. Draft:

> **Responsible use.** This work studies claims that machine-learning models
> have discovered new clinical knowledge, and argues that such claims are
> systematically easier to make than to justify. The societal risk it addresses
> is direct: an unvalidated "AI-discovered biomarker" can influence clinical
> research priorities, funding, and eventually patient care, and a claim that
> does not replicate consumes those resources without benefit. We demonstrate
> concretely that an apparently significant association in our own analysis
> failed to survive a change of data split and a fuller enumeration of existing
> knowledge.
>
> The risk our own positive result carries is that it be over-read. We report a
> model's causal dependence on atrial information not captured by three
> classical P-wave measurements. This is a pointer for investigation, **not a
> clinical finding**: it establishes neither biological causation, nor clinical
> utility, nor that no existing measurement captures the signal. It must not
> inform patient care. Our mitigation is to state the claim's exact scope in the
> main text rather than the appendix, to report effect sizes alongside
> significance against a threshold fixed before analysis, and to release the
> code that would let a reader reproduce or refute every number.
>
> Both datasets are fully de-identified, publicly released under CC BY 4.0, and
> used within their licence terms; no new patient data was collected. We note
> that both cohorts are single-country and that our analysis is restricted to
> adults aged 18–89, so nothing here should be assumed to transfer to
> paediatric populations or to other health systems.

### Anonymisation checklist (double-blind)

- [ ] Strip author name, affiliation, acknowledgements
- [ ] Cite own prior work in the third person
- [ ] Search the PDF for personal names, GitHub and Hugging Face usernames
- [ ] Anonymous code mirror via **anonymous.4open.science**
- [ ] Scrub run-directory paths and git config from any released artifact
      (`runs/*/config.json` records `argv` and the git commit)

"""Rhythm statements as known features, and as a stratification variable.

WHY THIS EXISTS
---------------
The framework asks whether a model's signal is already known. Rhythm diagnosis
is about as classical as ECG knowledge gets - a cardiologist reads atrial
fibrillation off a trace immediately - yet the known-feature set used elsewhere
in this project is entirely continuous measurements: intervals, amplitudes,
axes. Rhythm is absent from it.

That gap matters for one result in particular. Occluding the P wave produces a
trace that resembles atrial fibrillation, AF prevalence rises steeply with age,
and the association between AF and age is entirely classical. So a model that
had learned nothing more than "no organised P wave means older" would produce
exactly the P-wave attribution and occlusion results this project reports, and
that would be rediscovery rather than discovery.

Two things follow, and this module supports both:

**Rhythm belongs in the known-feature set.** If the atrial dependence is
mediated by rhythm, adding rhythm indicators to the enumeration should absorb
it - which is the project's own instrument applied to its own surviving claim.

**Sinus-only stratification is the decisive test.** Within sinus rhythm every
recording has a P wave, so P-wave *presence* cannot carry the signal. An effect
that survives there is not an atrial-fibrillation detector.

SCOPE
-----
PTB-XL only. Chapman codes diagnoses as SNOMED-CT rather than SCP statements
and they do not map cleanly onto these categories, so a Chapman equivalent
needs its own mapping rather than a rename of this one.
"""

from __future__ import annotations

import ast
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "RHYTHM_FEATURE_NAMES",
    "RHYTHM_CODES",
    "parse_scp_codes",
    "rhythm_indicators",
    "is_sinus_rhythm",
]

#: SCP statements grouped into the categories a cardiologist would name first.
#: Deliberately coarse: the question is whether rhythm *class* explains the
#: atrial dependence, and splitting into a dozen rare categories would produce
#: indicators too sparse to fit.
RHYTHM_CODES: dict[str, tuple[str, ...]] = {
    "rhythm_atrial_fibrillation": ("AFIB",),
    "rhythm_atrial_flutter": ("AFLT",),
    "rhythm_paced": ("PACE",),
    # Sinus rhythm and its rate variants: the P wave is present and organised in
    # all of them, which is what the stratification below depends on.
    "rhythm_sinus": ("SR", "SARRH", "STACH", "SBRAD"),
    # Supraventricular rhythms other than AF/flutter, where atrial activity is
    # present but abnormal.
    "rhythm_other_supraventricular": ("SVTAC", "PSVT", "SVARR"),
}

RHYTHM_FEATURE_NAMES: tuple[str, ...] = tuple(RHYTHM_CODES)

#: Statements under which an organised P wave cannot be assumed. Used to define
#: the sinus-only subset by *exclusion* rather than by requiring an explicit
#: sinus code, because a recording carrying no rhythm statement at all is
#: unlabelled, not confirmed sinus.
_NON_SINUS = tuple(
    code
    for name, codes in RHYTHM_CODES.items()
    if name != "rhythm_sinus"
    for code in codes
)


def parse_scp_codes(raw: object) -> dict[str, float]:
    """PTB-XL's ``scp_codes`` cell as a dict, tolerating malformed rows.

    Matches the parsing used for diagnostic superclasses in ``ptbxl_dataset``:
    the column holds a stringified dict, and a row that will not parse is
    treated as carrying no statements rather than raising, since one bad row
    should not abort a cohort load.
    """
    if isinstance(raw, Mapping):
        return {str(k): float(v) for k, v in raw.items()}
    if not isinstance(raw, str):
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def rhythm_indicators(scp_codes: Iterable[object]) -> pd.DataFrame:
    """Multi-hot rhythm indicators, one row per recording.

    Parameters
    ----------
    scp_codes:
        The ``scp_codes`` column of PTB-XL's metadata, in recording order.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`RHYTHM_FEATURE_NAMES`, values 0 or 1. A recording may be
        all-zero: not every PTB-XL record carries a rhythm statement, and an
        absent statement is reported as absent rather than imputed to sinus.
        Rows are not mutually exclusive, since a record can carry more than one
        statement.
    """
    parsed = [parse_scp_codes(raw) for raw in scp_codes]
    return pd.DataFrame({
        name: [int(any(code in codes for code in group)) for codes in parsed]
        for name, group in RHYTHM_CODES.items()
    })


def is_sinus_rhythm(scp_codes: Iterable[object], require_explicit: bool = True) -> np.ndarray:
    """Boolean mask selecting recordings with an organised P wave.

    Parameters
    ----------
    require_explicit:
        When true (the default) a recording must carry a sinus statement to be
        included. When false, it is included unless it carries a non-sinus one.

        The default is the conservative choice for the question this mask
        exists to answer. Treating unlabelled recordings as sinus would
        readmit exactly the rhythms - undetected AF among them - that the
        stratification is meant to exclude, and the resulting subset would no
        longer support the claim that every recording in it has a P wave.
    """
    parsed = [parse_scp_codes(raw) for raw in scp_codes]
    sinus_codes = RHYTHM_CODES["rhythm_sinus"]
    non_sinus = np.array(
        [any(code in codes for code in _NON_SINUS) for codes in parsed]
    )
    if not require_explicit:
        return ~non_sinus
    explicit = np.array(
        [any(code in codes for code in sinus_codes) for codes in parsed]
    )
    return explicit & ~non_sinus

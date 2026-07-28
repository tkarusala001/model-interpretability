"""Gradient-based attribution, implemented directly against PyTorch autograd.

**No interpretability library is used here.** Input x gradient and Integrated
Gradients are short enough to write directly, and writing them directly matters
for this project: the fiducial-segment aggregation in
:mod:`ecg_discovery.interpretability.fiducial_attribution` is claimed as a
methodological adaptation, and a claim about how attribution is *aggregated* is
weaker if the attribution itself arrives from an opaque dependency. The formulas
below follow Sundararajan, Taly & Yan, "Axiomatic Attribution for Deep
Networks" (ICML 2017).

WHAT ATTRIBUTION ANSWERS HERE
-----------------------------
For one recording, attribution assigns every sample of every lead a number
saying how much that sample contributed to the predicted age. It answers "which
parts of this signal drove this prediction", and nothing more. In particular it
does not say the model is *right*, and a large attribution on a segment is not
evidence that the segment carries real physiological information. Establishing
that is what Phases 8 and 9 are for; this module only produces the input to
that argument.

Attribution is taken with respect to the waveform while sex is held fixed at
the patient's actual value, so the result describes the signal's contribution
for this patient rather than mixing in the demographic covariate.

CHOOSING A BASELINE - THE AWKWARD PART
--------------------------------------
Integrated Gradients measures the prediction's change along a straight path
from a *baseline* input to the real one, so every attribution is implicitly the
answer to "compared with what?". For images the convention is a black image.
For a physiological signal there is no equally obvious neutral, and the choice
genuinely changes the numbers, so it is exposed as a parameter and the default
is argued for rather than assumed:

``zero`` (default)
    The all-zeros tensor. This project's models consume per-lead z-scored
    input, so **zero is exactly the per-lead training-set mean**: the zero and
    mean-signal baselines coincide in normalised space, which removes what
    would otherwise be an awkward choice between them. Physiologically it reads
    as an absence of deflection - a flat trace - which is the closest ECG
    analogue of a black image.
``flatline``
    Each lead held at that recording's own isoelectric level. This asks "what
    did this patient's deflections contribute, relative to their own resting
    level", which removes any per-recording amplitude offset from the
    comparison. Preferable when working with unnormalised millivolt data.
``mean_signal``
    An explicitly supplied reference waveform, for instance the training-set
    average recording. Note that averaging real ECGs whose beats fall at
    different times largely cancels the beats, so this tends to approximate a
    flat trace anyway.

Whatever the choice, :func:`integrated_gradients` returns a **convergence
delta**: the amount by which the attributions fail to sum to the difference in
model output between the real input and the baseline. That sum is exact in the
continuous formulation (the *completeness* axiom), so a large delta means the
result cannot be trusted. It is reported rather than hidden, because silently
non-converged attributions look entirely normal.

THE COMPLETENESS CHECK IS NOT DECORATIVE
----------------------------------------
It has already caught a real, silent correctness failure in this project.
Running attribution on Apple's Metal (MPS) backend, gradients came back as
**exactly zero** for 18 of 20 recordings at batch size 20, while being correct
at batch size 8 and correct on CPU at every size. No exception was raised and
no warning printed; the attributions were simply wrong, and the resulting
segment summaries looked entirely reasonable. Only the completeness check
exposed it - a relative error of 0.90 against 0.004 on CPU.

Two safeguards follow from that, both on by default:

1. **MPS is avoided.** Unless a device is explicitly requested, a model sitting
   on MPS is evaluated on CPU instead. The models here are small enough that
   this costs little. CUDA is unaffected.
2. **Completeness is verified, not merely reported.** Integrated Gradients
   warns when the relative convergence error exceeds ``max_relative_error``.
   Attribution that violates its own defining axiom should never reach a
   figure in a paper unremarked.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "AttributionResult",
    "make_baseline",
    "input_x_gradient",
    "integrated_gradients",
]

BaselineKind = Literal["zero", "flatline", "mean_signal"]


@dataclass(frozen=True)
class AttributionResult:
    """Per-sample attributions for a batch of recordings.

    Attributes
    ----------
    attributions:
        ``(batch, n_leads, n_samples)`` array. Positive values pushed the
        predicted age up, negative values pushed it down.
    prediction:
        ``(batch,)`` model output for the real input.
    baseline_prediction:
        ``(batch,)`` model output for the baseline. The difference between this
        and ``prediction`` is what the attributions decompose.
    convergence_delta:
        ``(batch,)`` difference between the summed attributions and
        ``prediction - baseline_prediction``. Zero in exact arithmetic for
        Integrated Gradients; a large value means too few integration steps.
        ``NaN`` for methods with no completeness guarantee.
    method:
        Name of the method that produced this result.
    """

    attributions: np.ndarray
    prediction: np.ndarray
    baseline_prediction: np.ndarray
    convergence_delta: np.ndarray
    method: str

    @property
    def relative_convergence_error(self) -> np.ndarray:
        """Convergence delta as a fraction of the output difference it should match.

        The scale-free version of the completeness check: a delta of 0.5 years
        is negligible when the model moved 40 years from the baseline and
        alarming when it moved 1.
        """
        span = np.abs(self.prediction - self.baseline_prediction)
        return np.abs(self.convergence_delta) / np.maximum(span, 1e-8)


def make_baseline(
    waveform: Tensor, kind: BaselineKind = "zero", reference: Tensor | None = None
) -> Tensor:
    """Construct the baseline input that attributions are measured against.

    Parameters
    ----------
    waveform:
        ``(batch, n_leads, n_samples)`` real input.
    kind:
        ``zero``, ``flatline`` or ``mean_signal``. See the module docstring for
        what each one means and when it is appropriate.
    reference:
        Required for ``mean_signal``: a ``(n_leads, n_samples)`` or
        ``(batch, n_leads, n_samples)`` reference recording.
    """
    if kind == "zero":
        return torch.zeros_like(waveform)
    if kind == "flatline":
        # Median rather than mean: the isoelectric level is where the trace
        # spends most of its time, and a mean is pulled upward by the QRS spike.
        return waveform.median(dim=-1, keepdim=True).values.expand_as(waveform).contiguous()
    if kind == "mean_signal":
        if reference is None:
            raise ValueError("baseline kind 'mean_signal' requires a reference waveform")
        if reference.dim() == 2:
            reference = reference.unsqueeze(0)
        return reference.to(waveform.device, waveform.dtype).expand_as(waveform).contiguous()
    raise ValueError(
        f"unknown baseline kind {kind!r}; expected 'zero', 'flatline' or 'mean_signal'"
    )


def _resolve_attribution_device(model: nn.Module, requested: str | None) -> torch.device:
    """Pick the device attribution runs on, avoiding MPS unless asked explicitly.

    Apple's Metal backend has been observed in this project to return **exactly
    zero** gradients for some batch sizes, silently and without error (see the
    module docstring). Attribution is therefore moved off MPS by default. An
    explicit ``device`` argument overrides this, since the behaviour may be
    fixed in later PyTorch releases - but the default should be the one that
    cannot quietly produce a wrong figure.
    """
    if requested is not None:
        return torch.device(requested)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
    if device.type == "mps":
        return torch.device("cpu")
    return device


def _prepare(
    model: nn.Module,
    waveform: Tensor,
    sex: Tensor | None,
    device: str | None,
) -> tuple[nn.Module, Tensor, Tensor | None]:
    """Normalise shapes to a batch, detach, and place model and inputs together.

    Training may have run on GPU or MPS while the caller holds CPU arrays; a
    device mismatch otherwise surfaces as an obscure error from inside a
    convolution rather than as anything resembling the actual problem.
    """
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 3:
        raise ValueError(
            f"expected (batch, n_leads, n_samples) or (n_leads, n_samples), got "
            f"{tuple(waveform.shape)}"
        )
    if sex is not None:
        sex = sex.reshape(-1)
        if sex.shape[0] != waveform.shape[0]:
            raise ValueError(
                f"sex has {sex.shape[0]} entries but waveform batch is "
                f"{waveform.shape[0]}"
            )

    target = _resolve_attribution_device(model, device)
    model = model.to(target)
    dtype = torch.float32
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        pass

    waveform = waveform.detach().to(device=target, dtype=dtype)
    if sex is not None:
        sex = sex.detach().to(device=target, dtype=dtype)
    return model, waveform, sex


@torch.no_grad()
def _predict(model: nn.Module, waveform: Tensor, sex: Tensor | None) -> Tensor:
    return model(waveform, sex) if sex is not None else model(waveform)


def _gradient(model: nn.Module, waveform: Tensor, sex: Tensor | None) -> Tensor:
    """Gradient of the summed prediction with respect to the input waveform.

    Summing over the batch is safe because each recording's prediction depends
    only on its own row - true here because the model is evaluated in ``eval``
    mode, where batch normalisation uses fixed running statistics rather than
    batch statistics. In training mode recordings would influence one another
    through the batch statistics and per-sample attribution would be meaningless.
    """
    waveform = waveform.clone().requires_grad_(True)
    output = model(waveform, sex) if sex is not None else model(waveform)
    # `allow_unused` covers the case where the output does not depend on the
    # input at all - a fully saturated network, or a model that ignores some
    # leads. The mathematically correct gradient there is zero, and autograd
    # would otherwise raise rather than say so.
    (gradient,) = torch.autograd.grad(output.sum(), waveform, allow_unused=True)
    if gradient is None:
        return torch.zeros_like(waveform)
    return gradient


def input_x_gradient(
    model: nn.Module,
    waveform: Tensor,
    sex: Tensor | None = None,
    baseline_kind: BaselineKind = "zero",
    reference: Tensor | None = None,
    device: str | None = None,
) -> AttributionResult:
    """Attribution by elementwise input times gradient.

    The cheapest useful attribution: one backward pass, multiplying each input
    sample by the prediction's sensitivity to it. It is a first-order
    approximation and offers no completeness guarantee - the attributions do not
    sum to the change in model output - so ``convergence_delta`` is reported as
    NaN. It is included as a fast cross-check on Integrated Gradients: the two
    disagreeing sharply is a signal that the model is strongly non-linear over
    the path, and that the integration deserves more steps.

    Parameters
    ----------
    model:
        A trained regressor. Placed in ``eval`` mode for the duration.
    waveform:
        ``(batch, n_leads, n_samples)`` or ``(n_leads, n_samples)`` input.
    sex:
        Per-recording sex covariate, held fixed.
    baseline_kind, reference:
        Used only to report ``baseline_prediction``; the attribution itself does
        not depend on the baseline.
    device:
        Where to run. Defaults to the model's device, except that MPS falls back
        to CPU - see the module docstring on silent zero gradients.
    """
    was_training = model.training
    model.eval()
    try:
        model, waveform, sex = _prepare(model, waveform, sex, device)
        baseline = make_baseline(waveform, baseline_kind, reference)
        gradient = _gradient(model, waveform, sex)
        attributions = (waveform * gradient).detach()

        prediction = _predict(model, waveform, sex)
        baseline_prediction = _predict(model, baseline, sex)
    finally:
        model.train(was_training)

    return AttributionResult(
        attributions=attributions.cpu().numpy(),
        prediction=prediction.cpu().numpy(),
        baseline_prediction=baseline_prediction.cpu().numpy(),
        convergence_delta=np.full(waveform.shape[0], np.nan),
        method="input_x_gradient",
    )


def integrated_gradients(
    model: nn.Module,
    waveform: Tensor,
    sex: Tensor | None = None,
    baseline_kind: BaselineKind = "zero",
    reference: Tensor | None = None,
    n_steps: int = 64,
    step_batch: int = 16,
    device: str | None = None,
    max_relative_error: float | None = 0.05,
    max_absolute_error: float = 1.0,
) -> AttributionResult:
    r"""Attribution by Integrated Gradients (Sundararajan et al., 2017).

    Each input sample is credited with the gradient accumulated along a straight
    path from the baseline to the real input:

    .. math::

        \mathrm{IG}_i(x) = (x_i - x'_i)
            \int_0^1 \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i}\, d\alpha

    where :math:`x'` is the baseline. The integral is approximated by a Riemann
    sum using the **midpoint** rule - sampling :math:`\alpha` at
    :math:`(k + 0.5)/n` rather than at the interval edges - which converges
    faster than the left- or right-endpoint rules for the same number of model
    evaluations.

    Integrating along the path rather than reading a single gradient is what
    makes this preferable to input x gradient: a saturated model can have near
    zero gradient at the input itself while the feature was decisive, and the
    path integral still recovers it.

    Parameters
    ----------
    model:
        Trained regressor. Placed in ``eval`` mode for the duration, which also
        makes each recording's attribution independent of the others in the
        batch.
    waveform:
        ``(batch, n_leads, n_samples)`` or ``(n_leads, n_samples)`` input.
    sex:
        Per-recording sex covariate, held fixed along the path. Attribution is
        with respect to the waveform only.
    baseline_kind, reference:
        Baseline definition; see the module docstring on why this choice
        matters for ECG.
    n_steps:
        Riemann steps. More steps cost proportionally more compute and reduce
        the convergence delta; check that value rather than assuming 64 is
        enough for a given model.
    step_batch:
        How many path points to evaluate in a single forward pass, trading
        memory against speed. Does not affect the result: in ``eval`` mode
        batch normalisation uses fixed running statistics, so stacked path
        points cannot influence one another.
    device:
        Where to run. Defaults to the model's device, except that MPS falls
        back to CPU - see the module docstring on silent zero gradients.
    max_relative_error, max_absolute_error:
        Completeness tolerance, applied with the same logic as
        :func:`numpy.isclose`: a recording is flagged only when its convergence
        delta exceeds ``max_absolute_error + max_relative_error * span``, where
        span is that recording's ``|prediction - baseline_prediction|``.

        Both terms are needed. A purely relative test is meaningless for a
        recording whose prediction happens to sit near its baseline prediction:
        the span is then tiny, and a completely negligible delta of a fraction
        of a year divides out to a huge relative error. A purely absolute test
        would conversely miss real failures on recordings that moved a long way
        from the baseline. Set ``max_relative_error`` to ``None`` to disable the
        check entirely, though leaving it on is strongly recommended - this is
        what catches a backend silently returning wrong gradients.

    Returns
    -------
    AttributionResult
        Attributions plus the completeness check.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if step_batch < 1:
        raise ValueError(f"step_batch must be at least 1, got {step_batch}")

    was_training = model.training
    model.eval()
    try:
        model, waveform, sex = _prepare(model, waveform, sex, device)
        baseline = make_baseline(waveform, baseline_kind, reference)
        difference = waveform - baseline
        batch, n_leads, n_samples = waveform.shape

        alphas = (torch.arange(n_steps, dtype=waveform.dtype, device=waveform.device)
                  + 0.5) / n_steps
        accumulated = torch.zeros_like(waveform)

        for start in range(0, n_steps, step_batch):
            chunk = alphas[start : start + step_batch]
            n_chunk = int(chunk.numel())
            # Stack this chunk of path points into one forward/backward pass.
            points = (
                baseline.unsqueeze(0) + chunk.view(n_chunk, 1, 1, 1) * difference.unsqueeze(0)
            ).reshape(n_chunk * batch, n_leads, n_samples)
            chunk_sex = None if sex is None else sex.repeat(n_chunk)
            gradient = _gradient(model, points, chunk_sex)
            accumulated += gradient.reshape(n_chunk, batch, n_leads, n_samples).sum(dim=0)

        average_gradient = accumulated / n_steps
        attributions = (difference * average_gradient).detach()

        prediction = _predict(model, waveform, sex)
        baseline_prediction = _predict(model, baseline, sex)
    finally:
        model.train(was_training)

    # Completeness: the attributions must sum to the change in model output.
    summed = attributions.sum(dim=(1, 2))
    delta = (summed - (prediction - baseline_prediction)).cpu().numpy()

    result = AttributionResult(
        attributions=attributions.cpu().numpy(),
        prediction=prediction.cpu().numpy(),
        baseline_prediction=baseline_prediction.cpu().numpy(),
        convergence_delta=delta,
        method="integrated_gradients",
    )

    if max_relative_error is not None:
        span = np.abs(result.prediction - result.baseline_prediction)
        tolerance = max_absolute_error + max_relative_error * span
        failures = np.abs(delta) > tolerance
        if failures.any():
            worst = int(np.argmax(np.abs(delta) - tolerance))
            warnings.warn(
                f"Integrated Gradients failed its completeness check for "
                f"{int(failures.sum())} of {delta.size} recordings: worst "
                f"convergence delta {delta[worst]:+.3f} years against a "
                f"model output change of {span[worst]:.3f} years. The "
                f"attributions do not sum to what they should and cannot be "
                f"trusted. Try more n_steps; if the error does not fall, "
                f"suspect the compute backend - this project has seen Apple "
                f"MPS return silently zeroed gradients at some batch sizes. "
                f"Pass device='cpu' to rule that out.",
                RuntimeWarning,
                stacklevel=2,
            )

    return result

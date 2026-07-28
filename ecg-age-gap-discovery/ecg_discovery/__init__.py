"""Representation-level interpretability for ECG age-gap models.

This package trains a small convolutional network **from scratch** to predict
chronological age from a 12-lead resting ECG, and then asks what the model's
prediction error - the "ECG age gap" - actually encodes:

``ecg_discovery.signal_processing``
    From-scratch R-peak detection, wave delineation and classical interval
    measurement (heart rate, QRS duration, PR, QT). These are the *known*
    quantities the discovery claim is measured against.
``ecg_discovery.interpretability``
    Hand-implemented input x gradient and Integrated Gradients, plus the
    fiducial-segment aggregation that turns per-sample saliency into a
    cardiologist-legible P/QRS/T summary.
``ecg_discovery.validation``
    The residual decomposition that separates rediscovery from discovery: how
    much of the age-gap residual is a repackaging of intervals cardiology
    already measures, and how much is genuinely unexplained.

Provenance constraints (deliberate, and load-bearing for the paper's claims):
no pretrained weights are loaded anywhere, and no interpretability library is
used - the attribution methods are implemented directly against PyTorch
autograd.
"""

__version__ = "0.1.0"

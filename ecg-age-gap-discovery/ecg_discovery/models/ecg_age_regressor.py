"""A small 1D convolutional network that predicts age from a 12-lead ECG.

**No pretrained weights are used anywhere in this file, or anywhere in this
repository.** The network is randomly initialised and trained only on the data
in this project. That constraint is not incidental: the paper's claim is that
the model's age-gap residual reflects what *this* data taught *this* model, and
a pretrained initialisation would import knowledge from an unaudited source
into the middle of that claim.

WHAT THE MODEL DOES
-------------------
It reads a 10-second, 12-lead resting ECG and outputs a single number: the
patient's estimated age in years. Predicting age from an ECG is an established
task and nothing here is claimed as novel - the model exists so that its
*errors* can be studied. The difference between predicted and true age (the
"ECG age gap") is what Phases 6 to 9 interrogate.

WHY THIS ARCHITECTURE, AT THIS SIZE
-----------------------------------
PTB-XL provides roughly 18,900 patients. A large network trained from scratch on
a dataset that size will memorise it, and an overfitted model's age gap is
mostly memorisation noise - which would make every downstream analysis a study
of nothing. The default configuration is therefore deliberately small:
**326,497 parameters**, about 17 per patient, regularised with batch
normalisation and dropout.

*Receptive field.* The design target is that the deepest features see more than
one complete heartbeat. At 100 Hz a cardiac cycle is about 100 samples; four
stride-2 stages of 7-wide kernels give a receptive field of **637 samples, 6.4
seconds**, so a single output feature can express both beat *morphology* and the
rhythm across several beats. Heart rate is a real age signal, and a network that
could only see one beat at a time could not represent it.
:attr:`ECGAgeRegressor.receptive_field_samples` computes this for whatever
configuration is actually in use, rather than leaving the claim in a comment.

*Channel widths.* Widths grow (24 -> 80) as the time axis shrinks, the standard
trade that keeps per-layer cost roughly level while letting deeper layers hold
more abstract features. The specific widths were chosen by measurement rather
than convention: the widest stage dominates the parameter count, so narrowing
the stages from (32, 64, 64, 128) to (24, 40, 56, 80) halves the model - 683k
parameters down to 326k - while leaving the receptive field completely
unchanged. Capacity was the thing worth spending carefully; temporal context was
not.

*Global pooling.* Features are pooled over the whole recording by concatenating
average and maximum pooling. Average pooling captures what is typical of the
recording; max pooling captures the strongest single occurrence of a feature.
Both matter for an ECG, where a property may be sustained (rate) or intermittent
(an occasional abnormal beat). Pooling also makes the model independent of
recording length.

*Sex as an auxiliary input.* Sex is included because it substantially affects
ECG morphology - QT interval and QRS amplitude differ systematically between
men and women - and letting the model infer sex from the waveform would spend
capacity on something already known. The consequence for interpretation is that
the age gap is *conditional* on sex, which is consistent with Phase 8, where
sex is also a regression covariate. Set ``use_sex_input: false`` to ablate it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ecg_discovery.config import BackboneConfig

__all__ = ["ECGAgeRegressor", "ResidualBlock"]


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection.

    The skip connection lets gradients reach early layers directly, which is
    what makes a network of this depth trainable from random initialisation
    without careful warm-up. When a block changes the number of channels or the
    time resolution, the skip path gets a 1x1 convolution so the two paths can
    be added.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block to ``(batch, channels, time)`` input."""
        identity = self.skip(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.activation(out + identity)


class ECGAgeRegressor(nn.Module):
    """Predicts chronological age in years from a 12-lead ECG.

    Parameters
    ----------
    config:
        Architecture settings. Defaults are used if omitted.

    Examples
    --------
    >>> import torch
    >>> model = ECGAgeRegressor()
    >>> waveform = torch.randn(4, 12, 1000)     # 4 recordings, 10 s at 100 Hz
    >>> sex = torch.tensor([0.0, 1.0, 1.0, 0.0])
    >>> model(waveform, sex).shape
    torch.Size([4])

    Notes
    -----
    The forward pass is differentiable with respect to its waveform input, which
    Phase 6 relies on: the attribution methods take gradients of the predicted
    age with respect to individual signal samples.
    """

    def __init__(self, config: BackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or BackboneConfig()
        cfg = self.config

        padding = cfg.kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                cfg.in_channels, cfg.stem_channels, cfg.kernel_size,
                stride=1, padding=padding, bias=False,
            ),
            nn.BatchNorm1d(cfg.stem_channels),
            nn.ReLU(inplace=False),
        )

        stages: list[nn.Module] = []
        channels = cfg.stem_channels
        for stage_channels, stride in zip(cfg.stage_channels, cfg.stride_per_stage):
            for block_index in range(cfg.blocks_per_stage):
                stages.append(
                    ResidualBlock(
                        in_channels=channels,
                        out_channels=stage_channels,
                        kernel_size=cfg.kernel_size,
                        # Only the first block of a stage downsamples.
                        stride=stride if block_index == 0 else 1,
                        dropout=cfg.dropout,
                    )
                )
                channels = stage_channels
        self.stages = nn.Sequential(*stages)
        self.feature_channels = channels

        # Average and max pooling concatenated: what is typical of the recording,
        # and the strongest single occurrence of each feature.
        head_input = 2 * channels + (1 if cfg.use_sex_input else 0)
        self.head = nn.Sequential(
            nn.Linear(head_input, cfg.head_hidden),
            nn.ReLU(inplace=False),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Randomly initialise every parameter. Nothing is loaded from disk.

        Convolutions use Kaiming initialisation, which sets the initial weight
        scale so that signal variance is preserved through a ReLU network - the
        thing that makes a randomly initialised network of this depth trainable.

        The final layer gets its bias set to the population mean age and its
        weight scaled *small but not zero*. Zeroing it would make the model's
        first predictions exactly the prior, which looks tidy but is a trap: with
        a zero output weight the gradient with respect to every upstream
        parameter is also exactly zero, so the entire backbone is dead on the
        first step and only starts learning once the head has grown away from
        zero. A small non-zero scale gives predictions that begin near the prior
        while letting gradient reach the whole network immediately.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.xavier_uniform_(final.weight, gain=0.01)
        nn.init.constant_(final.bias, self.config.age_prior_mean)

    def forward(self, waveform: Tensor, sex: Tensor | None = None) -> Tensor:
        """Predict age in years.

        Parameters
        ----------
        waveform:
            ``(batch, 12, n_samples)`` tensor of ECG signal. Any length is
            accepted, since features are pooled over time.
        sex:
            ``(batch,)`` tensor of sex codes (0 = male, 1 = female). Required
            when the model was built with ``use_sex_input``; ignored otherwise.

        Returns
        -------
        torch.Tensor
            ``(batch,)`` tensor of predicted ages in years.
        """
        if waveform.dim() != 3:
            raise ValueError(
                f"expected (batch, channels, time) input, got shape "
                f"{tuple(waveform.shape)}"
            )
        if waveform.shape[1] != self.config.in_channels:
            raise ValueError(
                f"model expects {self.config.in_channels} leads but received "
                f"{waveform.shape[1]}"
            )

        features = self.stages(self.stem(waveform))
        pooled = torch.cat(
            [features.mean(dim=-1), features.amax(dim=-1)], dim=-1
        )

        if self.config.use_sex_input:
            if sex is None:
                raise ValueError(
                    "this model was configured with use_sex_input=True, so `sex` "
                    "must be provided; pass use_sex_input=False to ablate it"
                )
            sex_column = sex.reshape(-1, 1).to(dtype=pooled.dtype, device=pooled.device)
            if sex_column.shape[0] != pooled.shape[0]:
                raise ValueError(
                    f"sex has batch size {sex_column.shape[0]} but waveform has "
                    f"{pooled.shape[0]}"
                )
            pooled = torch.cat([pooled, sex_column], dim=-1)

        return self.head(pooled).squeeze(-1)

    # ---------------------------------------------------------------------- #
    # Introspection
    # ---------------------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def receptive_field_samples(self) -> int:
        """How many input samples influence one deepest-layer feature.

        Computed from the actual layer configuration rather than asserted in a
        comment, so the claim that the model sees more than one cardiac cycle
        stays true if the architecture is changed. Divide by the sampling rate
        for the span in seconds: at 100 Hz a cardiac cycle is ~100 samples.
        """
        cfg = self.config
        receptive_field = 1
        jump = 1

        def add_conv(kernel_size: int, stride: int) -> None:
            nonlocal receptive_field, jump
            receptive_field += (kernel_size - 1) * jump
            jump *= stride

        add_conv(cfg.kernel_size, 1)                      # stem
        for _, stride in zip(cfg.stage_channels, cfg.stride_per_stage):
            for block_index in range(cfg.blocks_per_stage):
                add_conv(cfg.kernel_size, stride if block_index == 0 else 1)
                add_conv(cfg.kernel_size, 1)
        return receptive_field

    @property
    def total_downsampling(self) -> int:
        """Input samples per output timestep."""
        return self.config.total_downsampling

    def extra_repr(self) -> str:
        return (
            f"parameters={self.n_parameters:,}, "
            f"receptive_field={self.receptive_field_samples} samples, "
            f"use_sex_input={self.config.use_sex_input}"
        )

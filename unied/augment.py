# -*- coding: utf-8 -*-
"""
Nonuniformity-aware perturbation augmentation (improvement scheme A).

During joint training (and optionally real fine-tuning) the degraded input is
additionally perturbed with degradation types that lie OUTSIDE the degree-3
polynomial assumption of the NUC module:

  1. local high-frequency residual  (white noise gated by a smooth spatial mask)
  2. column-wise stripe residual    (random per-column offsets, readout style)
  3. multiplicative smooth gain     (1 + smooth field, non-uniform gain)

The NUC residual branch is thus explicitly trained to absorb unmodelled
degradation (its L_res target is Y_pert - C), while the detection branch is
regularised to stay robust to whatever residual remains. All perturbations are
randomised per image and per type; nothing is applied at evaluation/inference.

Design notes:
  * amplitudes are in [0,1] image space (~2-10 gray levels / 255);
  * every type is applied with probability `p`; a batch element can receive
    several types at once;
  * outputs are clipped to [0,1].
"""
import torch
import torch.nn.functional as F


def _smooth_field(b, h, w, device, div: int = 8):
    """Standard-1 smooth random field on the (h, w) grid (bilinear upsample)."""
    lh, lw = max(2, h // div), max(2, w // div)
    z = torch.randn(b, 1, lh, lw, device=device)
    f = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
    return f / (f.std() + 1e-6)


class PerturbationAug(object):
    """
    y: [B, 3, H, W] float in [0, 1] -> perturbed y (same shape).
    Config (yaml `train.perturb`):
        enabled: bool   (default True)
        p: float        per-type probability in (0, 1)  (default 0.5)
        types: list     subset of ["hres", "stripe", "gain"]
        strength: float multiplier on the default amplitude ranges
    """

    DEFAULTS = {
        "hres": (0.02, 0.06),    # additive local high-frequency residual
        "stripe": (0.02, 0.08),  # additive column offsets
        "gain": (0.01, 0.05),    # multiplicative smooth gain
    }

    def __init__(self, enabled: bool = True, p: float = 0.5,
                 types=("hres", "stripe", "gain"), strength: float = 1.0):
        self.enabled = bool(enabled)
        self.p = float(p)
        self.types = tuple(types)
        self.strength = float(strength)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.p <= 0.0 or not self.types:
            return y
        b, _, h, w = y.shape
        out = y.clone()
        for i in range(b):
            for t in self.types:
                if torch.rand(1).item() > self.p:
                    continue
                s_min, s_max = self.DEFAULTS[t]
                amp = torch.empty(1).uniform_(s_min, s_max).item() * self.strength
                if t == "hres":
                    out[i] = self._hres(out[i : i + 1], amp, h, w)
                elif t == "stripe":
                    out[i] = self._stripe(out[i : i + 1], amp, h, w)
                elif t == "gain":
                    out[i] = self._gain(out[i : i + 1], amp, h, w)
        return out.clamp_(0.0, 1.0)

    # -- individual types ---------------------------------------------------
    def _hres(self, y: torch.Tensor, amp: float, h: int, w: int) -> torch.Tensor:
        """Local high-frequency residual: white noise localised by a smooth mask."""
        mask = _smooth_field(1, h, w, y.device)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        noise = torch.randn(1, 1, h, w, device=y.device)
        return y + amp * noise * mask

    def _stripe(self, y: torch.Tensor, amp: float, h: int, w: int) -> torch.Tensor:
        """Column-wise stripe residual (random per-column offsets)."""
        offsets = torch.randn(1, 1, 1, w, device=y.device) * amp
        return y + offsets

    def _gain(self, y: torch.Tensor, amp: float, h: int, w: int) -> torch.Tensor:
        """Multiplicative smooth gain nonuniformity."""
        g = 1.0 + amp * _smooth_field(1, h, w, y.device)
        return y * g

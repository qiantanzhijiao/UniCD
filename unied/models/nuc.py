# -*- coding: utf-8 -*-
"""
Prior- and data-driven nonuniformity correction (NUC) module (Sec. 3.2),
extended with a parametric + residual dual-decomposition bias model
(improvement scheme B).

Baseline (paper reproduction):
  Y --(2x downsample)--> Y_down --(3x3 conv, stride 2)--> F_proj (16ch)
     F_global = GBFE(F_proj)   : 2 Swin-Transformer layers, hidden dim 16 (RSTB style)
     F_local  = LBFE(F_proj)   : 2 spatial-attention modules in series
     F_fused  = concat(F_global, F_local)  (32ch)
     a_hat    = FC(AdaptiveAvgPool(Conv5(F_fused)))   -> K coefficients (10 for D=3)
  bias field  B_poly(a_hat) : degree-3 bivariate polynomial (Eq. 2)

Improvement B (residual decomposition):
  The real bias fields of IRBFD-syn contain non-polynomial content (column
  stripes, JPEG / high-frequency residuals): degree-3 LSQ fitting leaves a
  residual RMSE of about 5-34 gray levels that degree-6 cannot remove.
  We therefore model the bias field as
      B_total = B_poly(a_hat) + g * Up(B_res)
  where
      B_res   = ResidualBiasBranch(F_fused)   -- lightweight conv decoder
                producing a free-form residual bias map at 1/4 resolution,
                bilinearly upsampled to the full grid;
      g       = BiasGate(F_fused)             -- per-image gate in (0,1)
                trained by the *detection* losses (L_det / L_BR) so the
                residual correction keeps only detection-beneficial content.

  Config key `model.residual`:
      enabled: true/false   (false -> pure UniCD polynomial reproduction)
      gate:    true/false   (false -> g fixed to 1; ablation of the gate)
  With gate=false in stage-1 pre-training the residual branch is still fully
  trained by L_res; the gate is only learned once detection losses are on.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import poly


def _init_trunc_normal_(tensor, std=0.02):
    """Truncated-normal init without torch.nn.init.trunc_normal_ (torch<1.9 safe)."""
    nn.init.normal_(tensor, 0.0, std)
    with torch.no_grad():
        tensor.clamp_(-2.0 * std, 2.0 * std)


# ---------------------------------------------------------------------------
# Swin transformer primitives (C == 16, window size 8)
# ---------------------------------------------------------------------------
class WindowAttention(nn.Module):
    """Window-based multi-head self attention (W-MSA) with 8x8 windows."""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        _init_trunc_normal_(self.relative_position_bias_table, std=0.02)
        # relative index [W*W, W*W] without torch.meshgrid(indexing=) (torch<1.10 safe)
        ar = torch.arange(window_size)
        coords_h = ar.view(-1, 1).expand(window_size, window_size).reshape(-1)  # row of flattened pos
        coords_w = ar.view(1, -1).expand(window_size, window_size).reshape(-1)  # col of flattened pos
        rel_h = coords_h.view(-1, 1) - coords_h.view(1, -1)
        rel_w = coords_w.view(-1, 1) - coords_w.view(1, -1)
        rel_h = rel_h + (window_size - 1)
        rel_w = rel_w + (window_size - 1)
        relative_index = rel_h * (2 * window_size - 1) + rel_w
        self.register_buffer("relative_position_index", relative_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C]"""
        B, H, W, C = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        n_w_h, n_w_w = Hp // self.window_size, Wp // self.window_size

        x = (
            x.view(B, n_w_h, self.window_size, n_w_w, self.window_size, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, self.window_size * self.window_size, C)
        )  # [B*n_w, Ww*Wh, C]

        qkv = self.qkv(x).reshape(-1, self.window_size**2, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*n_w, heads, tokens, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size**2, self.window_size**2, -1
        )
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # [1, heads, tokens, tokens]
        attn = attn + rel_bias
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size**2, C)
        out = self.proj(out)
        out = out.view(B, n_w_h, n_w_w, self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if pad_h or pad_w:
            out = out[:, :H, :W, :].contiguous()
        return out


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (SwinIR style).
    x -> LN -> W-MSA -> LN -> MLP, plus a residual shortcut.
    """

    def __init__(self, dim: int = 16, window_size: int = 8, num_heads: int = 2, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        shortcut = x
        y = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        y = self.norm1(y)
        y = self.attn(y)
        y = self.norm2(y)
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y + shortcut


# ---------------------------------------------------------------------------
# Spatial attention (CBAM-style) used by the LBFE
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """Channel-pooled spatial attention: avg/max pool -> 7x7 conv -> sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg, mx], dim=1)
        att = torch.sigmoid(self.conv(att))
        return x * att


# ---------------------------------------------------------------------------
# GBFE / LBFE encoders
# ---------------------------------------------------------------------------
class GBFE(nn.Module):
    """
    Global bias field encoder: two Swin-Transformer layers (RSTB style),
    hidden channel dimension reduced to 16.
    """

    def __init__(self, dim: int = 16, window_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.rstbs = nn.Sequential(
            RSTB(dim, window_size, num_heads),
            RSTB(dim, window_size, num_heads),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rstbs(x)


class LBFE(nn.Module):
    """Local bias field encoder: two spatial attention modules in series."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.attn = nn.Sequential(
            SpatialAttention(kernel_size),
            SpatialAttention(kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


# ---------------------------------------------------------------------------
# Improvement B: residual bias branch & detection-driven gate
# ---------------------------------------------------------------------------
class ResidualBiasBranch(nn.Module):
    """
    Lightweight conv decoder producing the free-form residual bias map.

    F_fused (32ch at H/4 x W/4) -> 3 conv layers -> 1ch -> bilinear upsample
    to the full input resolution. Only a few thousand parameters.
    """

    def __init__(self, in_ch: int = 32, hidden: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        # zero-init the last conv so the branch starts as "no residual"
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, f_fused: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """f_fused: [B, C, H', W'] -> residual bias [B, 1, H, W]."""
        b = self.conv(f_fused)
        return F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False)


class BiasGate(nn.Module):
    """
    Per-image gate g = sigmoid(FC(GAP(F_fused))) in (0, 1).

    Trained by the detection losses (L_det, L_BR): the gate can shrink the
    residual correction when it does not help detection, implementing
    "detection-friendly residual pruning". Initialised near 1 so training
    starts close to the un-gated residual branch.
    """

    def __init__(self, in_ch: int = 32):
        super().__init__()
        self.fc = nn.Linear(in_ch, 1)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 2.0)  # sigmoid(2) ~ 0.88

    def forward(self, f_fused: torch.Tensor) -> torch.Tensor:
        """f_fused: [B, C, H', W'] -> gate [B, 1, 1, 1]."""
        g = torch.sigmoid(self.fc(f_fused.mean(dim=(2, 3)))).unsqueeze(-1).unsqueeze(-1)
        return g


# ---------------------------------------------------------------------------
# Full NUC predictor
# ---------------------------------------------------------------------------
class NUCPredictor(nn.Module):
    """
    Lightweight bias-field predictor with optional parametric + residual
    decomposition.

    Predicted coefficients are resolution independent: the polynomial bias
    field is reconstructed on the full-resolution grid by `poly.bias_from_coefs`.
    """

    def __init__(self, degree: int = poly.DEFAULT_DEGREE, hidden: int = 16,
                 window_size: int = 8, residual=None):
        super().__init__()
        self.degree = degree
        self.n_coefs = poly.num_coefficients(degree)
        residual = residual or {}
        self.use_residual = bool(residual.get("enabled", False))
        self.use_gate = bool(residual.get("gate", False)) and self.use_residual

        self.downsample = nn.AvgPool2d(2, stride=2)          # 2x downsample (paper)
        self.proj = nn.Conv2d(3, hidden, 3, stride=2, padding=1)  # -> 160x128 for 640x512 input
        self.gbfe = GBFE(dim=hidden, window_size=window_size)
        self.lbfe = LBFE()

        # Conv5: five 3x3 conv layers on the fused (2*hidden) channel map
        self.conv5 = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden, self.n_coefs)

        if self.use_residual:
            self.residual_branch = ResidualBiasBranch(in_ch=2 * hidden, hidden=hidden)
            if self.use_gate:
                self.gate_head = BiasGate(in_ch=2 * hidden)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: [B, 3, H, W] degraded image -> a_hat: [B, K] (compatibility)."""
        a_hat, _ = self._forward_full(y)
        return a_hat

    def _forward_full(self, y: torch.Tensor):
        """Returns (a_hat [B, K], f_fused [B, 2h, H/4, W/4])."""
        yd = self.downsample(y)
        f = self.proj(yd)
        f_global = self.gbfe(f)
        f_local = self.lbfe(f)
        f_fused = torch.cat([f_global, f_local], dim=1)
        f5 = self.conv5(f_fused)
        f5 = self.pool(f5).flatten(1)
        a_hat = self.fc(f5)
        return a_hat, f_fused

    def residual_bias(self, f_fused: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Full-resolution residual bias map [B, 1, H, W] (None if disabled)."""
        if not self.use_residual:
            return None
        return self.residual_branch(f_fused, h, w)

    def gate(self, f_fused: torch.Tensor):
        """Per-image gate [B, 1, 1, 1] (None if disabled)."""
        if not self.use_gate:
            return None
        return self.gate_head(f_fused)

    def predict_bias(self, y: torch.Tensor, basis: torch.Tensor,
                     h: int, w: int, gate_mode: str = "learn") -> torch.Tensor:
        """
        Predict the full-resolution bias field B_total = B_poly + g * Up(B_res).

        basis: [K, H, W]; returns [B, 1, H, W].
        gate_mode: 'learn' uses the learned per-image gate, 'fixed' forces g=1
        (used in stage-1 pre-training where no detection signal is present).
        """
        a_hat, f_fused = self._forward_full(y)
        b = poly.bias_from_coefs(a_hat, basis).unsqueeze(1)   # [B, 1, H, W]
        if self.use_residual:
            b_res = self.residual_bias(f_fused, h, w)          # [B, 1, H, W]
            if self.use_gate:
                g = self.gate(f_fused)                          # [B, 1, 1, 1]
                if gate_mode == "fixed":
                    g = torch.ones_like(g)
                b = b + g * b_res
            else:
                b = b + b_res
        return b

    def correct(self, y: torch.Tensor, basis: torch.Tensor,
                h: int, w: int, gate_mode: str = "learn") -> torch.Tensor:
        """Return the corrected image R = Y - B_total."""
        return y - self.predict_bias(y, basis, h, w, gate_mode=gate_mode)

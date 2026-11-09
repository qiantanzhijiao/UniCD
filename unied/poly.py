# -*- coding: utf-8 -*-
"""
Bivariate polynomial bias-field model (Eq. 2 of the UniCD paper).

The bias field B is modelled as a high-order bivariate polynomial

    B(x_i, y_j) = sum_{t=0}^{D} sum_{s=0}^{D-t} a_{t,s} x_i^t y_j^s

with D = 3 (10 coefficients, degree 3), coordinates normalised to [-1, 1].

The monomial order is lexicographic in (t, s):
    (0,0) (0,1) (0,2) (0,3) (1,0) (1,1) (1,2) (2,0) (2,1) (3,0)
which matches the construction loop used by `fit_polynomial` in
`unied/datasets/coef_fit.py`, so prediction targets are consistent.
"""
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DEGREE = 3


def num_coefficients(degree: int) -> int:
    """K = (D+1)(D+2)/2 coefficients for a bivariate polynomial of degree D."""
    return (degree + 1) * (degree + 2) // 2


def monomial_basis(
    height: int,
    width: int,
    degree: int = DEFAULT_DEGREE,
    normalized: bool = True,
    dtype=torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build the monomial basis stack for a HxW grid.

    Returns a tensor of shape [K, H, W] where K = num_coefficients(degree),
    and basis[k, i, j] = x_j^t * y_i^s for the k-th (t, s) pair.
    Coordinates are normalised to [-1, 1] (x spans columns, y spans rows).
    """
    yy = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
    xx = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
    # (H, W) coordinate grids without torch.meshgrid(indexing=) (torch<1.10 safe)
    y = yy.view(-1, 1).expand(height, width)   # y[i, j] = row coordinate
    x = xx.view(1, -1).expand(height, width)   # x[i, j] = col coordinate
    if normalized:
        x = 2.0 * x - 1.0
        y = 2.0 * y - 1.0
    terms = []
    for t in range(degree + 1):
        for s in range(degree + 1 - t):
            terms.append((x**t) * (y**s))
    return torch.stack(terms, dim=0)


def bias_from_coefs(
    coefs: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct the bias field from predicted coefficients.

    coefs: [..., K]  (any leading dims, e.g. batch)
    basis: [K, H, W]
    returns: [..., H, W]
    """
    return torch.einsum("...k,khw->...hw", coefs, basis)


def bias_from_coefs_np(
    coefs: np.ndarray,
    basis_np: np.ndarray,
) -> np.ndarray:
    """NumPy counterpart of bias_from_coefs (used by the coefficient fitting cache)."""
    return np.einsum("...k,kHW->...HW", coefs, basis_np)


def upsample_bias(bias_low: torch.Tensor, target_size, mode: str = "bilinear") -> torch.Tensor:
    """Upsample a low-resolution bias field to the full image resolution."""
    return F.interpolate(
        bias_low.unsqueeze(1), size=target_size, mode=mode, align_corners=False
    ).squeeze(1)

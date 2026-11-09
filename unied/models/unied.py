# -*- coding: utf-8 -*-
"""
UniCD: the union framework (Fig. 2 of the paper), extended with the
parametric + residual dual-decomposition NUC (improvement scheme B).

    degraded Y --NUC--> corrected R --detector--> predictions
    clear    C -------------------detector--> features  (only for joint training)

Losses:
  * L_cor   (Eq. 5): MAE between predicted and fitted polynomial coefficients
                    -- used for NUC pre-training.
  * L_res   (improvement B): L1 between the total predicted bias field
                    B_total = B_poly + g*B_res and the observed bias Y - C.
                    Trains the residual branch to capture the non-polynomial
                    part of the real bias field.
  * L_det   (Eq. 8): L_cls + L_reg + lambda * L_TEBS
  * L_TEBS  (Eq. 7): BCE between target mask M and backbone features at 4 stages.
  * L_BR    (Eq. 10): 1/4 * sum(1 - cos_sim(F_C^(i), F_R^(i))) at 4 stages.
  * L_uni   (Eq. 11): L_det + L_BR [+ lambda_res * L_res]

`model.residual.enabled=false` reproduces the original UniCD loss set exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import poly
from .nuc import NUCPredictor
from .head import Detector, build_targets, DetectionLoss


class TEBSLoss(nn.Module):
    """Target Enhancement and Background Suppression loss (Eq. 6, 7)."""

    def __init__(self):
        super().__init__()

    def forward(self, features: list, masks: list):
        """
        features: list of 4 backbone feature maps [B, C, H_i, W_i]
        masks:    list of 4 binary masks [B, H_i, W_i] (0 background, 1 target)
        """
        total = 0.0
        for f, m in zip(features, masks):
            p = f.mean(dim=1).sigmoid()          # [B, H, W]
            m = m.to(f.device).float()
            total = total + F.binary_cross_entropy(p, m, reduction="mean")
        return total / len(features)


class BRLoss(nn.Module):
    """Bias-Robust self-supervised loss (Eq. 9, 10): feature consistency
    between the corrected image R and the clear image C."""

    def __init__(self):
        super().__init__()

    def forward(self, feats_r: list, feats_c: list):
        total = 0.0
        for fr, fc in zip(feats_r, feats_c):
            fr = fr.flatten(1)                       # [B, C*H*W]
            fc = fc.flatten(1)
            cos = F.cosine_similarity(fc, fr, dim=1).mean()  # mean over batch
            total = total + (1.0 - cos)
        return total / len(feats_r)


class UniCD(nn.Module):
    def __init__(
        self,
        degree: int = poly.DEFAULT_DEGREE,
        nuc_hidden: int = 16,
        backbone_channels=(16, 32, 64, 128),
        fpn_ch: int = 64,
        lambda_tebs: float = 1.0,
        residual=None,
        lambda_res: float = 1.0,
    ):
        super().__init__()
        self.degree = degree
        self.n_coefs = poly.num_coefficients(degree)
        self.residual = residual or {}
        self.lambda_res = lambda_res
        self.nuc = NUCPredictor(degree=degree, hidden=nuc_hidden, residual=self.residual)
        self.detector = Detector(backbone_channels=backbone_channels, fpn_ch=fpn_ch)
        self.tebs_loss = TEBSLoss()
        self.br_loss = BRLoss()
        self.det_loss = DetectionLoss()
        self.lambda_tebs = lambda_tebs
        self._basis = None  # cached full-res basis, set by set_image_size

    @property
    def use_residual(self):
        return self.nuc.use_residual

    @torch.no_grad()
    def set_image_size(self, h: int, w: int, device: str = "cpu"):
        self._basis_key = (h, w, str(device))
        self._basis = poly.monomial_basis(h, w, self.degree, device=device)

    def _get_basis(self, h: int, w: int, device: str):
        key = (h, w, str(device))
        if getattr(self, "_basis_key", None) != key:
            self._basis_key = key
            self._basis = poly.monomial_basis(h, w, self.degree, device=device)
        return self._basis

    # ------------------------------------------------------------------
    # Correction (no detection)
    # ------------------------------------------------------------------
    def predict_coefs(self, y: torch.Tensor) -> torch.Tensor:
        return self.nuc(y)

    def _correct_and_bias(self, y: torch.Tensor, gate_mode: str = "learn"):
        """R = Y - B_total and B_total. y: [B,3,H,W]."""
        h, w = y.shape[2], y.shape[3]
        basis = self._get_basis(h, w, y.device)
        b = self.nuc.predict_bias(y, basis, h, w, gate_mode=gate_mode)
        return y - b, b

    def correct(self, y: torch.Tensor) -> torch.Tensor:
        """R = Y - B(Y). y: [B,3,H,W] -> [B,3,H,W]"""
        r, _ = self._correct_and_bias(y, gate_mode="learn")
        return r

    def bias_field(self, y: torch.Tensor) -> torch.Tensor:
        """B_total = B_poly + g*Up(B_res). [B,1,H,W]"""
        h, w = y.shape[2], y.shape[3]
        basis = self._get_basis(h, w, y.device)
        return self.nuc.predict_bias(y, basis, h, w, gate_mode="learn")

    # ------------------------------------------------------------------
    # Training forwards
    # ------------------------------------------------------------------
    def forward_nuc_pretrain(self, y: torch.Tensor, coef_gt: torch.Tensor,
                             c: torch.Tensor = None, lambda_res: float = None):
        """
        Stage-1: NUC only. L_cor = mean(|a_hat - a|) (Eq. 5) and, when the
        residual branch is enabled and the clear pair is available,
        L_res = L1(B_total, Y - C) on the residual part.
        The gate is forced to 1 (no detection signal at this stage).
        """
        if lambda_res is None:
            lambda_res = self.lambda_res
        h, w = y.shape[2], y.shape[3]
        basis = self._get_basis(h, w, y.device)
        a_hat, f_fused = self.nuc._forward_full(y)
        b = poly.bias_from_coefs(a_hat, basis).unsqueeze(1)
        if self.use_residual:
            b_res = self.nuc.residual_bias(f_fused, h, w)
            if self.nuc.use_gate:
                g = torch.ones_like(self.nuc.gate(f_fused))  # fixed at 1
                b = b + g * b_res
            else:
                b = b + b_res

        losses = {"cor": F.l1_loss(a_hat, coef_gt)}
        total = losses["cor"]

        if self.use_residual and c is not None:
            b_true = (y[:, :1] - c[:, :1]).detach()
            losses["res"] = F.l1_loss(b, b_true)
            total = total + lambda_res * losses["res"]
        else:
            losses["res"] = torch.zeros((), device=y.device)

        losses["total"] = total
        losses["a_hat"] = a_hat
        return losses

    def forward_joint(
        self,
        y: torch.Tensor,
        c: torch.Tensor,
        boxes: torch.Tensor,
        img_size: tuple,
        use_br: bool = True,
        lambda_tebs: float = 1.0,
        lambda_res: float = None,
        gate_mode: str = "learn",
    ):
        """
        Stage-2: joint training on IRBFD-syn (clear pairs available).
        y, c: [B,3,H,W]; boxes: [B, N, 4] xyxy (padded); returns loss dict.
        y may carry the stage-2 perturbation augmentation (scheme A): the
        residual supervision target below is then B_true = y_pert - C, i.e. the
        residual branch is explicitly trained to absorb unmodelled degradation.
        """
        if lambda_res is None:
            lambda_res = self.lambda_res
        r, b = self._correct_and_bias(y, gate_mode=gate_mode)
        cls_maps, reg_maps = self.detector(r)

        # detection targets on R
        det_targets = [build_targets(boxes[i], img_size) for i in range(boxes.shape[0])]
        # batch targets: build per-level concatenated tensors
        per_level = _collate_targets(det_targets, device=y.device)
        l_cls_l1, l_cls, l_reg = self.det_loss(cls_maps, reg_maps, per_level)

        # TEBS loss on R features (4 backbone stages)
        feats_r = self.detector.backbone(r)
        masks = [tgt["mask"] for tgt in per_level]
        l_tebs = self.tebs_loss(feats_r, masks)

        l_det = l_cls + l_reg + lambda_tebs * l_tebs

        losses = {"total": l_det, "det": l_det, "cls": l_cls, "reg": l_reg, "tebs": l_tebs}

        if use_br and c is not None:
            feats_c = self.detector.backbone(c)
            l_br = self.br_loss(feats_r, feats_c)
            losses["br"] = l_br
            losses["total"] = l_det + l_br

        if self.use_residual and c is not None:
            b_true = (y[:, :1] - c[:, :1]).detach()
            losses["res"] = F.l1_loss(b, b_true)
            losses["total"] = losses["total"] + lambda_res * losses["res"]
        else:
            losses["res"] = torch.zeros((), device=y.device)

        return losses

    def forward_det_only(self, y: torch.Tensor, boxes: torch.Tensor, img_size: tuple):
        """Stage-3: detection-only fine-tuning on IRBFD-real (no clear, no BR)."""
        r = self.correct(y)
        cls_maps, reg_maps = self.detector(r)
        det_targets = [build_targets(boxes[i], img_size) for i in range(boxes.shape[0])]
        per_level = _collate_targets(det_targets, device=y.device)
        l_cls_l1, l_cls, l_reg = self.det_loss(cls_maps, reg_maps, per_level)
        feats_r = self.detector.backbone(r)
        masks = [tgt["mask"] for tgt in per_level]
        l_tebs = self.tebs_loss(feats_r, masks)
        l_det = l_cls + l_reg + self.lambda_tebs * l_tebs
        return {"total": l_det, "det": l_det, "cls": l_cls, "reg": l_reg, "tebs": l_tebs}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect(self, y: torch.Tensor, score_thr: float = 0.05):
        """
        Correct + detect. y: [B,3,H,W]. Returns list of (boxes, scores) per image.
        """
        from .head import decode_predictions

        r = self.correct(y)
        cls_maps, reg_maps = self.detector(r)
        results = []
        for i in range(y.shape[0]):
            boxes, scores = decode_predictions(
                [m[i : i + 1] for m in cls_maps],
                [m[i : i + 1] for m in reg_maps],
                score_thr=score_thr,
            )
            results.append((boxes.cpu(), scores.cpu()))
        return results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _collate_targets(targets, device):
    """
    targets: list of B lists, each inner list has 4 level dicts.
    Return list of 4 level dicts with tensors of shape [B, H, W] / [B, 4, H, W].
    """
    n_levels = len(targets[0])
    out = []
    for l in range(n_levels):
        cls = torch.stack([t[l]["cls"].to(device) for t in targets])
        reg = torch.stack([t[l]["reg"].to(device) for t in targets])
        mask = torch.stack([t[l]["mask"].to(device) for t in targets])
        out.append({"cls": cls, "reg": reg, "mask": mask})
    return out

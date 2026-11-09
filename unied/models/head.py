# -*- coding: utf-8 -*-
"""
Anchor-free multi-scale detection head (FCOS-lite).

The UniCD paper keeps "the classification and regression losses consistent
with the baseline method" (DANet) without specifying them. We therefore use a
standard, minimal, trainable head:

  * classification : focal loss on a single-class inside-map
  * regression     : smooth-L1 on per-pixel distances (l, t, r, b) to the box,
                     normalised by the feature-map stride

Predictions are made on a 4-level pyramid (strides 4, 8, 16, 32), which covers
the multi-scale UAV targets (50 m - 2 km) of the IRBFD benchmark.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCOSHead(nn.Module):
    """Shared head for all pyramid levels. Single class (UAV)."""

    def __init__(self, in_ch: int = 64, num_convs: int = 2):
        super().__init__()
        convs = []
        for _ in range(num_convs):
            convs.append(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False))
            convs.append(nn.GroupNorm(8, in_ch))
            convs.append(nn.ReLU(inplace=True))
        self.shared = nn.Sequential(*convs)
        self.cls = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.reg = nn.Conv2d(in_ch, 4, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        f = self.shared(x)
        return self.cls(f), self.reg(f)


class FPN(nn.Module):
    """Lightweight top-down FPN over the 4 backbone stages (C=64)."""

    def __init__(self, in_channels, out_ch: int = 64):
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, out_ch, 1) for c in in_channels]
        )
        self.out = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, feats: list):
        lat = [lateral(f) for lateral, f in zip(self.laterals, feats)]
        out = [lat[-1]]
        for i in range(len(lat) - 2, -1, -1):
            up = F.interpolate(out[0], size=lat[i].shape[-2:], mode="nearest")
            out = [lat[i] + up] + out
        return [self.out(p) for p in out]  # P2..P5


class Detector(nn.Module):
    """Backbone + FPN + FCOS head. Returns pyramid (cls, reg) maps."""

    strides = (4, 8, 16, 32)

    def __init__(self, backbone_channels=(16, 32, 64, 128), fpn_ch: int = 64):
        super().__init__()
        from .backbone import MultiScaleBackbone

        self.backbone = MultiScaleBackbone(channels=backbone_channels)
        self.fpn = FPN(backbone_channels, fpn_ch)
        self.head = FCOSHead(fpn_ch)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        pyr = self.fpn(feats)
        outs = [self.head(p) for p in pyr]
        cls_maps = [o[0] for o in outs]   # [B,1,Hi,Wi] x4
        reg_maps = [o[1] for o in outs]   # [B,4,Hi,Wi] x4
        return cls_maps, reg_maps


# ---------------------------------------------------------------------------
# Targets & losses
# ---------------------------------------------------------------------------
def build_targets(
    boxes: torch.Tensor,  # [N, 4] xyxy in absolute pixels
    img_size: tuple,      # (H, W)
    strides: tuple = (4, 8, 16, 32),
):
    """
    Build classification/regression targets for the FCOS head.

    Returns per-level dicts: {'cls': [H,W], 'reg': [4,H,W], 'mask': [H,W]}
    where mask selects pixels that lie inside at least one GT box.
    """
    H, W = img_size
    targets = []
    for s in strides:
        h, w = H // s, W // s
        cls = torch.zeros(h, w, dtype=boxes.dtype)
        reg = torch.zeros(4, h, w, dtype=boxes.dtype)
        mask = torch.zeros(h, w, dtype=torch.bool)
        for box in boxes:
            x1, y1, x2, y2 = box.float().clamp(min=0, max=W).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            col0 = int(x1 // s)
            col1 = int(min(x2 // s, w - 1))
            row0 = int(y1 // s)
            row1 = int(min(y2 // s, h - 1))
            for r in range(row0, row1 + 1):
                for c in range(col0, col1 + 1):
                    cx = (c + 0.5) * s
                    cy = (r + 0.5) * s
                    l, t = cx - x1, cy - y1
                    rgt, b = x2 - cx, y2 - cy
                    if l < 0 or t < 0 or rgt < 0 or b < 0:
                        continue
                    cls[r, c] = 1.0
                    reg[:, r, c] = torch.tensor([l, t, rgt, b], dtype=reg.dtype)
                    mask[r, c] = True
        targets.append({"cls": cls, "reg": reg / s, "mask": mask})
    return targets


class DetectionLoss(nn.Module):
    """Focal (cls) + smooth-L1 (reg), both normalised by the number of positives.

    Batched input:
        cls_maps: list of 4 tensors [B, 1, H_i, W_i]
        reg_maps: list of 4 tensors [B, 4, H_i, W_i]
        targets:  list of 4 dicts {'cls': [B,H,W], 'reg': [B,4,H,W], 'mask': [B,H,W]}
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_maps, reg_maps, targets):
        cls_sum = 0.0
        reg_sum = 0.0
        n_pos = 0
        for cls_map, reg_map, tgt in zip(cls_maps, reg_maps, targets):
            cls = cls_map.sigmoid()[:, 0]     # [B,H,W]
            reg = reg_map                     # [B,4,H,W]
            t_cls = tgt["cls"]
            t_reg = tgt["reg"]
            m = tgt["mask"]                   # [B,H,W]
            pt = torch.where(t_cls > 0.5, cls, 1 - cls)
            focal = (1 - pt) ** self.gamma
            alpha_w = torch.where(
                t_cls > 0.5,
                torch.full_like(pt, self.alpha),
                torch.full_like(pt, 1 - self.alpha),
            )
            cls_sum += (focal * alpha_w * F.binary_cross_entropy(cls, t_cls, reduction="none")).sum()
            n_pos += int(m.sum().item())
            if m.any():
                reg_sel = reg.permute(0, 2, 3, 1)[m]        # [P, 4]
                tgt_sel = t_reg.permute(0, 2, 3, 1)[m]
                reg_sum += F.smooth_l1_loss(reg_sel, tgt_sel, beta=0.1, reduction="sum")
        n_pos = max(n_pos, 1)
        cls_loss = cls_sum / n_pos
        reg_loss = reg_sum / n_pos
        return cls_loss + reg_loss, cls_loss, reg_loss


# ---------------------------------------------------------------------------
# Inference decoding
# ---------------------------------------------------------------------------
def decode_predictions(cls_maps, reg_maps, strides=(4, 8, 16, 32), score_thr: float = 0.05):
    """
    Decode pyramid outputs into boxes.

    Returns (boxes [M,4] xyxy, scores [M]) for a single image.
    """
    boxes, scores = [], []
    for cls_map, reg_map, s in zip(cls_maps, reg_maps, strides):
        cls = cls_map[0, 0].sigmoid()
        reg = reg_map[0]
        h, w = cls.shape
        # coordinate grids without torch.meshgrid(indexing=) (torch<1.10 safe)
        ys = torch.arange(h, device=cls.device).view(-1, 1).expand(h, w)
        xs = torch.arange(w, device=cls.device).view(1, -1).expand(h, w)
        conf = cls[ys, xs]
        sel = conf > score_thr
        if sel.sum() == 0:
            continue
        xs_s, ys_s = xs[sel], ys[sel]
        conf_s = conf[sel]
        ltrb = reg[:, ys_s, xs_s] * s  # [4, M]
        cx = (xs_s.float() + 0.5) * s
        cy = (ys_s.float() + 0.5) * s
        x1 = cx - ltrb[0]
        y1 = cy - ltrb[1]
        x2 = cx + ltrb[2]
        y2 = cy + ltrb[3]
        boxes.append(torch.stack([x1, y1, x2, y2], dim=1))
        scores.append(conf_s)
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0)
    boxes = torch.cat(boxes)
    scores = torch.cat(scores)
    keep = torchvision_ops_nms(boxes, scores, iou_threshold=0.5)
    return boxes[keep], scores[keep]


def torchvision_ops_nms(boxes, scores, iou_threshold):
    from torchvision.ops import nms

    return nms(boxes, scores, iou_threshold)

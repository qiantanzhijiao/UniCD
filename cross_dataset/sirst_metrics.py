# -*- coding: utf-8 -*-
"""
SIRST detection metrics: Pd, Fa, nIoU (standard protocol in SIRST literature).

Definitions (pixel-level, as in DNANet/ISNet/ACM-Net papers):
    Pd   = N_detected_targets / N_gt_targets
           a target (connected component in GT mask) is detected if the
           predicted binary map has >=1 pixel inside it (intersection > 0).
    Fa   = N_false_alarm_pixels / N_pixels   (per-image average, x1e-3 reported)
    nIoU = mean over images of |Pred ∩ GT| / |Pred ∪ GT|
           (per-image IoU averaged, "normalized IoU")

Predicted boxes are rasterized to a binary mask before comparison.
"""
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _boxes_to_mask(boxes: np.ndarray, scores: np.ndarray, h: int, w: int,
                   score_thr: float = 0.5) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    for box, s in zip(boxes, scores):
        if s < score_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 >= x1 and y2 >= y1:
            m[y1:y2 + 1, x1:x2 + 1] = 1
    return m


def _components(mask: np.ndarray, min_px: int = 2):
    binm = (mask > 0).astype(np.uint8)
    if _HAS_CV2:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        comps = []
        for i in range(1, n):
            if stats[i][4] >= min_px:
                comps.append(labels == i)
        return comps
    return [binm > 0] if binm.any() else []


def pd_fa_niou(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Compute (Pd_hit, n_gt, fa_pixels, n_pixels, iou) for one image pair.
    Caller aggregates across the dataset.
    """
    h, w = gt_mask.shape
    n_pixels = h * w

    gt_comps = _components(gt_mask)
    n_gt = len(gt_comps)
    hits = 0
    for comp in gt_comps:
        if (pred_mask & comp).any():
            hits += 1

    fa_pixels = int(((pred_mask > 0) & (gt_mask == 0)).sum())

    inter = float(((pred_mask > 0) & (gt_mask > 0)).sum())
    union = float(((pred_mask > 0) | (gt_mask > 0)).sum())
    iou = inter / union if union > 0 else (1.0 if inter == 0 else 0.0)

    return hits, n_gt, fa_pixels, n_pixels, iou


class SIRSTMetrics:
    def __init__(self, score_thr: float = 0.5):
        self.score_thr = score_thr
        self.reset()

    def reset(self):
        self.hits = 0
        self.n_gt = 0
        self.fa_px = 0
        self.n_px = 0
        self.ious = []

    def update(self, gt_mask, boxes, scores):
        pred = _boxes_to_mask(np.asarray(boxes), np.asarray(scores),
                              gt_mask.shape[0], gt_mask.shape[1], self.score_thr)
        h, n, fa, px, iou = pd_fa_niou(np.asarray(gt_mask) > 0, pred > 0)
        self.hits += h
        self.n_gt += n
        self.fa_px += fa
        self.n_px += px
        self.ious.append(iou)

    def compute(self):
        return {
            "Pd": self.hits / max(self.n_gt, 1),
            "Fa": self.fa_px / max(self.n_px, 1),
            "nIoU": float(np.mean(self.ious)) if self.ious else 0.0,
            "n_gt": self.n_gt,
            "n_images": len(self.ious),
        }

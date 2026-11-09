# -*- coding: utf-8 -*-
"""
Evaluation metrics used in the UniCD paper.

NUC:      PSNR, SSIM
Detection: precision (P), recall (R)
Enhancement: SCRG (signal-to-clutter ratio gain)
"""
import numpy as np
from scipy.ndimage import gaussian_filter

EPS = 1e-12


# ---------------------------------------------------------------------------
# Image-quality metrics
# ---------------------------------------------------------------------------
def psnr(ref: np.ndarray, rec: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB. Inputs in [0,1] or same range."""
    ref = ref.astype(np.float64)
    rec = rec.astype(np.float64)
    mse = np.mean((ref - rec) ** 2)
    if mse < 1e-10:
        return float("inf")
    data_range = max(1.0, float(ref.max() - ref.min()))
    return float(10.0 * np.log10(data_range**2 / mse))


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    w = np.arange(size) - size // 2
    g = np.exp(-(w**2) / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def ssim(ref: np.ndarray, rec: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity (11x11 Gaussian window, default constants)."""
    ref = ref.astype(np.float64)
    rec = rec.astype(np.float64)
    if ref.ndim == 3:
        ref = ref.mean(axis=2)
        rec = rec.mean(axis=2)
    win = _gaussian_window()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = gaussian_filter(ref, 1.5, mode="reflect")
    mu2 = gaussian_filter(rec, 1.5, mode="reflect")
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = gaussian_filter(ref * ref, 1.5, mode="reflect") - mu1_sq
    sigma2_sq = gaussian_filter(rec * rec, 1.5, mode="reflect") - mu2_sq
    sigma12 = gaussian_filter(ref * rec, 1.5, mode="reflect") - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(num / (den + EPS)))


# ---------------------------------------------------------------------------
# Detection metrics (VOC-style P/R with IoU matching, single class)
# ---------------------------------------------------------------------------
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw, ih = x2 - x1, y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 0.0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 0.0)
    return inter / (area_a + area_b - inter + EPS)


def compute_pr(
    preds,          # list of (boxes [M,4], scores [M]) per image
    gts,            # list of gt boxes [N,4] per image
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
):
    """
    VOC-style P/R at a fixed confidence threshold.

    Returns (precision, recall, tp, fp, fn, n_pred, n_gt).
    """
    tp = fp = fn = 0
    n_pred_total = 0
    n_gt_total = 0
    for pred, gt in zip(preds, gts):
        boxes, scores = pred
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
        n_gt_total += gt.shape[0]
        if len(gt) == 0:
            fp += int((scores > score_thr).sum())
            n_pred_total += int((scores > score_thr).sum())
            continue
        mask = scores > score_thr
        boxes = boxes[mask]
        scores = scores[mask]
        n_pred_total += len(boxes)
        matched_gt = set()
        # greedy matching by descending score
        order = np.argsort(-scores)
        for i in order:
            best_iou, best_g = 0.0, -1
            for g in range(len(gt)):
                if g in matched_gt:
                    continue
                iou = _iou(boxes[i], gt[g])
                if iou > best_iou:
                    best_iou, best_g = iou, g
            if best_iou >= iou_thr and best_g >= 0:
                tp += 1
                matched_gt.add(best_g)
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec, tp, fp, fn, n_pred_total, n_gt_total


def compute_ap50(preds, gts, iou_thr: float = 0.5, n_bins: int = 101):
    """Average precision (interpolated P-R) at IoU 0.5, single class."""
    all_preds = []
    n_gt = 0
    for pred, gt in zip(preds, gts):
        boxes, scores = pred
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
        n_gt += gt.shape[0]
        all_preds.append((boxes, scores, gt))
    if n_gt == 0:
        return 0.0
    # global score-sorted list of (score, box, image index)
    ordered = []
    for img_idx, (boxes, scores, _gt) in enumerate(all_preds):
        for b, s in zip(boxes, scores):
            if s > 0:
                ordered.append((float(s), b, img_idx))
    if not ordered:
        return 0.0
    ordered.sort(key=lambda x: -x[0])
    matched = {i: set() for i in range(len(all_preds))}
    tp = fp = 0
    precs, recs = [], []
    for s, box, img_idx in ordered:
        gt = all_preds[img_idx][2]
        best_iou, best_g = 0.0, -1
        for g in range(len(gt)):
            if g in matched[img_idx]:
                continue
            iou = _iou(box, gt[g])
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_iou >= iou_thr:
            tp += 1
            matched[img_idx].add(best_g)
        else:
            fp += 1
        precs.append(tp / (tp + fp))
        recs.append(tp / n_gt)
    # interpolated AP
    ap = 0.0
    for t in np.linspace(0, 1, n_bins):
        p_vals = [p for p, r in zip(precs, recs) if r >= t]
        ap += max(p_vals) if p_vals else 0.0
    return ap / n_bins


# ---------------------------------------------------------------------------
# SCR / SCRG
# ---------------------------------------------------------------------------
def compute_scr(image: np.ndarray, box, dilation: int = 5) -> float:
    """Signal-to-clutter ratio for one target box.

    SCR = (mu_t - mu_b) / sigma_b, where the clutter region is a dilated ring
    around the target box. Approximate (dataset GT is box-based, not pixel-mask).
    """
    img = image.astype(np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    target = img[y1:y2, x1:x2]
    mu_t = target.mean()
    bg_mask = np.ones((h, w), dtype=bool)
    bg_mask[y1:y2, x1:x2] = False
    dx1, dx2 = max(x1 - dilation, 0), min(x2 + dilation, w - 1)
    dy1, dy2 = max(y1 - dilation, 0), min(y2 + dilation, h - 1)
    ring = np.zeros((h, w), dtype=bool)
    ring[dy1:dy2, dx1:dx2] = True
    ring[y1:y2, x1:x2] = False
    bg_pix = img[ring]
    if bg_pix.size < 5 or bg_pix.std() < 1e-6:
        return 0.0
    return float((mu_t - bg_pix.mean()) / bg_pix.std())


def scrg(orig: np.ndarray, corr: np.ndarray, boxes) -> float:
    """Mean SCRG over all GT boxes."""
    gains = []
    for box in np.asarray(boxes).reshape(-1, 4):
        s_o = compute_scr(orig, box)
        s_c = compute_scr(corr, box)
        if abs(s_o) > 1e-6:
            gains.append(s_c / s_o)
    return float(np.mean(gains)) if gains else 0.0

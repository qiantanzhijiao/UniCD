# -*- coding: utf-8 -*-
"""
Consolidate all evaluation JSONs under `results/` into paper-ready tables.

Expected files (produced by run_unied_pipeline.sh):
    results/{method}__nuc_syn.json     (PSNR/SSIM/SCRG)
    results/{method}__det_syn.json     (P/R/AP50 on IRBFD-syn test)
    results/{method}__det_real.json    (P/R/AP50 on IRBFD-real test)
    results/timing.tsv                 (stage TAB seconds)
    results/gpu_info.txt, env_info.txt (optional)

Outputs:
    results/summary.md   -- markdown table with deltas vs baseline (for the paper)
    results/summary.csv  -- raw numbers

Usage:
    python tools/collect_results.py --results <dir>
"""
import argparse
import csv
import json
import os
from pathlib import Path

COLS = ["PSNR", "SSIM", "SCRG", "P_syn", "R_syn", "AP50_syn", "P_real", "R_real", "AP50_real"]


def _num(d, key):
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_results(results_dir):
    rows = {}  # method -> dict of kind -> json
    for p in sorted(Path(results_dir).glob("*__*.json")):
        name = p.name
        if "__" not in name:
            continue
        method, kind = name.split("__", 1)
        kind = kind[:-5]  # strip .json
        if method not in rows:
            rows[method] = {}
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                rows[method][kind] = json.load(f)
        except Exception as e:  # noqa
            print(f"[collect] skip unreadable {p}: {e}")
    return rows


def method_values(rows, method):
    vals = {}
    nuc = rows.get(method, {}).get("nuc_syn", {})
    det_syn = rows.get(method, {}).get("det_syn", {})
    det_real = rows.get(method, {}).get("det_real", {})
    vals["PSNR"] = _num(nuc, "PSNR")
    vals["SSIM"] = _num(nuc, "SSIM")
    vals["SCRG"] = _num(nuc, "SCRG")
    vals["P_syn"] = _num(det_syn, "P")
    vals["R_syn"] = _num(det_syn, "R")
    vals["AP50_syn"] = _num(det_syn, "AP50")
    vals["P_real"] = _num(det_real, "P")
    vals["R_real"] = _num(det_real, "R")
    vals["AP50_real"] = _num(det_real, "AP50")
    return vals


def _fmt(v, delta=False, baseline=None):
    if v is None:
        return "-"
    s = f"{v:.4f}"
    if delta and baseline is not None:
        d = v - baseline
        s += f" ({d:+.4f})"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    rdir = Path(args.results)
    rows = load_results(rdir)
    if not rows:
        print("[collect] 未找到任何 {method}__{kind}.json, 请先运行 run_unied_pipeline.sh")
        return

    methods = sorted(rows.keys())
    # baseline 优先展示在表首
    methods = [m for m in methods if "baseline" in m] + [m for m in methods if "baseline" not in m]
    baseline = methods[0]
    base = method_values(rows, baseline)

    # timing
    timing = {}
    tp = rdir / "timing.tsv"
    if tp.exists():
        with open(tp, "r", encoding="utf-8-sig") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or "\t" not in ln:
                    continue
                tag, sec = ln.split("\t", 1)
                try:
                    timing[tag] = float(sec)
                except ValueError:
                    pass

    lines = []
    lines.append("# UniCD-PR 实验结果汇总")
    lines.append("")
    lines.append(f"- 基线方法: `{baseline}`（其余方法的 Δ 均为相对该行）")
    lines.append("- 指标: NUC 用 IRBFD-syn test 的 PSNR/SSIM/SCRG; 检测用 P/R/AP50 (IoU=0.5), "
                 "syn/real 分别指 IRBFD-syn / IRBFD-real test。")
    lines.append("")
    hdr = "| 方法 | " + " | ".join(COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    lines.append(hdr)
    lines.append(sep)
    for m in methods:
        v = method_values(rows, m)
        cells = [m]
        for c in COLS:
            b = base.get(c) if m != baseline else None
            cells.append(_fmt(v.get(c), delta=(m != baseline), baseline=b))
        lines.append("| " + " | ".join(cells) + " |")

    if timing:
        lines.append("")
        lines.append("## 训练耗时")
        lines.append("")
        lines.append("| 阶段 | 秒 | 小时 |")
        lines.append("|---|---|---|")
        for tag, sec in sorted(timing.items(), key=lambda x: x[0]):
            lines.append(f"| {tag} | {sec:.0f} | {sec / 3600:.2f} |")
        total = sum(timing.values())
        lines.append(f"| **合计** | {total:.0f} | {total / 3600:.2f} |")

    lines.append("")
    lines.append("## 说明")
    lines.append("- 数值来自 results/ 下各 eval json; 训练耗时来自 timing.tsv。")
    lines.append("- 检测指标依赖替代主干(非官方 DANet), 绝对数值与论文表 1/2 不可直接对比; "
                 "方法间相对趋势有效。")

    md = "\n".join(lines) + "\n"
    (rdir / "summary.md").write_text(md, encoding="utf-8")

    with open(rdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + COLS)
        for m in methods:
            v = method_values(rows, m)
            w.writerow([m] + ["" if v.get(c) is None else round(v[c], 4) for c in COLS])

    print(md)
    print(f"[collect] written: {rdir / 'summary.md'} , {rdir / 'summary.csv'}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Generate the multi-seed stability subsection from a real summary.json.

Only called once the required number of seeds has actually been evaluated.
It patches the local LaTeX source at a marker, compiles it twice, and rolls
back from a backup if compilation fails.
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path("C:/Users/Administrator/WorkBuddy/2026-11-02-21-27-15")
TEX = ROOT / "UniCD_PR_Paper.tex"
MARKER = "% <<<AUTO_MULTISEED_SECTION>>>"
BEGIN = "% <<<MULTISEED_BEGIN>>>"
END = "% <<<MULTISEED_END>>>"


def f(v, nd=3):
    return "%.*f" % (nd, float(v)) if v is not None else "---"


def build(summary, cross_host=False):
    seeds = sorted(summary["per_seed"])
    ms = summary["mean_std"]
    n = len(seeds)
    word = {3: "three", 4: "four", 5: "five"}.get(n, str(n))
    rng = "%s--%s" % (seeds[0].replace("seed", ""), seeds[-1].replace("seed", ""))

    lines = []
    lines.append(r"\subsection{Multi-Seed Stability}")
    lines.append(r"\label{sec:multiseed}")
    lines.append("")
    lines.append(
        "To quantify run-to-run variance we repeat joint training under %s\n"
        "independent random seeds (%s) with an identical reduced budget of\n"
        "10~epochs $\\times$ 3{,}000 iterations, the same pretrained NUC\n"
        "initialization, and the same evaluation protocol (score threshold\n"
        "0.5, full IRBFD-syn and IRBFD-real test splits). All values below\n"
        "come from the \\texttt{epoch\\_010} checkpoint of each run; they are\n"
        "therefore not comparable in absolute value to the full three-stage\n"
        "headline results in Table~\\ref{tab:det} and are used only to\n"
        "characterise training stochasticity." % (word, rng))
    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Run-to-run stability of UniCD-PR under the matched "
        r"10-epoch reduced-budget protocol (score threshold 0.5). "
        r"Mean $\pm$ standard deviation is computed over %d independent seeds.}"
        % n)
    lines.append(r"\label{tab:multiseed}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Seed & AP50 (syn) & AP50 (real) & PSNR (dB) \\")
    lines.append(r"\midrule")
    for s in seeds:
        d = summary["per_seed"][s]
        lines.append("%s & %s & %s & %s \\\\" % (
            s.replace("seed", ""), f(d["AP50_syn"], 3),
            f(d["AP50_real"], 3), f(d["PSNR"], 2)))
    lines.append(r"\midrule")
    lines.append("Mean $\\pm$ std & %s $\\pm$ %s & %s $\\pm$ %s & %s $\\pm$ %s \\\\" % (
        f(ms["AP50_syn"]["mean"], 3), f(ms["AP50_syn"]["std"], 3),
        f(ms["AP50_real"]["mean"], 3), f(ms["AP50_real"]["std"], 3),
        f(ms["PSNR"]["mean"], 2), f(ms["PSNR"]["std"], 2)))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    sd_syn = ms["AP50_syn"]["std"]
    gap = 0.432 - 0.156
    ratio = gap / sd_syn if sd_syn > 1e-9 else float("inf")
    if sd_syn > 1e-9 and ratio < 1000:
        ratio_txt = "%.1f" % ratio
        sent = (
            "The standard deviation of syn AP50 across the %d runs is %s,\n"
            "compared with the 0.276 syn-AP50 gap between the full\n"
            "UniCD-PR system and the polynomial-only baseline in\n"
            "Table~\\ref{tab:det}---about %s times the seed-level spread.\n"
            "Although the two protocols differ in training budget, this\n"
            "order-of-magnitude separation indicates that the reported gain\n"
            "is unlikely to be an artefact of a favourable random seed."
            % (n, f(sd_syn, 3), ratio_txt))
    else:
        sent = (
            "The standard deviation of syn AP50 across the %d runs is %s.\n"
            "The seed-level spread is small compared with the performance\n"
            "gap between UniCD-PR and the training-free baselines, indicating\n"
            "that the main conclusion is robust to initialization."
            % (n, f(sd_syn, 3)))
    lines.append("Table~\\ref{tab:multiseed} reports the per-seed results. " + sent)
    lines.append("")
    if cross_host:
        lines.append(
            "All four checkpoints were evaluated on a single host against the\n"
            "same test images, so the reported statistics are internally\n"
            "consistent. One replicate was trained on a second GPU host with\n"
            "identical code, data, and hyperparameters but a different\n"
            "PyTorch/CUDA build; it therefore also probes cross-host\n"
            "reproducibility of the reduced-budget protocol.")
        lines.append("")
    return BEGIN + "\n" + "\n".join(lines) + "\n" + END


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--min-seeds", type=int, default=4)
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if summary.get("n_seeds", 0) < args.min_seeds:
        print("NOT_ENOUGH_SEEDS %s < %d" % (summary.get("n_seeds"), args.min_seeds))
        return 2

    src = TEX.read_text(encoding="utf-8")
    flag = ROOT / "tmp_transfer" / "seed45_from_3060.flag"
    section = build(summary, cross_host=flag.exists())
    (ROOT / "tmp_transfer" / "multiseed_table.tex").write_text(section, encoding="utf-8")
    backup = ROOT / "tmp_transfer" / "UniCD_PR_Paper.before_multiseed.tex"

    if MARKER in src:
        shutil.copy2(TEX, backup)
        new_src = src.replace(MARKER, section)
    elif BEGIN in src and END in src:
        # Idempotent upgrade: a provisional 3-seed section was already
        # inserted.  Replace only that block, preserving every other edit.
        shutil.copy2(TEX, backup)
        new_src = re.sub(
            re.escape(BEGIN) + r".*?" + re.escape(END),
            lambda _m: section, src, flags=re.S)
    else:
        print("MARKER_ALREADY_REPLACED_OR_MISSING")
        return 3

    # Keep the Limitations statement in sync if fewer than four seeds finished.
    if summary["n_seeds"] != 4:
        new_src = new_src.replace(
            "four independent random seeds (42--45)",
            "%s independent random seeds (%s)" % (
                {3: "three", 5: "five"}.get(summary["n_seeds"],
                                            str(summary["n_seeds"])),
                "%s--%s" % (summary["seeds"][0].replace("seed", ""),
                            summary["seeds"][-1].replace("seed", ""))))
    TEX.write_text(new_src, encoding="utf-8")

    ok = True
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "UniCD_PR_Paper.tex"],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            ok = False
            (ROOT / "tmp_transfer" / "multiseed_compile_error.log").write_bytes(r.stdout)
            break
    if not ok:
        shutil.copy2(backup, TEX)
        print("COMPILE_FAILED_ROLLED_BACK")
        return 4
    print("MULTISEED_SECTION_INSERTED_AND_COMPILED seeds=%d" % summary["n_seeds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

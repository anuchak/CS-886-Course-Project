"""Figures and the results table for the report.

    python make_figures.py ../results ../report/figures

Writes figures/results.pdf (three panels) and figures/table_b50.tex.
Uses summarize.py's loader and threshold fits; a run counts as a success when
it holds >= 95% exact full-trace accuracy at 3 consecutive evaluations.
"""
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import load, n50, setting  # noqa: E402

# palette: categorical slots 1-2 (validated: CVD dE >= 9, normal-vision dE >= 24 on white)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def cells(rows):
    """setting -> N -> list of pass flags (final recipe only, answers-only excluded)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["config"]
        if c["arm"] != "none":
            out[setting(c)][c["N"]].append(bool(r["reached"]))
    return out


def pairs_per_step(arm, q, N):
    """Problems in which a given step and the step before it are both shown."""
    return q * N if arm == "complete" else q * q * N


def binned(points, lo=2.0 ** 3, hi=2.0 ** 17):
    """points: list of (x, passed, total) -> geometric-octave bins (x_center, rate, runs)."""
    edges = 2.0 ** np.arange(math.log2(lo) - 0.5, math.log2(hi) + 0.5, 1.0)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [(p, t) for x, p, t in points if a <= x < b]
        tot = sum(t for _, t in sel)
        if tot:
            out.append((math.sqrt(a * b), sum(p for p, _ in sel) / tot, tot))
    return out


def collapse_panel(ax, C, task, size, series, title):
    for label, arms_qs, color, marker, ls, fill in series:
        pts = []
        for (t, s, arm, q), byN in C.items():
            if t == task and s == size and (arm, q) in arms_qs:
                pts += [(pairs_per_step(arm, q, N), sum(v), len(v)) for N, v in byN.items()]
        b = binned(pts)
        ax.plot([x for x, _, _ in b], [y for _, y, _ in b], ls, color=color, lw=1.4, zorder=2)
        ax.plot([x for x, _, _ in b], [y for _, y, _ in b], marker, ms=4.2, color=color, label=label, zorder=3,
                mfc=color if fill else "white", mec=color, mew=1.0, linestyle="none")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("neighbouring shown pairs per step")
    ax.set_title(title, loc="left", color=INK)


def b50_with_ci(byN, arm, q, T, boots=300, seed=0):
    rng = np.random.default_rng(seed)
    point = n50(byN)
    if point is None:
        return None, None, None
    est = []
    for _ in range(boots):
        rs = {N: list(rng.choice(v, size=len(v), replace=True)) for N, v in byN.items()}
        f = n50(rs)
        if f is not None:
            est.append(f)
    lo, hi = (np.percentile(est, [5, 95]) if len(est) > boots // 2 else (None, None))
    k = q * T
    return point * k, (lo * k if lo is not None else None), (hi * k if hi is not None else None)


def ratio_ci(Cc, Cr, q, T, boots=300, seed=1):
    rng = np.random.default_rng(seed)
    pc, pr = n50(Cc), n50(Cr)
    if pc is None or pr is None:
        return None, None, None
    est = []
    for _ in range(boots):
        rc = {N: list(rng.choice(v, size=len(v), replace=True)) for N, v in Cc.items()}
        rr = {N: list(rng.choice(v, size=len(v), replace=True)) for N, v in Cr.items()}
        fc, fr = n50(rc), n50(rr)
        if fc and fr:
            est.append(fr * q / (fc * q))
    lo, hi = np.percentile(est, [5, 95]) if len(est) > boots // 2 else (None, None)
    return pr / pc, lo, hi


def main():
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    C = cells(load(root))

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.05), gridspec_kw=dict(width_ratios=[1, 1, 0.82]))
    allq = {0.75, 0.5, 0.25, 0.125}
    collapse_panel(axes[0], C, "parity", 30, [
        ("complete", {("complete", q) for q in allq}, BLUE, "o", "-", True),
        ("random", {("random", q) for q in allq}, ORANGE, "s", "-", True),
    ], "(a) Parity, n = 30, all q")
    collapse_panel(axes[1], C, "mult", "4x4", [
        ("complete", {("complete", q) for q in allq}, BLUE, "o", "-", True),
        ("random, q \u2265 1/2", {("random", 0.75), ("random", 0.5)}, ORANGE, "s", "-", True),
        ("random, q = 1/4", {("random", 0.25)}, ORANGE, "s", "--", False),
    ], "(b) Multiplication, 4\u00d74 digits")
    axes[0].set_ylabel("fraction of runs reaching 95%")
    axes[0].legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    axes[1].legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.1,
                   handlelength=1.2, labelspacing=0.25, fontsize=5.8)

    # (c) random / complete budget ratio vs n
    ax = axes[2]
    rows_c = []
    for q, marker, off in ((0.5, "o", -0.6), (0.25, "s", 0.6)):
        xs, ys, los, his = [], [], [], []
        for n in (10, 15, 20, 30):
            cc, cr = C.get(("parity", n, "complete", q)), C.get(("parity", n, "random", q))
            if not cc or not cr:
                continue
            r, lo, hi = ratio_ci(cc, cr, q, 8)
            if r is None:
                continue
            xs.append(n + off); ys.append(r); los.append(lo); his.append(hi)
            rows_c.append((q, n, r, lo, hi))
        ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, los)], [h - y for y, h in zip(ys, his)]],
                    fmt=marker, ms=4.2, color=INK, ecolor=INK2, elinewidth=0.8, capsize=1.5,
                    mfc=INK if marker == "o" else "white", mew=1.0, label=f"q = {'1/2' if q == 0.5 else '1/4'}")
        ax.axhline(1 / q, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.text(31.6, 1 / q, f"1/q = {int(1 / q)}", va="center", ha="left", color=INK2, fontsize=6.2)
    ax.axhline(1, color=GRID, lw=0.8, zorder=0)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(["1", "2", "4", "8", "16"])
    ax.set_xticks([10, 15, 20, 30]); ax.set_xlim(7, 31)
    ax.set_xlabel("input length n (parity, k = 8)")
    ax.set_ylabel("budget ratio, random / complete")
    ax.set_title("(c) Price of random traces vs n", loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(os.path.join(outdir, "results.pdf"))
    fig.savefig(os.path.join(outdir, "results.png"), dpi=220)

    # table: budget B50 at which half the runs reach 95%, with 90% bootstrap intervals
    fmt = lambda v: "--" if v is None else (f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}")
    qname = {0.75: "3/4", 0.5: "1/2", 0.25: "1/4", 0.125: "1/8"}
    lines = []
    for task, size, T, qs in (("parity", 30, 8, (0.75, 0.5, 0.25, 0.125)), ("mult", "4x4", 4, (0.75, 0.5, 0.25))):
        for q in qs:
            cc, cr = C.get((task, size, "complete", q)), C.get((task, size, "random", q))
            bc = b50_with_ci(cc, "complete", q, T) if cc else (None,) * 3
            br = b50_with_ci(cr, "random", q, T) if cr else (None,) * 3
            r = ratio_ci(cc, cr, q, T) if cc and cr else (None,) * 3
            per = C.get((task, size, "periodic", q))
            pstr = "--" if per is None else f"{sum(sum(v) for v in per.values())}/{sum(len(v) for v in per.values())}"
            rstr = "--" if r[0] is None else f"{r[0]:.1f} [{r[1]:.1f}, {r[2]:.1f}]"
            label = "Parity" if task == "parity" else "Mult."
            lines.append(f"{label} & {qname[q]} & {fmt(bc[0])} & {fmt(br[0])} & {rstr} & {1 / q:.1f} & {pstr} \\\\")
            print(lines[-1])
        lines.append("\\midrule")
    lines = lines[:-1]
    head = ["\\begin{tabular}{@{}llrrlrr@{}}", "\\toprule",
            "Task & $q$ & Compl. & Random & Ratio [90\\% CI] & $1/q$ & Periodic \\\\", "\\midrule"]
    with open(os.path.join(outdir, "table_b50.tex"), "w") as f:
        f.write("\n".join(head + lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    print("\n(c) ratios:", [(q, n, round(r, 2), lo and round(lo, 2), hi and round(hi, 2)) for q, n, r, lo, hi in rows_c])
    print("saved", outdir)


if __name__ == "__main__":
    main()

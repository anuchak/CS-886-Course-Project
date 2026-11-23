"""Numbers reported in the paper, computed from the result files.

    python summarize.py ../results [--json summary.json]

For every setting (task, n, arm, q) it reports
  * N*      : smallest N from which on every seed reached the target (the
              strict threshold), with the largest failing N below it;
  * N50     : data size at which half the runs reach the target, from a
              logistic fit of pass (0/1) against log2 N pooled over seeds
              (robust to isolated failures, which the strict rule is not);
  * B = q * N * T for both, T = number of trace steps (8 parity, 4 mult).
Only runs trained with the final recipe (warmup, clipping, hold=3) are used.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize


def load(root):
    rows = []
    for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        for line in open(path):
            r = json.loads(line)
            if "error" in r or r["config"].get("hold", 1) != 3:
                continue
            rows.append(r)
    return rows


def setting(c):
    if c["task"] == "parity":
        return ("parity", c["n"], c["arm"], c["q"])
    return ("mult", f"{c['da']}x{c['db']}", c["arm"], c["q"])


def steps(c):
    return c["k"] if c["task"] == "parity" else c["db"]


def strict(byN):
    best, below = None, None
    for N in sorted(byN, reverse=True):
        if not all(byN[N]):
            below = N
            break
        best = N
    return best, below


def n50(byN):
    """Logistic fit of P(pass) = sigmoid(a * (log2 N - m)); returns 2**m or None."""
    x = np.array([math.log2(N) for N, v in byN.items() for _ in v])
    y = np.array([float(ok) for N, v in byN.items() for ok in v])
    if y.sum() == 0 or y.sum() == len(y):
        return None
    def nll(p):
        a, m = p
        z = np.clip(a * (x - m), -30, 30)
        return np.sum(np.log1p(np.exp(-z)) * y + np.log1p(np.exp(z)) * (1 - y)) + 1e-3 * a * a
    best = min((minimize(nll, [a0, m0], method="Nelder-Mead") for a0 in (1.0, 4.0)
                for m0 in (x.min(), x.mean(), x.max())), key=lambda r: r.fun)
    a, m = best.x
    if a <= 0 or not (x.min() - 1 <= m <= x.max() + 1):
        return None
    return 2 ** m


def summarize(rows):
    cell = defaultdict(lambda: defaultdict(list))
    T = {}
    for r in rows:
        c = r["config"]
        if c["arm"] == "none":
            continue
        key = setting(c)
        cell[key][c["N"]].append(bool(r["reached"]))
        T[key] = steps(c)
    out = {}
    for key, byN in cell.items():
        task, size, arm, q = key
        Ns, below = strict(byN)
        fit = n50(byN)
        out[key] = dict(task=task, size=size, arm=arm, q=q, T=T[key],
                        runs=sum(len(v) for v in byN.values()), passed=sum(sum(v) for v in byN.values()),
                        N_star=Ns, N_below=below, N50=fit,
                        B_star=q * Ns * T[key] if Ns else None,
                        B50=q * fit * T[key] if fit else None,
                        curve={N: (sum(v), len(v)) for N, v in sorted(byN.items())})
    return out


def controls(rows):
    out = defaultdict(list)
    for r in rows:
        c = r["config"]
        if c["arm"] == "none":
            key = (c["task"], c.get("n") or f"{c['da']}x{c['db']}")
            out[key].append(max(h[3] for h in r["history"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = load(a.root)
    s = summarize(rows)
    fmt = lambda v: f"{v:9.0f}" if v else "        -"
    print(f"{'task':6s} {'size':>5s} {'q':>5s} {'arm':9s} {'pass':>8s} {'N*':>7s} {'(>fail)':>8s} {'B*':>9s} {'N50':>9s} {'B50':>9s}")
    for key in sorted(s, key=lambda k: (k[0], str(k[1]), -k[3], k[2])):
        v = s[key]
        print(f"{v['task']:6s} {str(v['size']):>5s} {v['q']:5.3f} {v['arm']:9s} {v['passed']:>3}/{v['runs']:<4} "
              f"{v['N_star'] or '-':>7} {('>' + str(v['N_below'])) if v['N_below'] else '':>8} {fmt(v['B_star'])} "
              f"{fmt(v['N50'])} {fmt(v['B50'])}")
    print("\nrandom / complete ratios (strict B*, and B50):")
    for key in sorted(s, key=lambda k: (k[0], str(k[1]), -k[3])):
        if key[2] != "random":
            continue
        c = s.get((key[0], key[1], "complete", key[3]))
        r = s[key]
        rs = r["B_star"] / c["B_star"] if c and c["B_star"] and r["B_star"] else None
        r5 = r["B50"] / c["B50"] if c and c["B50"] and r["B50"] else None
        print(f"  {key[0]:6s} {str(key[1]):>5s} q={key[3]:<6} strict {rs and round(rs, 2)}   N50 {r5 and round(r5, 2)}   1/q {1 / key[3]:.2f}")
    print("\nanswers-only controls (best answer accuracy per run):")
    for key, v in sorted(controls(rows).items(), key=lambda kv: str(kv[0])):
        print(f"  {key}: {[round(x, 3) for x in v]}")
    if a.json:
        json.dump({"|".join(map(str, k)): v for k, v in s.items()}, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

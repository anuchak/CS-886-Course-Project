"""Run a list of configs as batched ensembles on one GPU.

    python run_sweep.py --configs experiments/parity_n30.jsonl --out ../results/parity_n30.jsonl --ensemble 128

Configs are grouped by shape (cotb.ensemble.shape_key) and each group is
trained in chunks of up to --ensemble models in a single batched pass.
Finished configs (by id) already in --out are skipped, so a killed job can be
resubmitted and resumes. Results are appended as JSON lines under a file lock.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.environ.get("COT_COMMIT", "unknown")


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                if "error" not in r:
                    ids.add(r["id"])
            except Exception:
                pass
    return ids


def append(path, rows):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble", type=int, default=64, help="max models per batched pass")
    a = ap.parse_args()

    from cotb.ensemble import run_ensemble, shape_key
    from cotb.train import DEFAULTS, config_id
    cfgs = [json.loads(l) for l in open(a.configs) if l.strip()]
    finished = done_ids(a.out)
    groups = defaultdict(list)
    for c in cfgs:
        full = {**DEFAULTS, **c}
        if config_id({k: full[k] for k in sorted(full)}) not in finished:
            groups[shape_key(c)].append(c)
    todo = sum(len(g) for g in groups.values())
    print(f"{len(cfgs)} configs, {len(cfgs) - todo} done, {todo} to run in {len(groups)} shape groups", flush=True)

    commit, t0 = git_commit(), time.time()
    for key, group in groups.items():
        # sort so one chunk holds one arm (complete chunks then stop early together)
        group.sort(key=lambda c: (c.get("arm", ""), c.get("q", 0), c["N"], c["seed"]))
        batch = {**DEFAULTS, **group[0]}["batch"]
        size = max(1, min(a.ensemble, 65535 // batch))  # attention kernel limit: models * batch < 65536
        for i in range(0, len(group), size):
            chunk = group[i:i + size]
            try:
                rows = run_ensemble(chunk)
            except Exception as e:
                rows = [dict(id=None, config=c, error=repr(e)) for c in chunk]
            for r in rows:
                r["commit"] = commit
            append(a.out, rows)
            ok = sum(r.get("reached", False) for r in rows)
            print(f"[{time.time() - t0:7.0f}s] chunk of {len(chunk)} (N {chunk[0]['N']}..{chunk[-1]['N']}): "
                  f"{ok}/{len(chunk)} reached target, {rows[0].get('seconds')}s", flush=True)
    print(f"all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

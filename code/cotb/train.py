"""Config defaults and ids; `run` trains a single config.

Config keys: task, n, k, arm, q, N, seed, plus the model/training keys in
DEFAULTS. All training goes through cotb.ensemble (a single run is an ensemble
of one), so there is exactly one training and evaluation code path.
"""
import hashlib
import json

# version 2: periodic arm with balanced offsets; full-trace exact-match metric
DEFAULTS = dict(task="parity", n=30, k=8, d=64, layers=2, heads=2, lr=1e-3, wd=0.1,
                batch=256, max_steps=40000, eval_every=500, n_test=1000, target_acc=0.95,
                version=2)


def config_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run(cfg: dict, device: str = None) -> dict:
    from .ensemble import run_ensemble
    return run_ensemble([cfg], device=device)[0]

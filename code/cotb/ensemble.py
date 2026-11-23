"""Train M independent models in one batched pass.

The models are tiny, so a single run is limited by GPU kernel-launch overhead,
not compute. Stacking M models' weights along a leading axis makes one step cost
about the same as one model's step. Each model keeps its own data, its own loss
term (the total loss is a sum, so gradients do not mix) and its own AdamW state
(AdamW is elementwise), so results match M separate runs.

All configs in one ensemble must share every key except arm, q, N and seed
(see shape_key).

Metric: exact full-trace accuracy (every step and the answer right, the model
writing all steps itself). A run has reached the target once exact >= target_acc
at `hold` consecutive evaluations; steps_to_target is the first of them.
History rows are (step, train loss, exact acc, answer-only acc).

Optional training keys (absent = off, so older configs train exactly as before):
    warmup : linear learning-rate warmup over this many steps
    clip   : clip each model's own gradient norm to this value
    hold   : consecutive evaluations at >= target_acc required (default 1)
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import BLANK, make_dataset, make_task
from .train import DEFAULTS, config_id

PER_MODEL_KEYS = ("arm", "q", "N", "seed")   # the only keys that may differ within an ensemble


def shape_key(cfg):
    cfg = {**DEFAULTS, **cfg}
    return tuple(sorted((k, v) for k, v in cfg.items() if k not in PER_MODEL_KEYS))


class EnsembleGPT(nn.Module):
    """M pre-LN decoder-only transformers with stacked weights."""

    def __init__(self, M, vocab, max_len, d=128, layers=4, heads=4, mlp=4):
        super().__init__()
        self.M, self.d, self.heads, self.nl = M, d, heads, layers

        def P(*shape, std=None, val=None):
            t = torch.empty(M, *shape)
            if val is not None:
                t.fill_(val)
            else:
                t.normal_(0, std)
            return nn.Parameter(t)

        s, s_out = 0.02, 0.02 / math.sqrt(2 * layers)
        self.tok, self.pos = P(vocab, d, std=s), P(max_len, d, std=s)
        self.blk = nn.ParameterList()
        for _ in range(layers):
            self.blk.extend([
                P(d, val=1.0), P(d, val=0.0),                    # ln1
                P(d, 3 * d, std=s), P(3 * d, val=0.0),           # qkv
                P(d, d, std=s_out), P(d, val=0.0),               # proj
                P(d, val=1.0), P(d, val=0.0),                    # ln2
                P(d, mlp * d, std=s), P(mlp * d, val=0.0),       # fc1
                P(mlp * d, d, std=s_out), P(d, val=0.0),         # fc2
            ])
        self.lnf_w, self.lnf_b = P(d, val=1.0), P(d, val=0.0)
        self.head = P(d, vocab, std=s)

    @staticmethod
    def _ln(x, w, b):
        x = F.layer_norm(x, x.shape[-1:])
        return x * w[:, None, None] + b[:, None, None]

    @staticmethod
    def _lin(x, w, b):  # x (M,B,L,i), w (M,i,o), b (M,o)
        return torch.einsum("mbli,mio->mblo", x, w) + b[:, None, None]

    def forward(self, idx):  # idx (M, B, L) -> logits (M, B, L, V)
        M, B, L = idx.shape
        x = self.tok[torch.arange(M, device=idx.device)[:, None, None], idx] + self.pos[:, None, :L]
        h, dh = self.heads, self.d // self.heads
        for i in range(self.nl):
            l1w, l1b, qw, qb, pw, pb, l2w, l2b, f1w, f1b, f2w, f2b = self.blk[12 * i:12 * i + 12]
            qkv = self._lin(self._ln(x, l1w, l1b), qw, qb)                     # (M,B,L,3d)
            q, k, v = qkv.view(M * B, L, 3, h, dh).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # (MB,h,L,dh)
            x = x + self._lin(a.transpose(1, 2).reshape(M, B, L, self.d), pw, pb)
            x = x + self._lin(F.gelu(self._lin(self._ln(x, l2w, l2b), f1w, f1b)), f2w, f2b)
        x = self._ln(x, self.lnf_w, self.lnf_b)
        return torch.einsum("mbld,mdv->mblv", x, self.head)


@torch.no_grad()
def evaluate_all(model, tests, out_vocab, device, bs=250):
    """Full-trace test, the same for every arm: each model writes every token of
    all T trace steps itself (greedy, restricted to out_vocab), then the answer.
    Returns per-model (exact, answer) accuracy; exact counts a problem only if
    every trace token and every answer token is right.
    tests: list of M full-trace Batches."""
    model.eval()
    vocab = torch.tensor(out_vocab, device=device)
    seq_all = torch.stack([t.seq for t in tests]).to(device)       # (M, n, L)
    t0 = tests[0]
    ts, Lt, wa = t0.trace_start, t0.T * t0.w, t0.wa
    a0 = ts + Lt + 1                                               # first answer token (after ANS)
    write = list(range(ts, ts + Lt)) + list(range(a0, a0 + wa))   # positions the model writes
    exact = torch.zeros(len(tests), device=device)
    ans_ok = torch.zeros(len(tests), device=device)
    for i in range(0, seq_all.shape[1], bs):
        seq = seq_all[:, i:i + bs].clone()
        true = seq.clone()
        seq[:, :, ts:ts + Lt] = BLANK
        seq[:, :, a0:a0 + wa] = BLANK
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            for p in write:
                seq[:, :, p] = vocab[model(seq[:, :, :p])[:, :, -1, vocab].argmax(-1)]
        a_ok = (seq[:, :, a0:a0 + wa] == true[:, :, a0:a0 + wa]).all(-1)
        steps_ok = (seq[:, :, ts:ts + Lt] == true[:, :, ts:ts + Lt]).all(-1)
        exact += (a_ok & steps_ok).float().sum(1)
        ans_ok += a_ok.float().sum(1)
    model.train()
    n = seq_all.shape[1]
    return (exact / n).tolist(), (ans_ok / n).tolist()


def clip_per_model(params, M, max_norm):
    """Clip each model's gradient to norm max_norm separately. Parameters are
    stacked along dim 0 (one slice per model), so a global clip would let one
    model's large gradient shrink every other model's update."""
    grads = [p.grad for p in params if p.grad is not None]
    sq = torch.zeros(M, device=grads[0].device)
    for g in grads:
        sq += g.float().pow(2).flatten(1).sum(1)
    norm = sq.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.view(-1, *([1] * (g.dim() - 1))).to(g.dtype))
    return norm


def run_ensemble(cfgs, device=None):
    """Train len(cfgs) models together; returns one result dict per config."""
    cfgs = [{**DEFAULTS, **c} for c in cfgs]
    assert len({shape_key(c) for c in cfgs}) == 1, "configs must share shape keys"
    c0, M = cfgs[0], len(cfgs)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    torch.manual_seed(sum(c["seed"] * 1_000_003 + c["N"] for c in cfgs) % (2 ** 31))

    trains, tests = [], []
    for c in cfgs:
        task = make_task(c)
        trains.append(make_dataset(task, c["arm"], c["N"], c["q"], seed=10_000 + c["seed"]))
        tests.append(make_dataset(task, c["arm"], c["n_test"], c["q"], seed=20_000 + c["seed"], test=True))
    Ns = torch.tensor([c["N"] for c in cfgs], device=device)
    Nmax, L = int(Ns.max()), trains[0].seq.shape[1]
    seq = torch.zeros(M, Nmax, L, dtype=torch.long, device=device)
    tgt = torch.full((M, Nmax, L), -100, dtype=torch.long, device=device)
    for m, tr in enumerate(trains):
        seq[m, :tr.seq.shape[0]] = tr.seq.to(device)
        tgt[m, :tr.seq.shape[0]] = tr.target.to(device)

    out_vocab, vocab = task.out_vocab, task.vocab
    model = EnsembleGPT(M, vocab, L, c0["d"], c0["layers"], c0["heads"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c0["lr"], weight_decay=c0["wd"], betas=(0.9, 0.98))
    ar = torch.arange(M, device=device)[:, None]
    hist = [[] for _ in range(M)]
    reached_at, first_touch = [None] * M, [None] * M
    streak, streak_start = [0] * M, [None] * M
    warmup, clip, hold = c0.get("warmup", 0), c0.get("clip", 0), c0.get("hold", 1)
    params = list(model.parameters())
    best = [0.0] * M
    for step in range(1, c0["max_steps"] + 1):
        idx = (torch.rand(M, c0["batch"], device=device) * Ns[:, None]).long()
        x, y = seq[ar, idx], tgt[ar, idx]                                     # (M,B,L)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        ce = F.cross_entropy(logits.float().permute(0, 3, 1, 2), y, ignore_index=-100, reduction="none")
        per_model = ce.sum((1, 2)) / (y != -100).sum((1, 2)).clamp(min=1)    # mean CE per model
        opt.zero_grad(set_to_none=True)
        per_model.sum().backward()
        if clip:
            clip_per_model(params, M, clip)
        if warmup:
            for gr in opt.param_groups:
                gr["lr"] = c0["lr"] * min(1.0, step / warmup)
        opt.step()
        if step % c0["eval_every"] == 0 or step == c0["max_steps"]:
            loss_m = per_model.detach().float().cpu()
            exact, ans = evaluate_all(model, tests, out_vocab, device)
            for m in range(M):
                best[m] = max(best[m], exact[m])
                hist[m].append((step, round(loss_m[m].item(), 4), round(exact[m], 4), round(ans[m], 4)))
                if exact[m] >= c0["target_acc"]:
                    if streak[m] == 0:
                        streak_start[m] = step
                    streak[m] += 1
                    if first_touch[m] is None:
                        first_touch[m] = step
                    if reached_at[m] is None and streak[m] >= hold:
                        reached_at[m] = streak_start[m]
                else:
                    streak[m] = 0
            n_done = sum(r is not None for r in reached_at)
            print(f"    step {step}/{c0['max_steps']}: mean exact acc {sum(exact) / M:.3f}, "
                  f"max {max(exact):.3f}, reached {n_done}/{M}, {time.time() - t0:.0f}s", flush=True)
            if n_done == M:
                break

    secs = round(time.time() - t0, 1)
    dev = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    out = []
    for m, c in enumerate(cfgs):
        out.append(dict(id=config_id({k: c[k] for k in sorted(c)}), config=c,
                        reached=reached_at[m] is not None, steps_to_target=reached_at[m],
                        first_touch=first_touch[m],
                        best_acc=best[m], final_acc=hist[m][-1][2], final_answer_acc=hist[m][-1][3],
                        final_loss=hist[m][-1][1],
                        history=hist[m], seconds=secs, ensemble_size=M, device=dev,
                        torch=torch.__version__))
    return out

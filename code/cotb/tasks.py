"""Tasks and budget arms.

A problem has an input x, a trace of T steps (each step is w tokens), and a
final answer (wa tokens). A sequence is
    x  SEP  step_1 .. step_T  ANS  answer
where every token of a hidden step is replaced by BLANK. The model is trained
with next-token loss on every token of every shown step and on the answer.

Tasks:
    parity : XOR of a hidden subset S of k of n bits. Step j = running parity
             after the j-th selected bit (w = 1); T = k; answer = last step.
    mult   : a x b with a of da digits and b of db digits. Step j (one per digit
             of b, least significant first) = partial product a*b_j*10^(j-1)
             followed by the running sum; T = db. Every number is zero-padded
             to W = da + db digits and written least significant digit first
             (the standard format of Deng et al. 2023/2024 and Bai et al. 2025),
             so w = 2W and the answer (= last running sum) has wa = W tokens.

Budget arms (q = fraction of trace steps shown, B = q * N * T steps):
    complete : full traces on round(q*N) problems, all steps hidden on the rest
    random   : exactly B steps shown, chosen uniformly over all N*T steps
    periodic : stride s = 1/q with a per-problem offset o in {0..s-1}: problem
               shows steps o+1, o+1+s, ...; offsets are exactly balanced, so every
               step is shown in a q fraction of problems (as in random), but two
               neighbouring steps are never shown together
At test time every arm is scored the same way: the model writes all T steps
itself, then the answer (see cotb.ensemble.evaluate_all).
"""
from dataclasses import dataclass

import torch

# ---- shared vocabulary -------------------------------------------------
# digits 0..9 are tokens 0..9 (parity uses only 0 and 1)
SEP, ANS, BLANK, PAD, MUL = 10, 11, 12, 13, 14


@dataclass
class Batch:
    seq: torch.Tensor      # (N, L) int64 tokens, hidden steps already BLANK
    target: torch.Tensor   # (N, L) int64 next-token targets (-100 = no loss)
    answer: torch.Tensor   # (N, wa) correct answer tokens
    shown: torch.Tensor    # (N, T) bool, which trace steps are shown
    trace_start: int       # index of the first token of step 1 in seq
    T: int                 # number of trace steps
    w: int                 # tokens per step
    wa: int                # tokens in the answer


# ---- parity ------------------------------------------------------------
class Parity:
    """XOR of a hidden subset S of k positions in an n-bit string.
    Trace = running parity after each selected bit (T = k steps); the answer
    equals the last trace step and is always shown."""
    vocab = 14             # tokens 0..13 (no MUL), kept so parity runs are unchanged
    w, wa = 1, 1

    def __init__(self, n: int, k: int, task_seed: int):
        g = torch.Generator().manual_seed(task_seed)
        self.n, self.k, self.T = n, k, k
        self.S = torch.randperm(n, generator=g)[:k].sort().values
        self.out_vocab = [0, 1]

    def sample(self, N: int, g: torch.Generator):
        x = torch.randint(0, 2, (N, self.n), generator=g)
        trace = x[:, self.S].cumsum(1) % 2           # (N, k)
        return x, trace[:, :, None], trace[:, -1:].clone()


# ---- multiplication ----------------------------------------------------
def digits(v: torch.Tensor, W: int) -> torch.Tensor:
    """(N,) non-negative ints -> (N, W) digit tokens, least significant first."""
    return (v[:, None] // (10 ** torch.arange(W))) % 10


class Multiplication:
    """a x b, a uniform over da-digit numbers, b over db-digit numbers.
    Step j: [partial product a * b_j * 10^(j-1), running sum], each W digits."""
    vocab = 15
    out_vocab = list(range(10))

    def __init__(self, da: int, db: int):
        self.da, self.db, self.T = da, db, db
        self.W = da + db
        self.w, self.wa = 2 * self.W, self.W

    def sample(self, N: int, g: torch.Generator):
        a = torch.randint(10 ** (self.da - 1), 10 ** self.da, (N,), generator=g)
        b = torch.randint(10 ** (self.db - 1), 10 ** self.db, (N,), generator=g)
        bd = digits(b, self.db)                                   # (N, db), b_1 = last digit
        partial = a[:, None] * bd * (10 ** torch.arange(self.db))  # (N, T)
        running = partial.cumsum(1)
        trace = torch.stack([torch.stack([digits(partial[:, j], self.W), digits(running[:, j], self.W)], 1)
                             .reshape(N, self.w) for j in range(self.T)], 1)   # (N, T, w)
        x = torch.cat([digits(a, self.da), torch.full((N, 1), MUL), digits(b, self.db)], 1)
        return x, trace, digits(running[:, -1], self.W)


TASKS = {"parity": Parity, "mult": Multiplication}


def make_task(cfg: dict):
    if cfg["task"] == "parity":
        return Parity(cfg["n"], cfg["k"], task_seed=cfg["seed"])
    if cfg["task"] == "mult":
        return Multiplication(cfg["da"], cfg["db"])
    raise ValueError(cfg["task"])


# ---- budget arms -------------------------------------------------------
def make_shown(arm: str, N: int, T: int, q: float, g: torch.Generator) -> torch.Tensor:
    """Which trace steps are shown, as an (N, T) bool mask with budget q*N*T."""
    shown = torch.zeros(N, T, dtype=torch.bool)
    if arm == "complete":
        m = round(q * N)
        shown[:m] = True
    elif arm == "random":
        B = round(q * N * T)
        idx = torch.randperm(N * T, generator=g)[:B]
        shown.view(-1)[idx] = True
    elif arm == "periodic":
        s = round(1 / q)
        assert abs(s * q - 1) < 1e-9 and T % s == 0, f"stride 1/q={1/q} must divide T={T}"
        offset = torch.randperm(N, generator=g) % s            # balanced, random order
        shown = (torch.arange(T)[None, :] % s) == offset[:, None]
    elif arm == "none":
        pass
    else:
        raise ValueError(arm)
    return shown


def build(x, trace, answer, shown) -> Batch:
    """Assemble token sequences and next-token targets.
    x (N, n_in), trace (N, T, w), answer (N, wa), shown (N, T)."""
    N, n_in = x.shape
    _, T, w = trace.shape
    wa = answer.shape[1]
    sep = torch.full((N, 1), SEP)
    ans = torch.full((N, 1), ANS)
    steps = torch.where(shown[:, :, None], trace, torch.full_like(trace, BLANK)).reshape(N, T * w)
    seq = torch.cat([x, sep, steps, ans, answer], dim=1)       # length n_in + T*w + wa + 2
    L = seq.shape[1]
    trace_start = n_in + 1
    # target[t] is the token the model must predict from prefix seq[:t+1]
    target = torch.full((N, L), -100, dtype=torch.long)
    tgt_steps = torch.where(shown[:, :, None], trace, torch.full_like(trace, -100)).reshape(N, T * w)
    target[:, trace_start - 1:trace_start - 1 + T * w] = tgt_steps
    target[:, L - 1 - wa:L - 1] = answer                        # answer tokens, the first one from ANS
    return Batch(seq, target, answer, shown, trace_start, T, w, wa)


def make_dataset(task, arm: str, N: int, q: float, seed: int, test: bool = False) -> Batch:
    """Training set for an arm, or (test=True) a test set with the full trace,
    which the model must write out itself at evaluation."""
    g = torch.Generator().manual_seed(seed)
    x, trace, answer = task.sample(N, g)
    shown = torch.ones(N, task.T, dtype=torch.bool) if test else make_shown(arm, N, task.T, q, g)
    return build(x, trace, answer, shown)

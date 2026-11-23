# Complete vs. Partial Chain-of-Thought Traces under a Fixed Budget

Code and results for the CS 886 course project (Fall 2026).
The report is [`report/main.pdf`](report/main.pdf).

## Repository structure

```
code/
  cotb/              tasks, budget allocations and training
  grids/             the training runs of each experiment
  run_sweep.py       trains the runs listed in a grid file
  summarize.py       prints the numbers in the report
  make_figures.py    writes Figure 1 and Table 2
  slurm/             optional job scripts for a Slurm cluster
results/             one line per trained model, one file per experiment
report/              LaTeX source and PDF of the report
requirements.txt     Python dependencies
```

## Setup

Python 3.12:

```
pip install -r requirements.txt
```

## Reproduce results

### Reproduce figures from saved results

No GPU needed. Run from `code/`:

```
python summarize.py ../results
python make_figures.py ../results ../report/figures
cd ../report && latexmk -pdf main.tex
```

### Train models

Needs a GPU. Each file in `code/grids/` is one experiment: it lists every model to
train, one per line, with its task, budget allocation, fraction of steps shown,
number of training problems and seed. The models are trained to the matching file
in `results/`:

| Experiment | Models | Used in the report for |
|---|---|---|
| Parity, 30 bits, 8 bits hidden (`parity_n30`) | 522 | parity results (Figure 1a, Table 2) |
| Parity, 10, 15 and 20 bits, 8 bits hidden (`parity_n10`, `parity_n15`, `parity_n20`) | 105 each | price of random traces against input length (Figure 1c) |
| Multiplication, 4 digits by 4 digits (`multiplication`) | 378 | multiplication results (Figure 1b, Table 2) |

To train one experiment, for example parity with 30 bits, run from `code/`:

```
python run_sweep.py --configs grids/parity_n30.jsonl --out ../results/parity_n30.jsonl --ensemble 128
```

`--ensemble` is the number of models trained at once on one GPU; we used 128 for
parity and 39 for multiplication. Results are saved after every batch, and models
already in the output file are skipped, so an interrupted run continues when the
same command is run again.

On a Slurm cluster, `code/slurm/sweep.sh` runs the same command as a batch job. It
is set up for the Fir cluster, so change the account and paths in `code/slurm/`
before using it elsewhere.

**Compute used.** All models were trained on Fir cluster of Compute Canada. Each job used one NVIDIA H100 80 GB
GPU, 8 CPU cores and 64 GB RAM.

| | Parity | Multiplication |
|---|---|---|
| Models trained | 837 | 378 |
| Models per batch on one H100 | up to 128 | up to 39 |
| Time per batch | up to 51 minutes | up to 1 hour 45 minutes |
| Total GPU time | about 6.4 hours | about 17.6 hours |

In total: 1,215 models and about 24 GPU-hours. Every result line records the GPU
it ran on.

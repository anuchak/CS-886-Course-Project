#!/bin/bash
# Build the Python environment on node-local disk and save it to /scratch as a
# single tar (run inside a job, e.g.  srun --jobid=<id> --overlap bash -l slurm/build_env.sh).
# Building directly on /scratch or /home is very slow: thousands of small files.
set -e
COT_ROOT=${COT_ROOT:-/scratch/$USER/cot}
T0=$(date +%s); t(){ echo "[$(( $(date +%s) - T0 ))s] $*"; }
L=$SLURM_TMPDIR
export VIRTUALENV_OVERRIDE_APP_DATA=$L/.appdata PIP_CACHE_DIR=$L/.pipcache
module load python/3.12
virtualenv --no-download "$L/venv" > /dev/null; t venv
"$L/venv/bin/python" -m pip install --no-index --upgrade pip -q; t pip
"$L/venv/bin/python" -m pip install --no-index torch numpy matplotlib -q; t packages
"$L/venv/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mkdir -p "$COT_ROOT/env"
tar -C "$L" -cf "$COT_ROOT/env/venv.tar" venv; t tar

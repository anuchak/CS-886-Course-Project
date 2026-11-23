# Source at the start of every job step:  source slurm/env.sh
# Unpacks the prebuilt venv (one tar on /scratch) onto node-local disk once per
# job, so no job touches the slow shared filesystems for Python imports.
# Build the tar once with slurm/build_env.sh.
COT_ROOT=${COT_ROOT:-/scratch/$USER/cot}
module load python/3.12
if [ ! -x "$SLURM_TMPDIR/venv/bin/python" ]; then
  tar -C "$SLURM_TMPDIR" -xf "$COT_ROOT/env/venv.tar"
fi
export PATH="$SLURM_TMPDIR/venv/bin:$PATH"
export PYTHONUNBUFFERED=1

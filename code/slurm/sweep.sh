#!/bin/bash
# Run one grid on a full H100:  sbatch --job-name=NAME slurm/sweep.sh GRID.jsonl OUT.jsonl [ENSEMBLE]
# ENSEMBLE = models trained at once (default 128; 39 was used for multiplication).
# If the GPU on the assigned node is unusable, the job resubmits itself with that
# node excluded (at most 10 nodes), so a bad GPU never runs the sweep on CPU.
# Optional environment: COT_EXCLUDE=node1:node2 (':'-separated), COT_GPUS (e.g.
# nvidia_h100_80gb_hbm3_3g.40gb:1) and COT_MEM, which resubmissions keep.
#SBATCH --account=def-ssalihog
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/anuchak/cot/logs/%x-%j.out
set -euo pipefail
SELF=/scratch/anuchak/cot/code/slurm/sweep.sh
cd /scratch/anuchak/cot/code
source slurm/env.sh
echo "start $(date -Is) on $(hostname): $1 -> $2"
nvidia-smi --query-gpu=index,name,driver_version,mig.mode.current,compute_mode --format=csv,noheader || true
# COT_EXCLUDE holds the nodes to avoid separated by ':' (sbatch --export splits on ',')
if ! python -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda available:', ok); sys.exit(0 if ok else 3)"; then
  EX="${COT_EXCLUDE:+$COT_EXCLUDE:}$(hostname -s)"
  if [ "$(echo "$EX" | tr ':' '\n' | wc -l)" -le 10 ]; then
    echo "no usable GPU on $(hostname); resubmitting excluding $EX"
    sbatch --exclude="$(echo "$EX" | tr ':' ',')" --export=ALL,COT_EXCLUDE="$EX" --job-name="$SLURM_JOB_NAME" \
      --gpus="${COT_GPUS:-h100:1}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --mem="${COT_MEM:-64000M}" "$SELF" "$1" "$2" "${3:-128}"
  else
    echo "no usable GPU on $(hostname); giving up after excluding $EX"
  fi
  exit 3
fi
python run_sweep.py --configs "$1" --out "$2" --ensemble "${3:-128}"

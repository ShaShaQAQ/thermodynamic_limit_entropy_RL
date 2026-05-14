#!/usr/bin/env bash
#PBS -N tensor_mpo_mps
#PBS -q cmt
#PBS -j oe
#PBS -l nodes=1:ppn=24
#PBS -l walltime=02:00:00

set -euo pipefail

cd "$HOME/thermodynamic_limit_entropy_RL_tensor_upload/tensor_version"

PYTHON=${PYTHON:-$HOME/miniconda3/envs/thermodynamic_limit_entropy_rl/bin/python}
L=${L:-8}
N=${N:-$((L / 2))}
CHI_MAX=${CHI_MAX:-64}
STEPS=${STEPS:-50}
CUTOFF=${CUTOFF:-1e-12}
OUT=${OUT:-outputs/tensor_mpo_mps_L${L}_chi${CHI_MAX}.json}

"$PYTHON" run_tensor_mpo_mps.py \
  --L "$L" \
  --N "$N" \
  --chi-max "$CHI_MAX" \
  --steps "$STEPS" \
  --cutoff "$CUTOFF" \
  --out "$OUT"

# thermodynamic_limit_entropy_RL

A research-oriented continuation of the codebase from:

- A. Arriojas, J. Adamczyk, S. Tiomkin, and R. Kulkarni
- "Entropy regularized reinforcement learning using large deviation theory"
- Original repository: https://github.com/argearriojas/2023-EntRegRL

This repository keeps the original computational core while reorganizing the project for
continued development, reproducibility, and cluster execution.

## Current contents

- `main.py`: entry point used to reproduce the paper figures
- `utils.py`: matrix construction, Perron-eigenvector solvers, and evaluation helpers
- `frozen_lake_env.py`: custom Gym environment used to define the discrete MDPs
- `visualization.py`: plotting utilities for distributions and policies
- `jobs/run_entregl.pbs`: PBS batch script for running the project on W003
- `environment.yml`: conda environment definition
- `requirements.txt`: pip-style dependency list

## Environment

Conda:

```bash
conda env create -f environment.yml
conda activate thermodynamic_limit_entropy_rl
```

Pip:

```bash
python -m pip install -r requirements.txt
```

## Running locally or on a server

Interactive run:

```bash
MPLBACKEND=Agg python main.py
```

PBS run on W003:

```bash
cd ~/2023-EntRegRL
qsub jobs/run_entregl.pbs
```

## Notes

- The original upstream code used `gym`, which is now unmaintained. The current setup pins
  `gym==0.26.2` and constrains `numpy<2` for better compatibility.
- The repository is still intentionally close to the original implementation so that the paper
  derivations remain easy to trace into code.

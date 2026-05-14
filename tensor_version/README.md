# Tensor-network thermodynamic-limit RL branch

This folder contains the tensor-network route developed as an independent branch from the original EntRegRL reproduction code.

Core files:

- `controlled_chain_experiment.py`: exact, MPO/MPS model-based, and sampled u-theta checks for a controlled hard-core chain.
- `make_controlled_chain_notebook.py`: regenerates the explanatory notebook.
- `controlled_chain_workflow.ipynb`: notebook version of the workflow.
- `controlled_chain_workflow.html`: rendered notebook for quick reading.
- `articles/tensor_network_rl_supplement.tex`: LaTeX supplement explaining the tensor-network formulation.
- `outputs/*.json`: recorded small-system validation results.

The implementation encodes a state-action pair `(n, a)` as local symbols `y_j=(n_j,m_j)`, represents the positive Perron eigenfunction as an MPS `u_theta(n,a)=exp(f_theta(n,a))`, and validates a finite-automaton MPO representation of the tilted state-action kernel.

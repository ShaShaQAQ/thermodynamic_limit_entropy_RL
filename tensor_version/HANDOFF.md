# HANDOFF: Tensor-network thermodynamic-limit RL project

This handoff is for continuing the `tensor` branch development of `ShaShaQAQ/thermodynamic_limit_entropy_RL`.

## Current Local/Remote Locations

Local working directory on this machine:

```bash
/Users/shajianyu/CMP_manybody/Quantum_AI/final_project
```

W003 synchronized clean tensor-branch clone:

```bash
~/thermodynamic_limit_entropy_RL_tensor_upload
```

GitHub repository and branch:

```text
https://github.com/ShaShaQAQ/thermodynamic_limit_entropy_RL/tree/tensor
```

Current tensor files are stored under the repo subdirectory:

```text
tensor_version/
```

The original `main` branch of `thermodynamic_limit_entropy_RL` reproduces the Arriojas entropy-regularized RL paper. The `tensor` branch is intentionally more independent and contains the tensor-network extension route.

## Research Goal

The project extends the Arriojas et al. entropy-regularized RL / large-deviation framework from finite maze-like MDPs to controlled many-body stochastic lattice systems with a thermodynamic-limit interpretation.

The conceptual path is:

```text
finite entropy-regularized MDP
→ state-action tilted operator K_beta
→ Perron/Doob transform
→ controlled many-body stochastic chain
→ tensor-network representation of K_beta, u, v
→ scalable finite-L and thermodynamic-limit algorithms
```

The key research point is **not** to make the original maze larger. Instead, the model is a one-dimensional controlled hard-core particle chain where state-action pairs can be encoded as local symbols and represented by MPS/MPO structures.

## Current Model

The current controlled chain is implemented in:

```text
controlled_chain_experiment.py
```

State:

```math
n=(n_1,\dots,n_L), \qquad n_i\in\{0,1\}, \qquad \sum_i n_i=N.
```

Action:

```math
a=(i,\sigma), \qquad i=1,\dots,L-1, \qquad \sigma\in\{+,-\}.
```

Currently this is a **single-action-marker** model: each time step attempts one local bond move. The action is encoded as a marker string

```math
m_j\in\{0,+,-\},
```

with exactly one nonzero marker, except the last site cannot host a marker.

Local state-action physical index:

```math
y_j=(n_j,m_j), \qquad d_{loc}=2\times 3=6.
```

Reward:

```math
r(n',a)= -V\sum_i n'_i n'_{i+1}+\lambda J-c_\sigma.
```

Prior policy is currently uniform:

```math
\pi_0(a\mid n)=1/|\mathcal A|.
```

Tilted state-action kernel:

```math
K_\beta(n',a'\mid n,a)
= p(n'\mid n,a)\,\pi_0(a'\mid n')\,\exp[\beta r(n,a,n')].
```

Dense matrix convention in code:

```math
K[row=z', col=z]=K_\beta(z'\mid z).
```

Therefore the right continuation Perron eigenfunction obeys:

```math
K^\top u=\rho u.
```

The left Perron weight obeys:

```math
K v=\rho v.
```

Doob stationary state-action distribution:

```math
\mu^*(z)\propto v(z)u(z).
```

## Current Tensor Representation

### MPS for `u` and `v`

Right eigenfunction:

```math
u_\theta(n,a)=\exp f_\theta(y_1,\dots,y_L).
```

Left eigenfunction:

```math
v_\phi(n,a)=\exp g_\phi(y_1,\dots,y_L).
```

Both `f_theta` and `g_phi` are open-boundary MPS with local dimension 6 and bond dimension `chi`.

Implementation:

```text
MPSFunction
positive_u_from_f
train_model_based
```

`train_model_based` currently uses two independent MPS instances:

```text
u_model = MPSFunction(...)
v_model = MPSFunction(...)
```

It trains:

```math
K^\top u_\theta \approx \rho_u u_\theta,
```

and

```math
K v_\phi \approx \rho_v v_\phi.
```

It reports:

```text
rho_right
rho_left
rho_left_right_rel_mismatch
right_relative_residual
left_relative_residual
u_cosine_with_exact
v_cosine_with_exact
doob_stationary_l1_error
doob_stationary_cosine_with_exact
```

### Automaton MPO for `K_beta`

`ExplicitTiltedMPO` implements the tilted state-action kernel as a finite-state automaton MPO entry contraction.

It reads input/output local symbols:

```math
y_j=(n_j,m_j), \qquad y'_j=(n'_j,m'_j).
```

Its virtual state records:

1. whether the input marker has been seen;
2. whether the active bond gate is pending;
3. whether the output next-action marker has been seen;
4. previous output occupation for nearest-neighbor reward;
5. temporary payload for the active bond move.

Important: the current implementation validates the MPO by evaluating all restricted-sector matrix entries and building a dense matrix. This is useful as a small-system benchmark, but **it is not yet the final scalable tensor algorithm**.

## Current Numerical Results

Main result file:

```text
outputs/controlled_chain_results_L6_bd16_lr.json
```

For `L=6, N=3, chi=16`, current representative results are:

```text
rho_exact = 0.7752221495
rho_right = 0.7748279380
rho_left  = 0.7747603945
rho_left_right_rel_mismatch ≈ 8.7e-5
right_relative_residual ≈ 5.15e-4
left_relative_residual  ≈ 5.52e-4
u_cosine_with_exact ≈ 0.9999988
v_cosine_with_exact ≈ 0.9999980
doob_stationary_cosine_with_exact ≈ 0.9999983
```

The notebook now explicitly shows the difference between `rho_right` and `rho_left`, as well as the average model-based `rho` and sampled `rho`.

## Finite-Time Exact Tilted Ensemble

The code now includes exact finite-time tilted ensemble analysis:

```text
finite_time_tilted_analysis
```

For finite horizon `T`, it computes:

```math
Z_T=\mathbf 1^\top K_\beta^T p_0.
```

Forward/backward vectors:

```math
f_t=K_\beta^t p_0,
\qquad
b_{T-t}=(K_\beta^\top)^{T-t}\mathbf 1.
```

Finite-time time-slice marginal:

```math
\mu_{T,t}(z)=\frac{f_t(z)b_{T-t}(z)}{\sum_y f_t(y)b_{T-t}(y)}.
```

The midpoint `t=T/2` converges to the Doob bulk steady state:

```math
\mu_{T,T/2}\to \mu^*(z)\propto v(z)u(z).
```

Current result illustrates:

```text
T=2:   midpoint L1 to Doob stationary ≈ 0.997
T=16:  midpoint L1 to Doob stationary ≈ 0.093
T=64:  midpoint L1 to Doob stationary ≈ 4.18e-4
T=128: midpoint L1 to Doob stationary ≈ 1.50e-6
```

Start/end finite-time marginals remain far from the bulk Doob steady state because of boundary effects.

## Notebook

Generated notebook:

```text
controlled_chain_workflow.ipynb
```

Rendered HTML:

```text
controlled_chain_workflow.html
```

Notebook structure currently includes:

1. MDP definition and reward;
2. Perron/Doob equation;
3. exact policy visualization;
4. action-marker MPS/MPO construction;
5. right/left MPS workflow;
6. numerical comparisons;
7. finite-time tilted ensemble and bulk convergence;
8. explanation that finite-time calculations can also be tensor-networkized.

## Important Current Limitation

The current code is still a **small-system proof-of-concept**.

Although it builds `K_beta` through a finite-state automaton MPO entry contraction, it still forms a dense restricted matrix for training and exact finite-time propagation:

```text
K_mpo = ExplicitTiltedMPO(chain).build_restricted_matrix()
Ku = K_operator.T @ u
Kv = K_operator @ v
f_{t+1} = K @ f_t
b_{t+1} = K.T @ b_t
```

This is correct for validation, but it is **not** the intended large-system algorithm.

## Next Development Goal: Real Large-System Tensor Algorithm

The next development step should **not** be more dense exact matrix work.

The next step is to implement a genuinely scalable tensor-network algorithm for larger systems / thermodynamic-limit behavior.

## Current Dense-Free MPO-MPS Development Notes

The new local development path is now:

```text
tensor_mpo_mps.py
run_tensor_mpo_mps.py
bin/qsub_tensor_mpo_mps.sh
```

This is the model-based tensor-network route, not the sampled `u_theta`
prototype.  The first implementation uses full local `d=6` MPS tensors, while
initial states and target finite-size runs remain half-filled:

```text
N = L / 2
```

Local laptop usage rule:

```text
Use the local laptop only for unit tests and short L<=6 validation.
Do not run larger finite-size sweeps locally.
```

W003 usage rule:

```bash
cd ~/thermodynamic_limit_entropy_RL_tensor_upload/tensor_version
qsub -q cmt -l nodes=1:ppn=24,walltime=00:10:00 -j oe -N tensor_l8_smoke \
  -- /bin/bash -lc 'cd ~/thermodynamic_limit_entropy_RL_tensor_upload/tensor_version && ~/miniconda3/envs/thermodynamic_limit_entropy_rl/bin/python run_tensor_mpo_mps.py --L 8 --N 4 --chi-max 32 --steps 1 --out outputs/tensor_mpo_mps_L8_chi32_smoke_cmd.json'

qsub -q cmt -l nodes=1:ppn=24,walltime=02:00:00 -j oe -N tensor_l8_chi64 \
  -- /bin/bash -lc 'cd ~/thermodynamic_limit_entropy_RL_tensor_upload/tensor_version && ~/miniconda3/envs/thermodynamic_limit_entropy_rl/bin/python run_tensor_mpo_mps.py --L 8 --N 4 --chi-max 64 --steps 50 --out outputs/tensor_mpo_mps_L8_chi64.json'
```

The script uses the `cmt` queue with `nodes=1:ppn=24`, because W003 queues
currently reject one-core PBS requests.  On this W003 setup, `qsub -v` did not
propagate variables into script jobs reliably, so parameterized runs should use
PBS command mode as shown above.

The qsub runner defaults to:

```text
~/miniconda3/envs/thermodynamic_limit_entropy_rl/bin/python
```

The new runner avoids importing `controlled_chain_experiment.py`, because W003's
available conda environments currently have NumPy/SciPy but not Torch.  Dense
exact validation remains local/small-system only.

Target direction:

```text
MPO-MPS / TEBD-like / transfer-matrix time-evolution algorithms
```

The goal is to avoid explicitly constructing the dense `K_beta` matrix.

### Required next capabilities

1. **Explicit local MPO tensors or automaton-MPO apply**

   Replace `build_restricted_matrix()` with a true MPO object or automaton-MPO apply routine.

   Needed operation:

   ```math
   |w\rangle = K_\beta^\top |u\rangle
   ```

   and

   ```math
   |q\rangle = K_\beta |v\rangle.
   ```

   These should output MPS/MPO-contracted states, not dense vectors.

2. **MPO-MPS application with compression**

   Applying an MPO to an MPS increases bond dimension roughly:

   ```math
   \chi \to D_{MPO}\chi.
   ```

   Need compression back to target `chi` through SVD sweeps or variational fitting.

3. **Right/left Perron fixed-point solver**

   Implement one of:

   - power method + compression;
   - variational residual minimization;
   - DMRG-like local optimization;
   - VUMPS-like fixed point for translationally invariant iMPS later.

   Must solve both:

   ```math
   K_\beta^\top u=\rho u,
   \qquad
   K_\beta v=\rho v.
   ```

4. **Finite-time large-system calculation**

   Implement finite-time tilted ensemble without dense powers:

   ```math
   f_{t+1}=K_\beta f_t,
   \qquad
   b_{t+1}=K_\beta^\top b_t.
   ```

   Here `f_t` and `b_t` should be MPS, evolved by MPO-MPS application and compression.

   Then compute bulk slice:

   ```math
   \mu_{T,t}(y)\propto f_t(y)b_{T-t}(y)
   ```

   through MPS contractions, not dense multiplication.

5. **Larger finite-size sequence**

   Once true MPO-MPS algorithms work, run:

   ```math
   L=8,10,12,16,24,\dots
   ```

   and examine:

   ```math
   \frac{1}{L}\log\rho_{\beta,L},
   \quad
   \langle n_i n_{i+1}\rangle,
   \quad
   J/L,
   \quad
   \text{correlation length}.
   ```

6. **Eventually iMPS/iMPO**

   If the controlled model is made translation-invariant in the bulk, implement infinite-chain fixed point:

   ```text
iMPS/iMPO, one-site or two-site unit cell
```

   This would be the true thermodynamic-limit version.

### Suggested implementation plan for the next session

1. Create a new module, for example:

   ```text
tensor_mpo_mps.py
```

2. Implement a minimal dense-free MPS class or use a small existing tensor-network library only if already available.

3. Start with finite open-boundary MPS tensors:

   ```math
   A^{[j]}_{\alpha_{j-1}, y_j, \alpha_j}
   ```

4. Implement canonicalization and SVD compression.

5. Implement a true MPO representation of the current automaton, or an automaton-MPO apply that contracts local physical indices sequentially.

6. Validate against current dense small-system results for `L=6,N=3`:

   - `rho`;
   - right/left residual;
   - finite-time midpoint marginal;
   - observables.

7. Only after validation, scale to larger `L`.

## Important Conceptual Reminder

For each finite `L`, Perron-Frobenius justifies the long-time limit if the state-action chain is finite, irreducible, aperiodic, and the reward is bounded.

The thermodynamic limit is a separate question:

```math
\lim_{L\to\infty}\frac{1}{L}\log\rho_{\beta,L}
```

is not automatically guaranteed by finite-matrix Perron theory. The tensor-network project is precisely to study this limit by exploiting locality and low-bond-dimension structure.

## Files to Read First in a New Session

1. `HANDOFF.md` — this file.
2. `controlled_chain_experiment.py` — current exact/MPS/dense benchmark implementation.
3. `make_controlled_chain_notebook.py` — source for the explanatory notebook.
4. `controlled_chain_workflow.ipynb` or `.html` — human-readable report.
5. `outputs/controlled_chain_results_L6_bd16_lr.json` — current metrics.

## Git / Sync Notes

Latest pushed tensor branch includes finite-time exact tilted ensemble comparison.

Before continuing, check:

```bash
git status --short --branch
```

On W003 clean tensor clone:

```bash
cd ~/thermodynamic_limit_entropy_RL_tensor_upload
git status --short --branch
```

Do not accidentally modify the dirty original W003 `~/thermodynamic_limit_entropy_RL` main working tree unless intentionally cleaning up that old reproduction project.

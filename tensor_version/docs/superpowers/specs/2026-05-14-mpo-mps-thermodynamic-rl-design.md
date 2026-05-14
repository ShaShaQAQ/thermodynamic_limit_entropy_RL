# Dense-Free MPO-MPS Thermodynamic-Limit RL Design

## Context

The current project studies entropy-regularized reinforcement learning and large deviations for a controlled one-dimensional hard-core particle chain.  The existing finite-size benchmark in `controlled_chain_experiment.py` has three useful pieces:

- an exact finite Markov state-action model for small `L`;
- an automaton description of the tilted kernel `K_beta(y' | y)`;
- MPS ansatz functions for the right and left Perron objects.

The current limitation is that the automaton MPO is still validated by building a dense restricted matrix:

```text
K_mpo = ExplicitTiltedMPO(chain).build_restricted_matrix()
Ku = K_mpo.T @ u
Kv = K_mpo @ v
```

This is correct for `L=6, N=3` validation, but it is not a scalable tensor-network algorithm.  The next development should therefore implement a true dense-free MPO-MPS / TEBD-like route.

## Scope

The first implementation target is a finite open-boundary chain at half filling:

```text
L = 6, 8, 10, 12, 16, ...
N = L / 2
```

The physical model remains the same controlled hard-core chain with local state-action symbols:

```text
y_i = (n_i, m_i), n_i in {0, 1}, m_i in {0, +, -}
d_local = 6
```

The first tensor implementation will use the full local Hilbert space rather than a block-sparse fixed-`N` basis.  The initial state and target comparisons are half-filled, and the MPO dynamics should preserve particle number.  Because the MPS representation itself is not explicitly `U(1)` constrained, every large-system run must report particle-number diagnostics and marker-legality diagnostics.

## Non-Goals For The First Pass

The first pass will not implement:

- a dense restricted `K_beta` matrix for large systems;
- direct `U(1)` / fixed-particle-number block-sparse MPS;
- infinite-chain iMPS/iMPO or VUMPS;
- production sampled `u_theta` learning.

Sampled `u_theta` learning remains an important later milestone, but the immediate priority is to build a reliable model-based tensor-network backbone.

## Recommended Approach

Implement an in-repository minimal finite-chain tensor-network module, because the available local conda environment has `numpy`, `scipy`, and `torch`, but not `quimb`, `TeNPy`, or `tensornetwork`.

The new module should be separate from the current dense benchmark code:

```text
tensor_mpo_mps.py
```

The module should expose:

```text
FiniteMPS
AutomatonMPO or TiltedAutomatonOperator
apply_mpo_to_mps
compress_mps
power_method_right_left
finite_time_evolution
diagnostics / observables
```

The existing dense code remains as a validation oracle for small systems.

## Architecture

### FiniteMPS

`FiniteMPS` represents an open-boundary MPS:

```text
A[i].shape = (chi_left, d_local, chi_right)
```

Required operations:

- construct product states and random positive states;
- convert to a dense vector only for small validation sizes;
- compute norms and overlaps by MPS contraction;
- canonicalize with left and right QR/SVD sweeps;
- compress to a target bond dimension using SVD truncation;
- compute local observables by contraction.

The dense conversion method must be clearly marked as a validation-only helper.

### Tilted Automaton Operator

The current `ExplicitTiltedMPO.entry(output_features, input_features)` is the trusted local-rule reference.  The new tensor implementation should start by reusing this automaton logic in a dense-free apply routine rather than constructing the full matrix.

The operator must support both orientations:

```text
K_beta   |v>
K_beta^T |u>
```

The first implementation can represent the automaton as a sparse finite-state transducer whose virtual state is the same information used by `ExplicitTiltedMPO`:

- input marker phase;
- pending active-bond move;
- output marker seen flag;
- previous output occupation for nearest-neighbor reward;
- active-gate payload.

This transducer is then contracted site by site with an input MPS to produce an output MPS.  Applying it increases the MPS bond dimension by the automaton bond dimension, so the output must be compressed after each application.

### MPO-MPS Apply And Compression

The core primitive is:

```text
new_mps = apply_operator_to_mps(operator, old_mps, transpose=False)
new_mps = compress_mps(new_mps, chi_max, cutoff)
```

`transpose=False` means applying `K_beta`; `transpose=True` means applying `K_beta^T`.

The first pass can use exact two-layer local contraction followed by SVD compression sweeps.  If the automaton-MPS contraction is easier to express as local sparse tensors, the implementation can materialize small local transition tensors, but must not materialize the global `6^L x 6^L` operator.

### Perron Solver

Use compressed power iteration first:

```text
u_{k+1} = normalize(compress(K_beta^T u_k))
v_{k+1} = normalize(compress(K_beta v_k))
rho_u   = overlap(u_k, K_beta^T u_k) / overlap(u_k, u_k)
rho_v   = overlap(v_k, K_beta v_k) / overlap(v_k, v_k)
```

The solver reports both right and left estimates:

```text
rho_right
rho_left
rho_left_right_rel_mismatch
right_update_residual
left_update_residual
```

For `L=6, N=3`, it must compare against the existing dense benchmark:

```text
rho_exact
u cosine / overlap against exact, if dense conversion is enabled
v cosine / overlap against exact, if dense conversion is enabled
Doob stationary comparison, if dense conversion is enabled
```

For larger `L`, it reports tensor-only diagnostics:

```text
(1/L) log rho_L
particle number mean and variance
illegal marker weight estimate
nearest-neighbor occupancy density
current density
compression discarded weight
max bond dimension reached
```

### Finite-Time Tilted Ensemble

After the Perron solver works, implement dense-free finite-time evolution:

```text
f_{t+1} = compress(K_beta f_t)
b_{h+1} = compress(K_beta^T b_h)
```

The midpoint tilted marginal is represented through the product of forward and backward MPS:

```text
mu_{T,t}(y) proportional to f_t(y) b_{T-t}(y)
```

For `L=6, N=3`, this should reproduce the dense finite-time benchmark.  For larger `L`, it should report midpoint observables and the finite-time SCGF estimate:

```text
(1/T) log Z_T
```

### Sampled Learning Path

Sampled `u_theta` learning is a later stage, not part of the first implementation pass.  The model-based tensor solver should be designed so that sampled learning can reuse:

- the `FiniteMPS` representation;
- normalization and gauge utilities;
- observables;
- dense-free validation metrics where available;
- model-based large-`L` results as a teacher or benchmark.

The later sampled stage should improve the current prototype with:

- replay buffer;
- target network or delayed target MPS;
- explicit gauge normalization;
- comparison against model-based MPO-MPS `u`, `rho`, policy, and observables;
- optional left-vector or occupancy-ratio learning if Doob stationary estimates are needed from samples.

## Data Flow

Small-system validation:

```text
ControlledHardCoreChain(L=6, N=3)
    -> existing dense K and exact Perron pair
    -> new dense-free MPO-MPS power iteration
    -> compare rho, u, v, Doob stationary distribution, observables
```

Large-system sequence:

```text
for L in [8, 10, 12, 16, ...]:
    N = L / 2
    initialize half-filled MPS boundary/state
    run compressed right/left power iteration
    record rho_L, log(rho_L)/L, observables, diagnostics
```

Finite-time sequence:

```text
initialize p0 MPS
forward evolve f_t with K_beta
backward evolve b_h with K_beta^T
contract f_t * b_{T-t} for midpoint observables
compare dense L=6 first, then scale L
```

## Error Handling And Diagnostics

The solver should fail early when:

- tensors contain `NaN` or `Inf`;
- SVD compression loses more than a configured discarded-weight threshold;
- marker legality diagnostics become too large;
- particle-number variance grows unexpectedly from a half-filled initialization;
- left and right rho estimates drift beyond a configured tolerance after convergence.

Every result file should include the algorithm settings:

```text
L, N, beta, V, lambda, q_plus, q_minus, chi_max, cutoff, steps, seed
```

and the convergence history:

```text
rho_right_history
rho_left_history
residual_history
discarded_weight_history
diagnostic_history
```

## Testing And Validation

The first implementation should add focused tests or validation scripts for:

- MPS dense conversion and overlap on small `L`;
- compression preserving product states and simple random states;
- automaton-MPS apply versus `ExplicitTiltedMPO.build_restricted_matrix()` for `L=4` or `L=6`;
- right/left power iteration matching dense `rho` for `L=6, N=3`;
- finite-time evolution matching dense midpoint convergence for small horizons.

The dense matrix is allowed only inside these small validation tests and benchmark scripts.

## Milestones

1. Minimal `FiniteMPS` utilities with compression and small dense conversion.
2. Dense-free automaton-MPS apply for `K_beta` and `K_beta^T`.
3. Compressed right/left Perron power solver validated on `L=6, N=3`.
4. Dense-free finite-time tilted ensemble validated on small systems.
5. Half-filled finite-size scaling runs for `L=8, 10, 12, 16, ...`.
6. Sampled `u_theta` learning rebuilt on top of the same MPS infrastructure.
7. Optional upgrade to `U(1)` block-sparse fixed-`N` MPS or iMPS/iMPO.

## Open Implementation Choice

The main implementation choice is how to express the first dense-free operator apply:

- build explicit local sparse MPO tensors from the automaton state graph; or
- implement a streaming automaton-MPS contraction that never exposes local MPO tensors as a public object.

The recommended first implementation is the streaming automaton-MPS contraction, because it is closest to the already validated `ExplicitTiltedMPO.entry` logic.  Once validated, it can be refactored into an explicit local MPO tensor object if that makes TEBD-style sweeps or documentation clearer.

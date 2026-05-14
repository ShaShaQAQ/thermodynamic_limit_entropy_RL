# Dense-Free MPO-MPS Thermodynamic RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first dense-free model-based MPO-MPS solver for the controlled hard-core chain, validated against the existing dense `L=6, N=3` benchmark and prepared for half-filled finite-size scaling on W003/qsub.

**Architecture:** Add a focused `tensor_mpo_mps.py` module with finite open-boundary MPS utilities, automaton-derived local MPO transitions, MPO-MPS application, SVD compression, and compressed right/left power iteration. Keep `controlled_chain_experiment.py` as the dense small-system oracle; large runs must be submitted to W003 rather than run on the local laptop.

**Tech Stack:** Python 3, NumPy/SciPy/Torch already present in `/opt/miniconda3/envs/myenv1`; pytest-style validation scripts; W003/qsub for larger `L` jobs.

---

## File Structure

- Create `tensor_mpo_mps.py`: finite-chain MPS class, legal half-filled state-action MPS constructor, automaton MPO transition builder, MPO-MPS apply, compression, diagnostics, power iteration.
- Create `tests/test_tensor_mpo_mps.py`: lightweight local tests for MPS dense conversion, compression, legal-sector MPS, automaton MPO entries, and one-step MPO-MPS apply. These tests must use small `L` only.
- Create `run_tensor_mpo_mps.py`: command-line runner for small validation and W003 jobs. It writes JSON result files under `outputs/`.
- Create `bin/qsub_tensor_mpo_mps.sh`: W003 PBS/qsub script template for half-filled finite-size runs.
- Modify `HANDOFF.md`: add the new dense-free model-based route, local-vs-W003 testing rule, and current command examples.

## Local Compute Rule

Local laptop runs are limited to small correctness checks:

```text
L <= 6, short iteration counts, unit tests, no long sweeps
```

Runs such as `L=8,10,12,16,...` or long convergence sweeps must run on W003 via qsub. Do not run heavy finite-size scaling locally.

---

### Task 1: FiniteMPS Core

**Files:**
- Create: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tensor_mpo_mps.py`
- Create: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tests/test_tensor_mpo_mps.py`

- [ ] **Step 1: Write failing tests for product MPS, dense conversion, overlap, and compression**

Add this test file:

```python
import numpy as np

from tensor_mpo_mps import FiniteMPS


def test_product_mps_dense_vector_places_single_nonzero_amplitude():
    mps = FiniteMPS.product_state([0, 1, 2], local_dim=3)
    dense = mps.to_dense()
    expected = np.zeros(27)
    expected[0 * 9 + 1 * 3 + 2] = 1.0
    np.testing.assert_allclose(dense, expected)


def test_inner_matches_dense_dot():
    a = FiniteMPS.product_state([0, 1, 0], local_dim=2)
    b = FiniteMPS.product_state([0, 1, 0], local_dim=2)
    c = FiniteMPS.product_state([1, 1, 0], local_dim=2)
    assert np.isclose(a.inner(b), 1.0)
    assert np.isclose(a.inner(c), 0.0)


def test_compress_preserves_small_random_state_dense_vector():
    rng = np.random.default_rng(123)
    mps = FiniteMPS.random(L=4, local_dim=3, bond_dim=5, rng=rng)
    before = mps.to_dense()
    compressed, info = mps.copy().compress(chi_max=12, cutoff=1e-14)
    after = compressed.to_dense()
    np.testing.assert_allclose(after, before, atol=1e-10, rtol=1e-10)
    assert info["max_bond_dim"] <= 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py -q
```

Expected: import failure because `tensor_mpo_mps.py` does not exist yet.

- [ ] **Step 3: Implement `FiniteMPS`**

Implement in `tensor_mpo_mps.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass
class CompressionInfo:
    discarded_weight: float
    max_bond_dim: int


class FiniteMPS:
    def __init__(self, tensors: Iterable[Array]):
        self.tensors = [np.asarray(t, dtype=np.float64).copy() for t in tensors]
        if not self.tensors:
            raise ValueError("FiniteMPS requires at least one tensor")
        for i, tensor in enumerate(self.tensors):
            if tensor.ndim != 3:
                raise ValueError(f"MPS tensor {i} must have shape (left, physical, right)")
        if self.tensors[0].shape[0] != 1 or self.tensors[-1].shape[2] != 1:
            raise ValueError("FiniteMPS expects open boundary dimensions 1 at both ends")
        for i in range(len(self.tensors) - 1):
            if self.tensors[i].shape[2] != self.tensors[i + 1].shape[0]:
                raise ValueError(f"bond mismatch between sites {i} and {i + 1}")

    @property
    def L(self) -> int:
        return len(self.tensors)

    @property
    def local_dim(self) -> int:
        return int(self.tensors[0].shape[1])

    @property
    def bond_dims(self) -> list[int]:
        return [int(self.tensors[0].shape[0])] + [int(t.shape[2]) for t in self.tensors]

    def copy(self) -> "FiniteMPS":
        return FiniteMPS([t.copy() for t in self.tensors])

    @classmethod
    def product_state(cls, symbols: list[int], local_dim: int) -> "FiniteMPS":
        tensors = []
        for y in symbols:
            tensor = np.zeros((1, local_dim, 1), dtype=np.float64)
            tensor[0, int(y), 0] = 1.0
            tensors.append(tensor)
        return cls(tensors)

    @classmethod
    def random(cls, L: int, local_dim: int, bond_dim: int, rng: np.random.Generator) -> "FiniteMPS":
        dims = [1] + [int(bond_dim)] * (int(L) - 1) + [1]
        tensors = [0.1 * rng.normal(size=(dims[i], local_dim, dims[i + 1])) for i in range(L)]
        return cls(tensors)

    def to_dense(self) -> Array:
        state = self.tensors[0][0, :, :]
        for tensor in self.tensors[1:]:
            state = np.tensordot(state, tensor, axes=([-1], [0]))
        return np.asarray(state.reshape(-1), dtype=np.float64)

    def inner(self, other: "FiniteMPS") -> float:
        if self.L != other.L or self.local_dim != other.local_dim:
            raise ValueError("MPS inner product requires matching L and local_dim")
        env = np.array([[1.0]], dtype=np.float64)
        for a, b in zip(self.tensors, other.tensors):
            env = np.einsum("ab,api,bpj->ij", env, a, b, optimize=True)
        return float(env[0, 0])

    def norm(self) -> float:
        return float(np.sqrt(max(self.inner(self), 0.0)))

    def scale(self, factor: float) -> "FiniteMPS":
        self.tensors[0] = self.tensors[0] * float(factor)
        return self

    def normalize(self) -> float:
        nrm = self.norm()
        if not np.isfinite(nrm) or nrm <= 0:
            raise ValueError(f"cannot normalize MPS with norm {nrm}")
        self.scale(1.0 / nrm)
        return nrm

    def compress(self, chi_max: int, cutoff: float = 0.0) -> tuple["FiniteMPS", dict]:
        tensors = [t.copy() for t in self.tensors]
        discarded_total = 0.0
        for site in range(self.L - 1):
            dl, d, dr = tensors[site].shape
            matrix = tensors[site].reshape(dl * d, dr)
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            keep = int(np.sum(s > cutoff))
            keep = max(1, min(keep, int(chi_max), len(s)))
            discarded_total += float(np.sum(s[keep:] ** 2))
            u = u[:, :keep]
            s = s[:keep]
            vh = vh[:keep, :]
            tensors[site] = u.reshape(dl, d, keep)
            transfer = s[:, None] * vh
            tensors[site + 1] = np.tensordot(transfer, tensors[site + 1], axes=([1], [0]))
        info = {
            "discarded_weight": discarded_total,
            "max_bond_dim": max(int(t.shape[2]) for t in tensors),
        }
        self.tensors = tensors
        return self, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py -q
```

Expected: all Task 1 tests pass.

---

### Task 2: Legal Half-Filled State-Action MPS

**Files:**
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tensor_mpo_mps.py`
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tests/test_tensor_mpo_mps.py`

- [ ] **Step 1: Add tests for legal half-filled MPS**

Append:

```python
from controlled_chain_experiment import ControlledHardCoreChain, Params
from tensor_mpo_mps import legal_state_action_mps


def test_legal_state_action_mps_matches_chain_restricted_sector():
    chain = ControlledHardCoreChain(Params(L=4, N=2))
    mps = legal_state_action_mps(L=4, N=2, local_dim=6)
    dense = mps.to_dense()
    expected = np.zeros(6**4)
    for s_idx in range(chain.num_states):
        for a_idx in range(chain.num_actions):
            features = chain.marker_features(s_idx, a_idx)
            idx = 0
            for y in features:
                idx = idx * 6 + y
            expected[idx] = 1.0
    np.testing.assert_allclose(dense, expected)
    assert np.isclose(mps.inner(mps), chain.dim)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py::test_legal_state_action_mps_matches_chain_restricted_sector -q
```

Expected: import failure for `legal_state_action_mps`.

- [ ] **Step 3: Implement legal-sector MPS constructor**

Add helper functions:

```python
def decode_local_symbol(y: int) -> tuple[int, int]:
    return divmod(int(y), 3)


def dense_to_mps(values: Array, L: int, local_dim: int, chi_max: int | None = None, cutoff: float = 0.0) -> FiniteMPS:
    tensor = np.asarray(values, dtype=np.float64).reshape((local_dim,) * L)
    tensors = []
    left_dim = 1
    work = tensor
    for site in range(L - 1):
        matrix = work.reshape(left_dim * local_dim, -1)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        keep = int(np.sum(s > cutoff))
        if chi_max is not None:
            keep = min(keep, int(chi_max))
        keep = max(1, keep)
        tensors.append(u[:, :keep].reshape(left_dim, local_dim, keep))
        work = (s[:keep, None] * vh[:keep, :])
        left_dim = keep
    tensors.append(work.reshape(left_dim, local_dim, 1))
    return FiniteMPS(tensors)


def legal_state_action_mps(L: int, N: int, local_dim: int = 6) -> FiniteMPS:
    if local_dim != 6:
        raise ValueError("legal_state_action_mps currently assumes y=(n,m) with local_dim=6")
    values = np.zeros(local_dim**L, dtype=np.float64)
    for flat in range(local_dim**L):
        tmp = flat
        symbols = [0] * L
        for site in range(L - 1, -1, -1):
            symbols[site] = tmp % local_dim
            tmp //= local_dim
        occ = 0
        markers = 0
        ok = True
        for site, y in enumerate(symbols):
            n, m = decode_local_symbol(y)
            occ += n
            markers += int(m != 0)
            if site == L - 1 and m != 0:
                ok = False
                break
        if ok and occ == N and markers == 1:
            values[flat] = 1.0
    return dense_to_mps(values, L=L, local_dim=local_dim, cutoff=1e-14)
```

- [ ] **Step 4: Run legal-sector test**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py::test_legal_state_action_mps_matches_chain_restricted_sector -q
```

Expected: pass.

---

### Task 3: Automaton MPO Transition Builder

**Files:**
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tensor_mpo_mps.py`
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tests/test_tensor_mpo_mps.py`

- [ ] **Step 1: Add tests comparing automaton entries to existing dense MPO reference**

Append:

```python
from tensor_mpo_mps import TiltedAutomatonMPO


def test_tilted_automaton_mpo_entries_match_explicit_mpo_reference():
    chain = ControlledHardCoreChain(Params(L=4, N=2))
    reference = chain.build_exact_K()
    operator = TiltedAutomatonMPO(chain)
    restricted = operator.build_restricted_matrix_validation()
    np.testing.assert_allclose(restricted, reference, atol=1e-12, rtol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py::test_tilted_automaton_mpo_entries_match_explicit_mpo_reference -q
```

Expected: import failure for `TiltedAutomatonMPO`.

- [ ] **Step 3: Implement `TiltedAutomatonMPO` by lifting existing automaton transition logic**

Implement a class that:

- stores `chain`, `params`, `num_actions`, `local_dim=6`;
- has `decode_feature`, `_pending_branches`, `_finish_pending` matching `ExplicitTiltedMPO`;
- has `step_states(site, states, y_out, y_in)` returning the next automaton-state dictionary;
- has `entry(output_features, input_features)` matching `ExplicitTiltedMPO.entry`;
- has `build_restricted_matrix_validation()` that loops only over the existing legal sector for small validation.

The implementation should reuse the exact state tuple convention:

```text
(phase, out_seen, prev_np, payload)
```

with start state:

```python
(0, 0, -1, ())
```

and accepting states:

```python
phase == 2 and out_seen == 1
```

- [ ] **Step 4: Run automaton entry test**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py::test_tilted_automaton_mpo_entries_match_explicit_mpo_reference -q
```

Expected: pass.

---

### Task 4: Dense-Free MPO-MPS Apply

**Files:**
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tensor_mpo_mps.py`
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tests/test_tensor_mpo_mps.py`

- [ ] **Step 1: Add tests for `K` and `K^T` MPO-MPS application**

Append:

```python
from tensor_mpo_mps import apply_mpo_to_mps


def test_apply_mpo_to_mps_matches_dense_K_on_small_legal_vector():
    chain = ControlledHardCoreChain(Params(L=4, N=2))
    operator = TiltedAutomatonMPO(chain)
    vec = legal_state_action_mps(L=4, N=2)
    applied, _ = apply_mpo_to_mps(operator, vec, transpose=False, chi_max=128, cutoff=1e-14)
    dense_applied = applied.to_dense()

    K = chain.build_exact_K()
    restricted_input = np.ones(chain.dim)
    restricted_output = K @ restricted_input

    expected = np.zeros(6**4)
    for z in range(chain.dim):
        s_idx, a_idx = chain.unpack_z(z)
        features = chain.marker_features(s_idx, a_idx)
        idx = 0
        for y in features:
            idx = idx * 6 + y
        expected[idx] = restricted_output[z]
    np.testing.assert_allclose(dense_applied, expected, atol=1e-10, rtol=1e-10)


def test_apply_mpo_to_mps_matches_dense_KT_on_small_legal_vector():
    chain = ControlledHardCoreChain(Params(L=4, N=2))
    operator = TiltedAutomatonMPO(chain)
    vec = legal_state_action_mps(L=4, N=2)
    applied, _ = apply_mpo_to_mps(operator, vec, transpose=True, chi_max=128, cutoff=1e-14)
    dense_applied = applied.to_dense()

    K = chain.build_exact_K()
    restricted_input = np.ones(chain.dim)
    restricted_output = K.T @ restricted_input

    expected = np.zeros(6**4)
    for z in range(chain.dim):
        s_idx, a_idx = chain.unpack_z(z)
        features = chain.marker_features(s_idx, a_idx)
        idx = 0
        for y in features:
            idx = idx * 6 + y
        expected[idx] = restricted_output[z]
    np.testing.assert_allclose(dense_applied, expected, atol=1e-10, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py::test_apply_mpo_to_mps_matches_dense_K_on_small_legal_vector tests/test_tensor_mpo_mps.py::test_apply_mpo_to_mps_matches_dense_KT_on_small_legal_vector -q
```

Expected: import failure for `apply_mpo_to_mps`.

- [ ] **Step 3: Implement dense-free apply**

Implement `apply_mpo_to_mps(operator, mps, transpose, chi_max, cutoff)` as a site-by-site automaton-MPS contraction:

```python
def apply_mpo_to_mps(operator, mps: FiniteMPS, transpose: bool, chi_max: int, cutoff: float):
    # left map keys are automaton states, values are left-boundary blocks
    # Each site combines one MPS tensor with all local y_in/y_out transition weights.
    # For K: output physical index is y_out and contracted physical index is y_in.
    # For K^T: output physical index is y_in and contracted physical index is y_out.
    # The resulting raw tensors have bonds carrying (automaton_state, mps_bond).
    # After the final accepting boundary, compress and return.
```

The implementation must not build a global `6^L x 6^L` matrix.  It may enumerate local `y_in, y_out in range(6)` and finite automaton states at each bond.

- [ ] **Step 4: Run apply tests**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py::test_apply_mpo_to_mps_matches_dense_K_on_small_legal_vector tests/test_tensor_mpo_mps.py::test_apply_mpo_to_mps_matches_dense_KT_on_small_legal_vector -q
```

Expected: pass.

---

### Task 5: Power Iteration Solver And Diagnostics

**Files:**
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tensor_mpo_mps.py`
- Create: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/run_tensor_mpo_mps.py`
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/tests/test_tensor_mpo_mps.py`

- [ ] **Step 1: Add a small convergence validation test**

Append:

```python
from tensor_mpo_mps import power_method_right_left


def test_power_method_gets_reasonable_small_system_rho():
    chain = ControlledHardCoreChain(Params(L=4, N=2))
    rho_exact, *_ = __import__("controlled_chain_experiment").dominant_exact_pair(chain.build_exact_K())
    result = power_method_right_left(
        chain,
        chi_max=128,
        cutoff=1e-13,
        steps=20,
        dense_validation=True,
    )
    assert abs(result["rho_right"] - rho_exact) / rho_exact < 5e-3
    assert abs(result["rho_left"] - rho_exact) / rho_exact < 5e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py::test_power_method_gets_reasonable_small_system_rho -q
```

Expected: import failure for `power_method_right_left`.

- [ ] **Step 3: Implement compressed power method**

Implement:

```python
def power_method_right_left(chain, chi_max: int, cutoff: float, steps: int, dense_validation: bool = False) -> dict:
    operator = TiltedAutomatonMPO(chain)
    u = legal_state_action_mps(chain.p.L, chain.p.N)
    v = legal_state_action_mps(chain.p.L, chain.p.N)
    u.normalize()
    v.normalize()
    history = []
    for step in range(steps):
        Ku, info_u = apply_mpo_to_mps(operator, u, transpose=True, chi_max=chi_max, cutoff=cutoff)
        Kv, info_v = apply_mpo_to_mps(operator, v, transpose=False, chi_max=chi_max, cutoff=cutoff)
        rho_u = Ku.norm()
        rho_v = Kv.norm()
        Ku.scale(1.0 / rho_u)
        Kv.scale(1.0 / rho_v)
        right_update_residual = (Ku.to_dense() - u.to_dense()).dot(Ku.to_dense() - u.to_dense()) ** 0.5 if dense_validation else float("nan")
        left_update_residual = (Kv.to_dense() - v.to_dense()).dot(Kv.to_dense() - v.to_dense()) ** 0.5 if dense_validation else float("nan")
        u, v = Ku, Kv
        history.append({
            "step": step + 1,
            "rho_right": float(rho_u),
            "rho_left": float(rho_v),
            "right_update_residual": float(right_update_residual),
            "left_update_residual": float(left_update_residual),
            "discarded_weight_right": info_u["discarded_weight"],
            "discarded_weight_left": info_v["discarded_weight"],
        })
    return {
        "rho_right": history[-1]["rho_right"],
        "rho_left": history[-1]["rho_left"],
        "rho_left_right_rel_mismatch": abs(history[-1]["rho_right"] - history[-1]["rho_left"]) / max(abs(0.5 * (history[-1]["rho_right"] + history[-1]["rho_left"])), 1e-14),
        "history": history,
    }
```

Use dense conversion only when `dense_validation=True`; the W003 large run path must keep this flag false.

- [ ] **Step 4: Add CLI runner**

Create `run_tensor_mpo_mps.py` with arguments:

```text
--L
--N
--chi-max
--cutoff
--steps
--dense-validation
--out
```

It should construct `Params(L=args.L, N=args.N)`, call `power_method_right_left`, and write JSON to `args.out`.

- [ ] **Step 5: Run local small validation only**

Run:

```bash
/opt/miniconda3/envs/myenv1/bin/python -m pytest tests/test_tensor_mpo_mps.py -q
/opt/miniconda3/envs/myenv1/bin/python run_tensor_mpo_mps.py --L 4 --N 2 --chi-max 128 --steps 20 --dense-validation --out outputs/tensor_mpo_mps_L4_validation.json
```

Expected: tests pass and JSON output is written.

---

### Task 6: W003 qsub Runner For Larger Half-Filled Runs

**Files:**
- Create: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/bin/qsub_tensor_mpo_mps.sh`
- Modify: `/Users/shajianyu/CMP_manybody/Quantum_AI/final_project/HANDOFF.md`

- [ ] **Step 1: Add qsub script**

Create:

```bash
#!/usr/bin/env bash
#PBS -N tensor_mpo_mps
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00

set -euo pipefail

cd "$HOME/thermodynamic_limit_entropy_RL_tensor_upload"

PYTHON=${PYTHON:-/opt/miniconda3/envs/myenv1/bin/python}
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
```

- [ ] **Step 2: Document local/W003 workflow in HANDOFF**

Add a short section:

```text
Large MPO-MPS tests should not run on the local laptop.  Sync local files to W003 clean tensor clone, then submit:

qsub -v L=8,CHI_MAX=64,STEPS=50 bin/qsub_tensor_mpo_mps.sh
qsub -v L=10,CHI_MAX=64,STEPS=50 bin/qsub_tensor_mpo_mps.sh

Use local laptop only for unit tests and L<=6 short validation.
```

- [ ] **Step 3: Sync to W003 and submit only after local tests pass**

Use the existing W003 clone path from `HANDOFF.md`:

```text
~/thermodynamic_limit_entropy_RL_tensor_upload
```

Do not touch the older dirty W003 main working tree.

Run W003 status before submit:

```bash
ssh W003 'cd ~/thermodynamic_limit_entropy_RL_tensor_upload && git status --short --branch'
```

Then sync files and submit parameterized runs with PBS command mode:

```bash
ssh W003 'cd ~/thermodynamic_limit_entropy_RL_tensor_upload/tensor_version; qsub -q cmt -l nodes=1:ppn=24,walltime=00:10:00 -j oe -N tensor_l8_smoke -- /bin/bash -lc "cd ~/thermodynamic_limit_entropy_RL_tensor_upload/tensor_version && ~/miniconda3/envs/thermodynamic_limit_entropy_rl/bin/python run_tensor_mpo_mps.py --L 8 --N 4 --chi-max 32 --steps 1 --out outputs/tensor_mpo_mps_L8_chi32_smoke_cmd.json"'
```

Expected: qsub returns a job id.

---

## Plan Self-Review

- Spec coverage: tasks cover finite MPS, legal half-filled initialization, automaton operator, dense-free `K`/`K^T` apply, right/left power iteration, local validation, and W003/qsub workflow.
- Sampled learning: intentionally not implemented in this plan; it remains a later milestone after the model-based tensor solver works.
- Dense matrix usage: allowed only in small tests and dense-validation CLI mode.
- Compute safety: large `L` runs are explicitly assigned to W003/qsub, not the local laptop.

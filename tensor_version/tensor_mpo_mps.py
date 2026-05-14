#!/usr/bin/env python3
"""Dense-free finite-chain tensor-network utilities for the controlled chain."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TensorParams:
    L: int = 6
    N: int = 3
    beta: float = 1.0
    V: float = 0.7
    lam: float = 0.25
    c_plus: float = 0.03
    c_minus: float = 0.03
    q_plus: float = 0.9
    q_minus: float = 0.9
    seed: int = 7


@dataclass
class CompressionInfo:
    discarded_weight: float
    max_bond_dim: int


class ControlledHardCoreChainLite:
    """Torch-free controlled hard-core chain used by dense-free tensor runs."""

    def __init__(self, params: TensorParams):
        self.p = params
        self.states = self._make_states(params.L, params.N)
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.actions = [(i, sig) for i in range(params.L - 1) for sig in (+1, -1)]
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        self.dim = self.num_states * self.num_actions

    @staticmethod
    def _make_states(L: int, N: int) -> list[tuple[int, ...]]:
        out = []
        for occ in combinations(range(L), N):
            n = [0] * L
            for i in occ:
                n[i] = 1
            out.append(tuple(n))
        return out

    def z_index(self, s_idx: int, a_idx: int) -> int:
        return s_idx * self.num_actions + a_idx

    def unpack_z(self, z: int) -> tuple[int, int]:
        return divmod(z, self.num_actions)

    def prior_prob(self, _s_idx: int, _a_idx: int) -> float:
        return 1.0 / self.num_actions

    def transition_outcomes(self, s_idx: int, a_idx: int) -> list[tuple[int, float, int]]:
        n = list(self.states[s_idx])
        bond, sig = self.actions[a_idx]
        i = bond
        j = bond + 1
        q = self.p.q_plus if sig == +1 else self.p.q_minus

        if sig == +1 and n[i] == 1 and n[j] == 0:
            moved = n.copy()
            moved[i], moved[j] = 0, 1
            return [
                (self.state_index[tuple(moved)], q, +1),
                (s_idx, 1.0 - q, 0),
            ]
        if sig == -1 and n[i] == 0 and n[j] == 1:
            moved = n.copy()
            moved[i], moved[j] = 1, 0
            return [
                (self.state_index[tuple(moved)], q, -1),
                (s_idx, 1.0 - q, 0),
            ]
        return [(s_idx, 1.0, 0)]

    def phi(self, s_idx: int) -> float:
        n = self.states[s_idx]
        return -self.p.V * sum(n[i] * n[i + 1] for i in range(self.p.L - 1))

    def reward(self, next_s_idx: int, a_idx: int, current: int) -> float:
        _, sig = self.actions[a_idx]
        c = self.p.c_plus if sig == +1 else self.p.c_minus
        return self.phi(next_s_idx) + self.p.lam * current - c

    def build_exact_K(self) -> Array:
        K = np.zeros((self.dim, self.dim), dtype=np.float64)
        for s_idx in range(self.num_states):
            for a_idx in range(self.num_actions):
                col = self.z_index(s_idx, a_idx)
                for ns_idx, prob, current in self.transition_outcomes(s_idx, a_idx):
                    w = prob * math.exp(self.p.beta * self.reward(ns_idx, a_idx, current))
                    for na_idx in range(self.num_actions):
                        row = self.z_index(ns_idx, na_idx)
                        K[row, col] += w * self.prior_prob(ns_idx, na_idx)
        return K

    def marker_features(self, s_idx: int, a_idx: int) -> list[int]:
        n = self.states[s_idx]
        bond, sig = self.actions[a_idx]
        ys = []
        for i, occ in enumerate(n):
            marker = 0
            if i == bond:
                marker = 1 if sig == +1 else 2
            ys.append(occ * 3 + marker)
        return ys


class FiniteMPS:
    """Open-boundary finite MPS with tensors shaped (left, physical, right)."""

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
        """Return the full dense vector.  Use only for small validation systems."""
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


def decode_local_symbol(y: int) -> tuple[int, int]:
    """Decode y=(n,m) with marker m in {0,+,-} encoded as occ*3+marker."""
    return divmod(int(y), 3)


def legal_state_action_mps(L: int, N: int, local_dim: int = 6) -> FiniteMPS:
    """Return an exact MPS with amplitude 1 on legal fixed-N state-action strings.

    The construction is a finite automaton over the full local d=6 space.  Its
    bonds track (particles seen, action marker seen), so it avoids dense 6**L
    enumeration and can be used for the larger half-filled finite chains.
    """
    if local_dim != 6:
        raise ValueError("legal_state_action_mps currently assumes y=(n,m) with local_dim=6")
    if not (0 <= N <= L):
        raise ValueError(f"N must satisfy 0 <= N <= L, got L={L}, N={N}")

    bond_states: list[list[tuple[int, int]]] = [[(0, 0)]]
    transitions: list[dict[tuple[int, int, int], tuple[int, int]]] = []
    accept = (int(N), 1)

    for site in range(L):
        current_states = bond_states[-1]
        next_states_set: set[tuple[int, int]] = set()
        site_transitions: dict[tuple[int, int, int], tuple[int, int]] = {}
        for left_idx, (count, marker_seen) in enumerate(current_states):
            for y in range(local_dim):
                occ, marker = decode_local_symbol(y)
                if site == L - 1 and marker != 0:
                    continue
                next_count = count + occ
                next_marker_seen = marker_seen + int(marker != 0)
                if next_count > N or next_marker_seen > 1:
                    continue
                next_state = (next_count, next_marker_seen)
                if site == L - 1 and next_state != accept:
                    continue
                site_transitions[(left_idx, y, -1)] = next_state
                next_states_set.add(next_state)
        if not next_states_set:
            raise ValueError(f"no legal state-action strings for L={L}, N={N}")
        next_states = sorted(next_states_set)
        if site == L - 1:
            next_states = [accept]
        bond_states.append(next_states)
        transitions.append(site_transitions)

    tensors = []
    for site, site_transitions in enumerate(transitions):
        left_states = bond_states[site]
        right_states = bond_states[site + 1]
        right_index = {state: i for i, state in enumerate(right_states)}
        tensor = np.zeros((len(left_states), local_dim, len(right_states)), dtype=np.float64)
        for (left_idx, y, _), right_state in site_transitions.items():
            if right_state in right_index:
                tensor[left_idx, y, right_index[right_state]] = 1.0
        tensors.append(tensor)

    return FiniteMPS(tensors)


class TiltedAutomatonMPO:
    """Finite-state automaton for K_beta(y_out | y_in)."""

    def __init__(self, chain):
        self.chain = chain
        self.p = chain.p
        self.num_actions = chain.num_actions
        self.local_dim = 6
        self._transition_layers_cache = None

    @staticmethod
    def decode_feature(y: int) -> tuple[int, int]:
        return decode_local_symbol(y)

    def _pending_branches(
        self, sig: int, n_left: int, np_left: int
    ) -> list[tuple[str, float, int]]:
        branches: list[tuple[str, float, int]] = []
        if sig == +1:
            if n_left == 1:
                if np_left == 0:
                    branches.append(("right_success", self.p.q_plus, +1))
                if np_left == 1:
                    branches.append(("stay_right", 1.0, 0))
            elif np_left == n_left:
                branches.append(("stay_invalid", 1.0, 0))
        else:
            if n_left == 0:
                if np_left == 1:
                    branches.append(("left_success", self.p.q_minus, -1))
                if np_left == 0:
                    branches.append(("stay_left", 1.0, 0))
            elif np_left == n_left:
                branches.append(("stay_invalid", 1.0, 0))
        return branches

    def _finish_pending(
        self, branch: str, prob: float, current: int, sig: int, n_right: int, np_right: int
    ) -> list[tuple[float, int]]:
        if branch == "right_success":
            return [(prob, current)] if n_right == 0 and np_right == 1 else []
        if branch == "left_success":
            return [(prob, current)] if n_right == 1 and np_right == 0 else []
        if branch == "stay_right":
            if n_right == 0 and np_right == n_right:
                return [(1.0 - self.p.q_plus, 0)]
            if n_right == 1 and np_right == n_right:
                return [(1.0, 0)]
            return []
        if branch == "stay_left":
            if n_right == 1 and np_right == n_right:
                return [(1.0 - self.p.q_minus, 0)]
            if n_right == 0 and np_right == n_right:
                return [(1.0, 0)]
            return []
        if branch == "stay_invalid":
            return [(1.0, 0)] if np_right == n_right else []
        raise ValueError(f"unknown branch {branch}")

    def step_states(
        self,
        site: int,
        states: dict[tuple[int, int, int, tuple], float],
        y_out: int,
        y_in: int,
    ) -> dict[tuple[int, int, int, tuple], float]:
        L = self.p.L
        n, m = self.decode_feature(y_in)
        np_occ, mp = self.decode_feature(y_out)
        next_states: dict[tuple[int, int, int, tuple], float] = {}

        if site == L - 1 and (m != 0 or mp != 0):
            return next_states

        for (phase, out_seen, prev_np, payload), weight in states.items():
            new_out_seen = out_seen + (1 if mp != 0 else 0)
            if new_out_seen > 1:
                continue
            reward_weight = 1.0
            if prev_np >= 0:
                reward_weight = math.exp(-self.p.beta * self.p.V * prev_np * np_occ)

            if phase == 0:
                if m == 0:
                    if np_occ != n:
                        continue
                    key = (0, new_out_seen, np_occ, ())
                    next_states[key] = next_states.get(key, 0.0) + weight * reward_weight
                else:
                    sig = +1 if m == 1 else -1
                    if site >= L - 1:
                        continue
                    for branch, prob, current in self._pending_branches(sig, n, np_occ):
                        key = (1, new_out_seen, np_occ, (sig, branch, prob, current))
                        next_states[key] = next_states.get(key, 0.0) + weight * reward_weight
            elif phase == 1:
                if m != 0:
                    continue
                sig, branch, prob, current = payload
                for final_prob, final_current in self._finish_pending(
                    branch, prob, current, sig, n, np_occ
                ):
                    cost = self.p.c_plus if sig == +1 else self.p.c_minus
                    active_weight = final_prob * math.exp(
                        self.p.beta * (self.p.lam * final_current - cost)
                    )
                    key = (2, new_out_seen, np_occ, ())
                    next_states[key] = (
                        next_states.get(key, 0.0) + weight * reward_weight * active_weight
                    )
            else:
                if m != 0 or np_occ != n:
                    continue
                key = (2, new_out_seen, np_occ, ())
                next_states[key] = next_states.get(key, 0.0) + weight * reward_weight
        return next_states

    def entry(self, output_features: list[int], input_features: list[int]) -> float:
        states: dict[tuple[int, int, int, tuple], float] = {(0, 0, -1, ()): 1.0}
        for site, (y_out, y_in) in enumerate(zip(output_features, input_features)):
            states = self.step_states(site, states, y_out, y_in)
            if not states:
                return 0.0

        total = 0.0
        for (phase, out_seen, _prev_np, _payload), weight in states.items():
            if phase == 2 and out_seen == 1:
                total += weight / self.num_actions
        return total

    def build_restricted_matrix_validation(self) -> Array:
        features = [
            self.chain.marker_features(s_idx, a_idx)
            for s_idx in range(self.chain.num_states)
            for a_idx in range(self.chain.num_actions)
        ]
        K = np.zeros((self.chain.dim, self.chain.dim), dtype=np.float64)
        for col, input_features in enumerate(features):
            for row, output_features in enumerate(features):
                K[row, col] = self.entry(output_features, input_features)
        return K

    def accepting_weight(self, state: tuple[int, int, int, tuple]) -> float:
        phase, out_seen, _prev_np, _payload = state
        if phase == 2 and out_seen == 1:
            return 1.0 / self.num_actions
        return 0.0


def _automaton_transition_layers(
    operator: TiltedAutomatonMPO,
) -> tuple[
    list[list[tuple[int, int, int, tuple]]],
    list[list[tuple[int, int, int, int, float]]],
]:
    if operator._transition_layers_cache is not None:
        return operator._transition_layers_cache

    states_by_bond: list[list[tuple[int, int, int, tuple]]] = [[(0, 0, -1, ())]]
    layers: list[list[tuple[int, int, int, int, float]]] = []

    for site in range(operator.p.L):
        left_states = states_by_bond[-1]
        right_index: dict[tuple[int, int, int, tuple], int] = {}
        right_states: list[tuple[int, int, int, tuple]] = []
        transitions: list[tuple[int, int, int, int, float]] = []

        for left_idx, left_state in enumerate(left_states):
            for y_out in range(operator.local_dim):
                for y_in in range(operator.local_dim):
                    next_states = operator.step_states(site, {left_state: 1.0}, y_out, y_in)
                    for right_state, weight in next_states.items():
                        if weight == 0.0:
                            continue
                        if right_state not in right_index:
                            right_index[right_state] = len(right_states)
                            right_states.append(right_state)
                        right_idx = right_index[right_state]
                        transitions.append((left_idx, right_idx, y_out, y_in, float(weight)))

        if not right_states:
            raise ValueError(f"automaton has no reachable states after site {site}")
        states_by_bond.append(right_states)
        layers.append(transitions)

    operator._transition_layers_cache = (states_by_bond, layers)
    return operator._transition_layers_cache


def apply_mpo_to_mps(
    operator: TiltedAutomatonMPO,
    mps: FiniteMPS,
    transpose: bool = False,
    chi_max: int = 128,
    cutoff: float = 1e-12,
) -> tuple[FiniteMPS, dict]:
    """Apply K or K.T to an MPS without building the global dense operator."""
    if mps.L != operator.p.L:
        raise ValueError(f"MPS length {mps.L} does not match operator length {operator.p.L}")
    if mps.local_dim != operator.local_dim:
        raise ValueError(
            f"MPS local_dim {mps.local_dim} does not match operator local_dim {operator.local_dim}"
        )

    states_by_bond, layers = _automaton_transition_layers(operator)
    raw_tensors: list[Array] = []

    for site, (tensor, transitions) in enumerate(zip(mps.tensors, layers)):
        dl, local_dim, dr = tensor.shape
        mpo_left_dim = len(states_by_bond[site])
        mpo_right_dim = len(states_by_bond[site + 1])
        out = np.zeros(
            (mpo_left_dim * dl, local_dim, mpo_right_dim * dr), dtype=np.float64
        )

        for mpo_left, mpo_right, y_out, y_in, weight in transitions:
            if transpose:
                output_symbol = y_in
                contracted_symbol = y_out
            else:
                output_symbol = y_out
                contracted_symbol = y_in
            block = weight * tensor[:, contracted_symbol, :]
            left_slice = slice(mpo_left * dl, (mpo_left + 1) * dl)
            right_slice = slice(mpo_right * dr, (mpo_right + 1) * dr)
            out[left_slice, output_symbol, right_slice] += block

        raw_tensors.append(out)

    final_states = states_by_bond[-1]
    final_tensor = raw_tensors[-1]
    dl, local_dim, right_dim = final_tensor.shape
    original_right_dim = mps.tensors[-1].shape[2]
    if original_right_dim != 1:
        raise ValueError("open-boundary input MPS must have final right bond dimension 1")
    final_tensor = final_tensor.reshape(dl, local_dim, len(final_states), original_right_dim)
    accept = np.array([operator.accepting_weight(state) for state in final_states], dtype=np.float64)
    collapsed = np.tensordot(final_tensor, accept, axes=([2], [0])).reshape(dl, local_dim, 1)
    raw_tensors[-1] = collapsed

    out_mps = FiniteMPS(raw_tensors)
    return out_mps.compress(chi_max=chi_max, cutoff=cutoff)


def _dense_update_residual(new_mps: FiniteMPS, old_mps: FiniteMPS) -> float:
    diff = new_mps.to_dense() - old_mps.to_dense()
    return float(np.linalg.norm(diff))


def power_method_right_left(
    chain,
    chi_max: int,
    cutoff: float,
    steps: int,
    dense_validation: bool = False,
) -> dict:
    """Compressed right/left power iteration for K.T u and K v."""
    operator = TiltedAutomatonMPO(chain)
    u = legal_state_action_mps(chain.p.L, chain.p.N, local_dim=operator.local_dim)
    v = legal_state_action_mps(chain.p.L, chain.p.N, local_dim=operator.local_dim)
    u.normalize()
    v.normalize()
    history = []

    for step in range(int(steps)):
        next_u, info_u = apply_mpo_to_mps(
            operator, u, transpose=True, chi_max=chi_max, cutoff=cutoff
        )
        next_v, info_v = apply_mpo_to_mps(
            operator, v, transpose=False, chi_max=chi_max, cutoff=cutoff
        )
        rho_u = next_u.norm()
        rho_v = next_v.norm()
        if rho_u <= 0 or rho_v <= 0 or not np.isfinite(rho_u + rho_v):
            raise ValueError(f"invalid power-method norms rho_u={rho_u}, rho_v={rho_v}")
        next_u.scale(1.0 / rho_u)
        next_v.scale(1.0 / rho_v)

        right_update_residual = (
            _dense_update_residual(next_u, u) if dense_validation else float("nan")
        )
        left_update_residual = (
            _dense_update_residual(next_v, v) if dense_validation else float("nan")
        )
        u = next_u
        v = next_v
        history.append(
            {
                "step": step + 1,
                "rho_right": float(rho_u),
                "rho_left": float(rho_v),
                "right_update_residual": right_update_residual,
                "left_update_residual": left_update_residual,
                "discarded_weight_right": float(info_u["discarded_weight"]),
                "discarded_weight_left": float(info_v["discarded_weight"]),
                "max_bond_dim_right": int(info_u["max_bond_dim"]),
                "max_bond_dim_left": int(info_v["max_bond_dim"]),
            }
        )

    rho_right = history[-1]["rho_right"]
    rho_left = history[-1]["rho_left"]
    rho_avg = 0.5 * (rho_right + rho_left)
    result = {
        "rho": rho_avg,
        "rho_right": rho_right,
        "rho_left": rho_left,
        "rho_left_right_rel_mismatch": abs(rho_right - rho_left) / max(abs(rho_avg), 1e-14),
        "log_rho_per_site": float(math.log(max(rho_avg, 1e-300)) / chain.p.L),
        "history": history,
    }
    if dense_validation:
        result["u_dense_norm"] = float(np.linalg.norm(u.to_dense()))
        result["v_dense_norm"] = float(np.linalg.norm(v.to_dense()))
    return result

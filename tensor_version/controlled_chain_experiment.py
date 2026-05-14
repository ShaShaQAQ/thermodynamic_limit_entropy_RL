#!/usr/bin/env python3
"""Exact, MPS model-based, and sampled u-theta checks for a controlled hard-core chain.

The model is the one described in articles/tensor_network_rl_supplement.tex:
state n is a fixed-particle occupation string, action a=(bond, direction), and
the reward contains a nearest-neighbor interaction after the move.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class Params:
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


class ControlledHardCoreChain:
    def __init__(self, params: Params):
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
        """Return [(next_state_index, probability, current), ...]."""
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

    def build_exact_K(self) -> np.ndarray:
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

    def apply_K_values(self, u_values: torch.Tensor) -> torch.Tensor:
        """Model-based application of K using local transition rules, not dense matmul.

        u_values has shape [num_states, num_actions] and returns (K u)(state, action).
        This follows the column-stochastic convention in the paper:
            (K u)(z) here means sum_{z'} K(z'|z) u(z') for the right eigenfunction.
        """
        rows = []
        next_action_avg = u_values.mean(dim=1)
        for s_idx in range(self.num_states):
            vals = []
            for a_idx in range(self.num_actions):
                total = u_values.new_tensor(0.0)
                for ns_idx, prob, current in self.transition_outcomes(s_idx, a_idx):
                    r = self.reward(ns_idx, a_idx, current)
                    total = total + prob * math.exp(self.p.beta * r) * next_action_avg[ns_idx]
                vals.append(total)
            rows.append(torch.stack(vals))
        return torch.stack(rows, dim=0)

    def sample_prior_transition(self, rng: random.Random) -> tuple[int, int, int, int, float]:
        s_idx = rng.randrange(self.num_states)
        a_idx = rng.randrange(self.num_actions)
        outcomes = self.transition_outcomes(s_idx, a_idx)
        rnum = rng.random()
        accum = 0.0
        ns_idx, current = outcomes[-1][0], outcomes[-1][2]
        for cand_ns, prob, cand_current in outcomes:
            accum += prob
            if rnum <= accum + 1e-12:
                ns_idx, current = cand_ns, cand_current
                break
        na_idx = rng.randrange(self.num_actions)
        reward = self.reward(ns_idx, a_idx, current)
        return s_idx, a_idx, ns_idx, na_idx, reward

    def marker_features(self, s_idx: int, a_idx: int) -> list[int]:
        """Local feature y_i=(n_i,m_i) encoded as 0..5, with m in {0,+,-}."""
        n = self.states[s_idx]
        bond, sig = self.actions[a_idx]
        ys = []
        for i, occ in enumerate(n):
            marker = 0
            if i == bond:
                marker = 1 if sig == +1 else 2
            ys.append(occ * 3 + marker)
        return ys

    def all_feature_tensor(self) -> torch.Tensor:
        feats = []
        for s_idx in range(self.num_states):
            for a_idx in range(self.num_actions):
                feats.append(self.marker_features(s_idx, a_idx))
        return torch.tensor(feats, dtype=torch.long)


class ExplicitTiltedMPO:
    """Finite-automaton MPO for the tilted state-action kernel.

    The local physical index is y_j=(n_j,m_j), encoded as occ*3+marker with
    marker in {0,+,-}.  The virtual state scans left to right and records:

    - whether the unique input action marker has been found;
    - whether the two-site active gate has a pending second site;
    - whether the unique output next-action marker has been found;
    - the previous output occupation, which supplies the nearest-neighbor
      diagonal reward exp[-beta V n'_j n'_{j+1}].

    For the small validation problem we evaluate MPO entries on the legal
    fixed-particle state-action sector.  This is an explicit MPO contraction
    over local symbols, not a call to transition_outcomes.
    """

    def __init__(self, chain: ControlledHardCoreChain):
        self.chain = chain
        self.p = chain.p
        self.num_actions = chain.num_actions

    @staticmethod
    def decode_feature(y: int) -> tuple[int, int]:
        return divmod(int(y), 3)

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

    def entry(self, output_features: list[int], input_features: list[int]) -> float:
        """Return K_beta(output | input) from the explicit local MPO automaton."""
        L = self.p.L
        # state: (phase, out_seen, prev_np, payload) -> weight
        # phase is 0 before marker, 1 pending second active site, 2 after marker.
        states: dict[tuple[int, int, int, tuple], float] = {(0, 0, -1, ()): 1.0}

        for site, (y_out, y_in) in enumerate(zip(output_features, input_features)):
            n, m = self.decode_feature(y_in)
            np_occ, mp = self.decode_feature(y_out)
            next_states: dict[tuple[int, int, int, tuple], float] = {}

            if site == L - 1 and (m != 0 or mp != 0):
                return 0.0

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
            states = next_states
            if not states:
                return 0.0

        total = 0.0
        for (phase, out_seen, _prev_np, _payload), weight in states.items():
            if phase == 2 and out_seen == 1:
                total += weight / self.num_actions
        return total

    def build_restricted_matrix(self) -> np.ndarray:
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


class MPSFunction(nn.Module):
    """Open-boundary MPS scalar function f(n,a) on action-marker inputs."""

    def __init__(self, L: int, local_dim: int = 6, bond_dim: int = 8):
        super().__init__()
        self.L = L
        self.local_dim = local_dim
        self.bond_dim = bond_dim
        dims = [1] + [bond_dim] * (L - 1) + [1]
        tensors = []
        scale = 0.15
        for i in range(L):
            t = torch.randn(dims[i], local_dim, dims[i + 1], dtype=torch.float64) * scale
            tensors.append(nn.Parameter(t))
        self.tensors = nn.ParameterList(tensors)

    def raw(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        vec = torch.ones(batch, 1, dtype=torch.float64, device=features.device)
        for i, A in enumerate(self.tensors):
            Ai = A[:, features[:, i], :].permute(1, 0, 2)
            vec = torch.bmm(vec.unsqueeze(1), Ai).squeeze(1)
        return vec.squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.raw(features)


def positive_u_from_f(f: torch.Tensor) -> torch.Tensor:
    f = f - f.mean()
    return torch.exp(torch.clamp(f, min=-30.0, max=30.0))


def dominant_exact_u(K: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Dominant eigenfunction u for sum_{z'} K(z'|z) u(z') = rho u(z).

    With the dense matrix convention K[row=z', col=z] this is the dominant
    right eigenvector of K.T, not K.
    """
    vals, vecs = scipy.linalg.eig(K.T)
    idx = int(np.argmax(vals.real))
    rho = float(vals[idx].real)
    u = vecs[:, idx].real
    if u.sum() < 0:
        u *= -1
    u = np.maximum(u, 1e-14)
    u /= np.linalg.norm(u)
    rel_res = np.linalg.norm(K.T @ u - rho * u) / np.linalg.norm(u)
    return rho, u, float(rel_res)


def policy_from_u(chain: ControlledHardCoreChain, u: np.ndarray) -> np.ndarray:
    U = u.reshape(chain.num_states, chain.num_actions)
    U = np.maximum(U, 1e-14)
    return U / U.sum(axis=1, keepdims=True)


def train_model_based(
    chain: ControlledHardCoreChain,
    exact_u: np.ndarray,
    exact_rho: float,
    bond_dim: int,
    steps: int,
    lr: float,
    K_operator: torch.Tensor | None = None,
) -> tuple[MPSFunction, dict]:
    torch.manual_seed(chain.p.seed + 101)
    model = MPSFunction(chain.p.L, bond_dim=bond_dim)
    # Do not initialize from the exact eigenvalue; exact_rho is used only for metrics.
    log_rho = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    opt = optim.Adam(list(model.parameters()) + [log_rho], lr=lr)
    features = chain.all_feature_tensor()

    for _ in range(steps):
        opt.zero_grad()
        f = model(features).reshape(chain.num_states, chain.num_actions)
        u = positive_u_from_f(f)
        if K_operator is None:
            Ku = chain.apply_K_values(u)
        else:
            Ku = (K_operator.T @ u.reshape(-1)).reshape(chain.num_states, chain.num_actions)
        rho = torch.exp(log_rho)
        residual = torch.log(Ku + 1e-30) - torch.log(rho * u + 1e-30)
        loss = (residual**2).mean() + 1e-5 * (f**2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        f = model(features).reshape(chain.num_states, chain.num_actions)
        u = positive_u_from_f(f)
        if K_operator is None:
            Ku = chain.apply_K_values(u)
        else:
            Ku = (K_operator.T @ u.reshape(-1)).reshape(chain.num_states, chain.num_actions)
        rho = torch.exp(log_rho)
        rel_res = torch.linalg.norm((Ku - rho * u).reshape(-1)) / torch.linalg.norm(u.reshape(-1))
        u_np = u.reshape(-1).numpy()
        u_np /= np.linalg.norm(u_np)
        if np.dot(u_np, exact_u) < 0:
            u_np *= -1
        cos = float(np.dot(u_np, exact_u) / (np.linalg.norm(u_np) * np.linalg.norm(exact_u)))
    return model, {
        "rho": float(rho.item()),
        "rho_rel_error": float(abs(rho.item() - exact_rho) / exact_rho),
        "relative_residual": float(rel_res.item()),
        "u_cosine_with_exact": cos,
    }


def train_sampled(
    chain: ControlledHardCoreChain,
    exact_u: np.ndarray,
    exact_rho: float,
    bond_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[MPSFunction, dict]:
    rng = random.Random(chain.p.seed + 202)
    torch.manual_seed(chain.p.seed + 202)
    model = MPSFunction(chain.p.L, bond_dim=bond_dim)
    # Do not initialize from the exact eigenvalue; exact_rho is used only for metrics.
    # A model-free scalar initialization can be estimated from prior samples with u=1.
    rho0_samples = [
        math.exp(chain.p.beta * chain.sample_prior_transition(rng)[4])
        for _ in range(4096)
    ]
    rho0 = max(float(np.mean(rho0_samples)), 1e-12)
    log_rho = nn.Parameter(torch.tensor(math.log(rho0), dtype=torch.float64))
    opt = optim.Adam(list(model.parameters()) + [log_rho], lr=lr)
    all_features = chain.all_feature_tensor()

    for _ in range(steps):
        batch = [chain.sample_prior_transition(rng) for _ in range(batch_size)]
        cur_feats = torch.tensor(
            [chain.marker_features(s, a) for s, a, _, _, _ in batch], dtype=torch.long
        )
        nxt_feats = torch.tensor(
            [chain.marker_features(ns, na) for _, _, ns, na, _ in batch], dtype=torch.long
        )
        rewards = torch.tensor([r for *_, r in batch], dtype=torch.float64)

        opt.zero_grad()
        f_cur = model(cur_feats)
        f_nxt = model(nxt_feats)
        u_cur = torch.exp(torch.clamp(f_cur - f_cur.detach().mean(), -30.0, 30.0))
        u_nxt = torch.exp(torch.clamp(f_nxt - f_cur.detach().mean(), -30.0, 30.0))
        rho = torch.exp(log_rho)
        target = (torch.exp(chain.p.beta * rewards) * u_nxt).detach()
        pred = rho * u_cur
        scale = u_cur.detach().mean() + 1e-12
        td_loss = (((target - pred) / scale) ** 2).mean()
        rho_target = torch.mean(target / (u_cur.detach() + 1e-12)).clamp_min(1e-12)
        rho_loss = (log_rho - torch.log(rho_target.detach())) ** 2
        loss = td_loss + 0.1 * rho_loss + 1e-5 * (f_cur**2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        f = model(all_features).reshape(chain.num_states, chain.num_actions)
        u = positive_u_from_f(f)
        Ku = chain.apply_K_values(u)
        rho = torch.exp(log_rho)
        rel_res = torch.linalg.norm((Ku - rho * u).reshape(-1)) / torch.linalg.norm(u.reshape(-1))
        u_np = u.reshape(-1).numpy()
        u_np /= np.linalg.norm(u_np)
        if np.dot(u_np, exact_u) < 0:
            u_np *= -1
        cos = float(np.dot(u_np, exact_u) / (np.linalg.norm(u_np) * np.linalg.norm(exact_u)))
    return model, {
        "rho": float(rho.item()),
        "rho_rel_error": float(abs(rho.item() - exact_rho) / exact_rho),
        "relative_residual": float(rel_res.item()),
        "u_cosine_with_exact": cos,
    }


def summarize_policy(chain: ControlledHardCoreChain, pi: np.ndarray, max_states: int = 5) -> list[dict]:
    rows = []
    for s_idx in range(min(max_states, chain.num_states)):
        best = int(np.argmax(pi[s_idx]))
        rows.append(
            {
                "state": "".join(map(str, chain.states[s_idx])),
                "best_action": [chain.actions[best][0], chain.actions[best][1]],
                "best_prob": float(pi[s_idx, best]),
                "nn_occupancy": int(sum(chain.states[s_idx][i] * chain.states[s_idx][i + 1] for i in range(chain.p.L - 1))),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--bond-dim", type=int, default=8)
    ap.add_argument("--model-steps", type=int, default=4000)
    ap.add_argument("--sample-steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=Path, default=Path("outputs/controlled_chain_results.json"))
    args = ap.parse_args()

    params = Params(L=args.L, N=args.N)
    np.random.seed(params.seed)
    random.seed(params.seed)
    torch.manual_seed(params.seed)

    chain = ControlledHardCoreChain(params)
    K = chain.build_exact_K()
    explicit_mpo = ExplicitTiltedMPO(chain)
    K_mpo = explicit_mpo.build_restricted_matrix()
    mpo_diff = K_mpo - K
    mpo_validation = {
        "max_abs_entry_error": float(np.max(np.abs(mpo_diff))),
        "frobenius_relative_error": float(np.linalg.norm(mpo_diff) / np.linalg.norm(K)),
        "construction": "finite-automaton MPO entry contraction on the legal fixed-N state-action sector",
    }
    rho, u, exact_res = dominant_exact_u(K)
    pi = policy_from_u(chain, u)
    K_mpo_torch = torch.tensor(K_mpo, dtype=torch.float64)

    _, model_metrics = train_model_based(
        chain,
        u,
        rho,
        bond_dim=args.bond_dim,
        steps=args.model_steps,
        lr=args.lr,
        K_operator=K_mpo_torch,
    )
    _, sample_metrics = train_sampled(
        chain,
        u,
        rho,
        bond_dim=args.bond_dim,
        steps=args.sample_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    result = {
        "params": asdict(params),
        "sizes": {
            "num_states": chain.num_states,
            "num_actions": chain.num_actions,
            "state_action_dim": chain.dim,
            "mps_bond_dim": args.bond_dim,
        },
        "step1_exact": {
            "rho": rho,
            "relative_residual": exact_res,
            "policy_examples": summarize_policy(chain, pi),
        },
        "step2a_explicit_mpo_validation": mpo_validation,
        "step2_mps_model_based": model_metrics,
        "step3_mps_sampled_u_theta": sample_metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

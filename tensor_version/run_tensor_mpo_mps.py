#!/usr/bin/env python3
"""Run dense-free model-based MPO-MPS power iteration for the controlled chain."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tensor_mpo_mps import ControlledHardCoreChainLite, TensorParams, power_method_right_left


def dense_dominant_rho_validation(K: np.ndarray) -> float:
    vals = np.linalg.eigvals(K.T)
    idx = int(np.argmax(vals.real))
    return float(vals[idx].real)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--chi-max", type=int, default=128)
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dense-validation", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/tensor_mpo_mps_validation.json"))
    args = parser.parse_args()

    N = args.N if args.N is not None else args.L // 2
    params = TensorParams(L=args.L, N=N)
    chain = ControlledHardCoreChainLite(params)
    result = power_method_right_left(
        chain,
        chi_max=args.chi_max,
        cutoff=args.cutoff,
        steps=args.steps,
        dense_validation=args.dense_validation,
    )
    payload = {
        "params": asdict(params),
        "settings": {
            "chi_max": args.chi_max,
            "cutoff": args.cutoff,
            "steps": args.steps,
            "dense_validation": args.dense_validation,
        },
        "tensor_mpo_mps": result,
    }
    if args.dense_validation:
        K = chain.build_exact_K()
        rho_exact = dense_dominant_rho_validation(K)
        payload["dense_validation"] = {
            "rho_exact": rho_exact,
            "rho_right_rel_error": abs(result["rho_right"] - rho_exact) / rho_exact,
            "rho_left_rel_error": abs(result["rho_left"] - rho_exact) / rho_exact,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

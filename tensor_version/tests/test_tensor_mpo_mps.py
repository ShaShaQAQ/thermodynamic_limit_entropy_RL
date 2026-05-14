import numpy as np

from controlled_chain_experiment import ControlledHardCoreChain, Params, dominant_exact_pair
from tensor_mpo_mps import FiniteMPS
from tensor_mpo_mps import TiltedAutomatonMPO
from tensor_mpo_mps import ControlledHardCoreChainLite, TensorParams
from tensor_mpo_mps import apply_mpo_to_mps
from tensor_mpo_mps import legal_state_action_mps
from tensor_mpo_mps import power_method_right_left


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


def test_lite_chain_matches_existing_dense_kernel():
    params = Params(L=4, N=2)
    reference = ControlledHardCoreChain(params)
    lite = ControlledHardCoreChainLite(TensorParams(L=4, N=2))
    np.testing.assert_allclose(lite.build_exact_K(), reference.build_exact_K(), atol=1e-14)
    assert lite.marker_features(0, 0) == reference.marker_features(0, 0)


def test_tilted_automaton_mpo_entries_match_explicit_mpo_reference():
    chain = ControlledHardCoreChain(Params(L=4, N=2))
    reference = chain.build_exact_K()
    operator = TiltedAutomatonMPO(chain)
    restricted = operator.build_restricted_matrix_validation()
    np.testing.assert_allclose(restricted, reference, atol=1e-12, rtol=1e-12)


def _features_to_full_index(features):
    idx = 0
    for y in features:
        idx = idx * 6 + y
    return idx


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
        expected[_features_to_full_index(chain.marker_features(s_idx, a_idx))] = restricted_output[z]
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
        expected[_features_to_full_index(chain.marker_features(s_idx, a_idx))] = restricted_output[z]
    np.testing.assert_allclose(dense_applied, expected, atol=1e-10, rtol=1e-10)


def test_power_method_gets_reasonable_small_system_rho():
    chain = ControlledHardCoreChain(Params(L=4, N=2))
    rho_exact, *_ = dominant_exact_pair(chain.build_exact_K())
    result = power_method_right_left(
        chain,
        chi_max=128,
        cutoff=1e-13,
        steps=20,
        dense_validation=True,
    )
    assert abs(result["rho_right"] - rho_exact) / rho_exact < 5e-3
    assert abs(result["rho_left"] - rho_exact) / rho_exact < 5e-3

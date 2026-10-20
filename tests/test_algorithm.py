"""Einheiten des Kerns an Mini-Instanzen mit von Hand gerechneten Werten (DEMO-PLAYBOOK §1b)."""

import warnings

import numpy as np
import pytest

import mbd_algorithm as A
from conftest import make_adj


def _mat(adj):
    return A.adjacency_matrix(adj)


# --- Netze --------------------------------------------------------------------------------------------------------------------------------------


def test_adjacency_matrix_and_induced_matrix(path3):
    assert _mat(path3).tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    adj = make_adj(5, [(0, 1), (1, 2), (3, 4)])
    assert A.largest_component(adj) == [0, 1, 2]
    assert A.induced_matrix(adj, [3, 4]).tolist() == [[0, 1], [1, 0]]
    sub = A.induced_matrix(adj, [2, 1, 0])
    assert sub.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]                                  # Pfad in umgekehrter Nummerierung, symmetrisch
    assert A.largest_component(make_adj(4, [(0, 1), (2, 3)])) == [0, 1]                       # Gleichstand: kleinster Knoten zuerst
    assert A.largest_component(()) == []


def test_sbm_matrix_is_symmetric_deterministic_and_has_the_inside_share():
    a, lab = A.sbm_matrix(240, 6, 6, 0.9, 5)
    assert np.array_equal(a, a.T) and np.trace(a) == 0 and np.array_equal(a, A.sbm_matrix(240, 6, 6, 0.9, 5)[0]) and not np.array_equal(a, A.sbm_matrix(240, 6, 6, 0.9, 6)[0])
    inner = sum(a[i, j] for i in range(240) for j in range(i + 1, 240) if lab[i] == lab[j])
    assert inner / (a.sum() / 2) == pytest.approx(0.9, abs=0.05) and np.bincount(lab).tolist() == [40] * 6
    full, _ = A.sbm_matrix(60, 6, 59, 1.0, 1)
    assert full.sum() > 0 and A.sbm_matrix(6, 6, 3, 0.5, 1)[0].shape == (6, 6)                  # Gruppen der Größe 1: keine Paare innen


# --- DeGroot ------------------------------------------------------------------------------------------------------------------------------------


def test_transition_matrix_hand_values(path3, pair2):
    w = A.transition_matrix(_mat(path3), 0.0)
    assert w.tolist() == [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    assert A.transition_matrix(_mat(path3), 0.5).tolist() == [[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]]
    assert A.transition_matrix(_mat(pair2), 1.0).tolist() == [[1, 0], [0, 1]]
    assert np.allclose(A.transition_matrix(_mat(path3), 0.3).sum(axis=1), 1.0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            A.transition_matrix(_mat(path3), bad)
    with pytest.raises(ValueError):
        A.transition_matrix(_mat(make_adj(3, [(0, 1)])), 0.0)                                  # Betrieb 2 ohne Partner


def test_stationary_weights_and_consensus_hand_values(path3, triangle, star5):
    assert A.stationary_weights(_mat(path3)).tolist() == [0.25, 0.5, 0.25]
    assert A.stationary_weights(_mat(triangle)).tolist() == pytest.approx([1 / 3] * 3)
    assert A.stationary_weights(_mat(star5)).tolist() == [0.5, 0.125, 0.125, 0.125, 0.125]
    assert A.consensus_value(A.stationary_weights(_mat(path3)), [1, 0, 0]) == 0.25
    assert A.consensus_value(A.stationary_weights(_mat(star5)), [1, 0, 0, 0, 0]) == 0.5


def test_degroot_run_reaches_the_consensus_with_self_weight(path3):
    w = A.transition_matrix(_mat(path3), 0.5)
    out = A.degroot_run(w, [1.0, 0.0, 0.0], 1e-9, 1000)
    assert out["converged"] and out["x"].tolist() == pytest.approx([0.25] * 3, abs=1e-8) and out["spans"][0] == 1.0 and len(out["spans"]) == out["steps"] + 1
    assert all(out["spans"][i] >= out["spans"][i + 1] - 1e-15 for i in range(len(out["spans"]) - 1))        # bei s = 1/2 fällt die Spanne monoton
    assert out["steps"] in out["frames"] and out["frames"][out["steps"]].tolist() == out["x"].tolist()


def test_degroot_run_on_a_pair_without_self_weight_swaps_forever(pair2):
    w = A.transition_matrix(_mat(pair2), 0.0)
    out = A.degroot_run(w, [1.0, 0.0], 1e-6, 50)
    assert not out["converged"] and out["steps"] == 50 and out["x"].tolist() == [1.0, 0.0] and out["spans"] == [1.0] * 51
    assert A.is_periodic(w) and A.min_eigenvalue(w) == pytest.approx(-1.0)
    half = A.degroot_run(A.transition_matrix(_mat(pair2), 0.5), [1.0, 0.0], 1e-12, 50)
    assert half["converged"] and half["steps"] == 1 and half["x"].tolist() == [0.5, 0.5] and not A.is_periodic(A.transition_matrix(_mat(pair2), 0.5))


def test_degroot_run_with_equal_opinions_stops_at_once(triangle):
    out = A.degroot_run(A.transition_matrix(_mat(triangle), 0.0), [0.3, 0.3, 0.3], 1e-6, 100)
    assert out["converged"] and out["steps"] == 0 and out["spans"] == [0.0]


def test_degroot_run_keeps_only_the_wanted_frames(path3):
    w = A.transition_matrix(_mat(path3), 0.5)
    out = A.degroot_run(w, [1.0, 0.0, 0.0], 1e-6, 1000, frames=[0, 2, 5])
    assert set(out["frames"]) == {0, 2, 5, out["steps"]} and out["frames"][0].tolist() == [1.0, 0.0, 0.0]
    assert out["frames"][2].tolist() == pytest.approx((np.linalg.matrix_power(w, 2) @ [1.0, 0.0, 0.0]).tolist())


def test_frame_schedule_is_sorted_unique_and_bounded():
    sched = A.frame_schedule(5000, 120)
    assert sched == sorted(set(sched)) and sched[0] == 0 and sched[1] == 1 and sched[-1] == 5000 and len(sched) <= 125
    assert A.frame_schedule(0, 10) == [0]
    assert A.frame_schedule(3, 120) == [0, 1, 2, 3]


def test_eigenvalues_of_the_triangle_and_the_star(triangle, star5):
    wt = A.transition_matrix(_mat(triangle), 0.0)
    ev = np.sort(np.linalg.eigvals(wt).real)
    assert ev.tolist() == pytest.approx([-0.5, -0.5, 1.0]) and A.second_eigenvalue_modulus(wt) == pytest.approx(0.5)
    ws = A.transition_matrix(_mat(star5), 0.0)
    assert A.second_eigenvalue_modulus(ws) == pytest.approx(1.0) and A.is_periodic(ws)                  # bipartit: Eigenwert −1
    assert A.second_eigenvalue_modulus(A.transition_matrix(_mat(star5), 0.5)) == pytest.approx(0.5)     # s = 1/2: Eigenwerte (1 + λ) / 2 → 1, 0.5·(0 + 1) ...
    assert A.second_eigenvalue_modulus(np.eye(1)) == 0.0


# --- Friedkin–Johnsen -----------------------------------------------------------------------------------------------------------------------


def test_fj_equilibrium_hand_values(pair2):
    w = A.transition_matrix(_mat(pair2), 0.0)
    assert A.fj_equilibrium(w, 0.5, [1.0, 0.0]).tolist() == pytest.approx([2 / 3, 1 / 3])
    assert A.fj_equilibrium(w, 0.0, [1.0, 0.0]).tolist() == pytest.approx([1.0, 0.0])               # völlig stur: nichts ändert sich
    assert A.fj_equilibrium(w, np.array([0.5, 0.0]), [1.0, 0.0]).tolist() == pytest.approx([0.5, 0.0])   # Betrieb 2 (a = 0) bleibt bei 0; Betrieb 1: x1 = 0,5 · x2 + 0,5 · 1 = 0,5
    for bad in ({"a_val": 1.5}, {"a_val": -0.1}, {"a_val": 1.0}):
        with pytest.raises(ValueError):
            A.fj_equilibrium(w, bad["a_val"], [1.0, 0.0])


def test_fj_iterate_and_run_reach_the_equilibrium(path3):
    w = A.transition_matrix(_mat(path3), 0.0)
    x0 = [1.0, 0.0, 0.5]
    star = A.fj_equilibrium(w, 0.8, x0)
    assert A.fj_iterate(w, 0.8, x0, 400).tolist() == pytest.approx(star.tolist(), abs=1e-12) and A.fj_iterate(w, 0.8, x0, 0).tolist() == x0
    out = A.fj_run(w, 0.8, x0, 1e-10, 10000, frames=[0, 3])
    assert out["converged"] and out["x"].tolist() == pytest.approx(star.tolist(), abs=1e-9) and out["x_star"].tolist() == star.tolist() and len(out["gaps"]) == out["steps"] + 1
    assert out["gaps"][0] == pytest.approx(np.max(np.abs(np.array(x0) - star))) and set(out["frames"]) == {0, 3, out["steps"]}
    capped = A.fj_run(w, 0.99, x0, 1e-14, 5)
    assert not capped["converged"] and capped["steps"] == 5


def test_fj_equilibrium_lies_between_the_extreme_opinions_and_keeps_diversity(path3):
    w = A.transition_matrix(_mat(path3), 0.0)
    star = A.fj_equilibrium(w, 0.9, [1.0, 0.0, 0.2])
    assert 0.0 <= star.min() and star.max() <= 1.0 and A.span(star) > 0.0


# --- Weisheit der Menge und Einfluss ------------------------------------------------------------------------------------------------------------


def test_effective_sample_size_and_error_variance_by_hand(path3, star5):
    assert A.effective_sample_size([0.25] * 4) == pytest.approx(4.0)
    pi = A.stationary_weights(_mat(path3))
    assert A.effective_sample_size(pi) == pytest.approx(1 / (1 / 16 + 1 / 4 + 1 / 16)) and A.consensus_error_variance(pi, 2.0) == pytest.approx(4 * (3 / 8))
    assert A.effective_sample_size(A.stationary_weights(_mat(star5))) == pytest.approx(1 / (0.25 + 4 / 64))
    assert A.consensus_error_variance([1.0], 1.0) == 1.0


def test_influence_sums_the_weights_of_the_holders(star5):
    pi = A.stationary_weights(_mat(star5))
    assert A.influence(pi, [0]) == 0.5 and A.influence(pi, [1, 2]) == 0.25 and A.influence(pi, []) == 0.0 and A.influence(pi, range(5)) == pytest.approx(1.0)


def test_initial_opinions_by_mode(star5):
    a = _mat(star5)
    rng = np.random.default_rng(3)
    x = A.initial_opinions("Zufällig", a, None, rng)
    assert x.shape == (5,) and 0.0 <= x.min() and x.max() <= 1.0 and not np.array_equal(x, A.initial_opinions("Zufällig", a, None, np.random.default_rng(4)))
    assert A.initial_opinions("Zufällig", a, None, np.random.default_rng(3)).tolist() == x.tolist()
    split = A.initial_opinions("Spaltung nach Gruppen", a, [0, 1, 2, 3, 4, 5][:5], rng)
    assert split.tolist() == [1, 1, 1, 0, 0]                                                       # Gruppen 0 bis 2 von 0 bis 4: kleiner als 2,5
    six = A.initial_opinions("Spaltung nach Gruppen", np.zeros((6, 6)), [0, 1, 2, 3, 4, 5], rng)
    assert six.tolist() == [1, 1, 1, 0, 0, 0]
    assert A.initial_opinions("Die Hubs dafür", a, None, rng, hubs=1).tolist() == [1, 0, 0, 0, 0] and A.initial_opinions("Die Hubs dafür", a, None, rng, hubs=2).tolist() == [1, 1, 0, 0, 0]
    assert A.initial_opinions("Die Hubs dafür", a, None, rng, hubs=99).tolist() == [1] * 5
    single = A.initial_opinions("Ein einzelner Betrieb", a, None, rng)
    assert single.sum() == 1.0 and set(single.tolist()) == {0.0, 1.0}
    with pytest.raises(ValueError):
        A.initial_opinions("unbekannt", a, None, rng)


def test_no_warnings_in_the_core(path3):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = A.transition_matrix(_mat(path3), 0.5)
        A.degroot_run(w, [1.0, 0.0, 0.0], 1e-6, 100)
        A.fj_run(w, 0.9, [1.0, 0.0, 0.0], 1e-6, 100)
        A.second_eigenvalue_modulus(w)


# --- Lücken der Mutationsprüfung -----------------------------------------------------------------------------------------------------------------


def test_fj_equilibrium_error_messages_and_a_fully_receptive_node(pair2):
    w = A.transition_matrix(_mat(pair2), 0.0)
    with pytest.raises(ValueError, match="nicht eindeutig"):
        A.fj_equilibrium(w, 1.0, [1.0, 0.0])                                                       # Empfänglichkeit 1 für alle: kein eindeutiges Gleichgewicht
    for bad in (1.5, -0.25, np.array([1.5, 0.5]), np.array([-0.1, 0.5])):
        with pytest.raises(ValueError, match="muss in"):
            A.fj_equilibrium(w, bad, [1.0, 0.0])
    assert A.fj_equilibrium(w, np.array([1.0, 0.5]), [1.0, 0.0]).tolist() == pytest.approx([0.0, 0.0])        # Betrieb 1 ganz empfänglich (a = 1) ist erlaubt: x1 = x2, x2 = 0,5 x1


def test_fj_iterate_runs_exactly_the_requested_number_of_steps(pair2):
    w = A.transition_matrix(_mat(pair2), 0.0)
    assert A.fj_iterate(w, 0.5, [1.0, 0.0], 1).tolist() == [0.5, 0.5]
    assert A.fj_iterate(w, 0.5, [1.0, 0.0], 2).tolist() == [0.75, 0.25]


def test_spectrum_of_two_nodes_one_node_and_the_triangle(pair2, triangle):
    swap = A.transition_matrix(_mat(pair2), 0.0)
    assert A.second_eigenvalue_modulus(swap) == pytest.approx(1.0) and A.spectrum_summary(swap) == pytest.approx((1.0, -1.0))
    half = A.transition_matrix(_mat(pair2), 0.5)
    assert A.second_eigenvalue_modulus(half) == pytest.approx(0.0, abs=1e-12) and A.spectrum_summary(half) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert A.spectrum_summary(np.eye(1)) == (0.0, 1.0)
    assert A.spectrum_summary(A.transition_matrix(_mat(triangle), 0.0)) == pytest.approx((0.5, -0.5))

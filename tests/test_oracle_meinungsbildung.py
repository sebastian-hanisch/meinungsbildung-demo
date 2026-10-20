"""Orakel-Tests (ORACLE-PLAYBOOK): der Kern wird gegen unabhängige Rechenwege geprüft, nicht gegen sich selbst.

* Konsens: linker Eigenvektor von W zum Eigenwert 1 (numpy.linalg.eig) und die hohe Matrixpotenz W^t.
* Friedkin–Johnsen: lineares System gegen 10⁴ Iterationsschritte und gegen die Neumann-Reihe Σ (ΛW)^k (I − Λ) x(0).
* Zweiter Eigenwert: symmetrische Form D^-1/2 A D^-1/2 (eigvalsh) und die gemessene Abklingrate der Spanne.
* Eigenwert −1: networkx.is_bipartite.
* Weisheit der Menge: Varianz des Konsenses aus einer Simulation gegen σ² Σ π².
"""

import networkx as nx
import numpy as np
import pytest

import mbd_algorithm as A
from conftest import make_adj, random_connected

NETS = 120


def _nets(count=NETS, seed=11):
    rng = np.random.default_rng(seed)
    for i in range(count):
        n = int(rng.integers(4, 40))
        adj = random_connected(rng, n, extra=int(rng.integers(0, 2 * n)))
        yield i, rng, adj, A.adjacency_matrix(adj)


def test_consensus_equals_the_left_eigenvector_of_w_on_many_random_networks():
    for _, rng, _, a in _nets():
        s = float(rng.choice([0.0, 0.1, 0.5, 0.9]))
        w = A.transition_matrix(a, s)
        vals, vecs = np.linalg.eig(w.T)
        k = int(np.argmin(np.abs(vals - 1.0)))
        left = np.real(vecs[:, k])
        left = left / left.sum()
        assert np.allclose(A.stationary_weights(a), left, atol=1e-9)
        x0 = rng.random(len(a))
        assert A.consensus_value(A.stationary_weights(a), x0) == pytest.approx(float(left @ x0), abs=1e-9)


def test_high_powers_of_w_converge_to_the_rank_one_matrix_of_the_stationary_weights():
    for i, rng, _, a in _nets(count=40, seed=12):
        w = A.transition_matrix(a, 0.5)
        limit = np.linalg.matrix_power(w, 4000)
        pi = A.stationary_weights(a)
        # Zeilen von W^t gehen gegen π, sofern 4000 Schritte für die Mischzeit reichen; Prüfung über die Abweichung, die der zweite Eigenwert vorhersagt.
        lam2 = A.second_eigenvalue_modulus(w)
        if lam2**4000 > 1e-9:
            continue
        assert np.allclose(limit, np.tile(pi, (len(a), 1)), atol=1e-8), i


def test_degroot_run_ends_at_the_consensus_when_it_converges():
    seen = 0
    for _, rng, _, a in _nets(count=60, seed=13):
        w = A.transition_matrix(a, 0.5)
        x0 = rng.random(len(a))
        out = A.degroot_run(w, x0, 1e-10, 20000)
        if out["converged"]:
            seen += 1
            assert np.allclose(out["x"], A.consensus_value(A.stationary_weights(a), x0), atol=1e-9)
    assert seen >= 55                                                                       # die meisten kleinen Netze mischen schnell


def test_fj_equilibrium_matches_ten_thousand_iterations_and_the_neumann_series():
    for _, rng, _, a in _nets(count=60, seed=14):
        n = len(a)
        w = A.transition_matrix(a, float(rng.choice([0.0, 0.3])))
        x0 = rng.random(n)
        for a_val in (0.5, 0.9, rng.uniform(0.3, 0.95, size=n)):
            star = A.fj_equilibrium(w, a_val, x0)
            assert np.allclose(star, A.fj_iterate(w, a_val, x0, 10_000), atol=1e-9)
            alpha = np.full(n, a_val) if np.isscalar(a_val) else a_val
            lam = np.diag(alpha)
            term = (np.eye(n) - lam) @ x0
            total = term.copy()
            for _ in range(3000):
                term = lam @ (w @ term)
                total = total + term
            assert np.allclose(star, total, atol=1e-8)


def test_fj_equilibrium_is_a_fixed_point_and_stays_inside_the_initial_range():
    for _, rng, _, a in _nets(count=60, seed=15):
        n = len(a)
        w = A.transition_matrix(a, 0.0)
        x0 = rng.random(n)
        alpha = rng.uniform(0.0, 0.99, size=n)
        star = A.fj_equilibrium(w, alpha, x0)
        assert np.allclose(star, alpha * (w @ star) + (1 - alpha) * x0, atol=1e-10)
        assert x0.min() - 1e-12 <= star.min() and star.max() <= x0.max() + 1e-12


def test_second_eigenvalue_matches_the_symmetric_form():
    for _, rng, _, a in _nets():
        s = float(rng.choice([0.0, 0.2, 0.5]))
        w = A.transition_matrix(a, s)
        d = a.sum(axis=1)
        sym = a / np.sqrt(np.outer(d, d))
        mu = np.sort(np.linalg.eigvalsh(sym))                                               # Eigenwerte der symmetrischen Form, aufsteigend; μ_max = 1
        lam = s + (1 - s) * mu                                                              # W = s I + (1 − s) D⁻¹A hat dieselben Eigenwerte wie s I + (1 − s) sym
        expected = max(abs(lam[-2]), abs(lam[0]))
        assert A.second_eigenvalue_modulus(w) == pytest.approx(expected, abs=1e-9)
        assert A.min_eigenvalue(w) == pytest.approx(lam[0], abs=1e-9)


def test_measured_decay_rate_of_the_span_matches_the_second_eigenvalue():
    checked = 0
    for _, rng, _, a in _nets(count=30, seed=16):
        w = A.transition_matrix(a, 0.5)
        lam2 = A.second_eigenvalue_modulus(w)
        if lam2 > 0.97:
            continue
        out = A.degroot_run(w, rng.random(len(a)), 1e-13, 20000)
        sp = out["spans"]
        if len(sp) < 120 or not out["converged"]:
            continue
        rate = (sp[-1] / sp[-41]) ** (1 / 40)
        assert rate == pytest.approx(lam2, rel=0.03)
        checked += 1
    assert checked >= 15


def test_eigenvalue_minus_one_appears_exactly_for_bipartite_networks_without_self_weight():
    n_bip = 0
    for i, rng, adj, a in _nets(count=NETS, seed=17):
        w0 = A.transition_matrix(a, 0.0)
        bip = nx.is_bipartite(nx.from_numpy_array(a))
        assert A.is_periodic(w0) == bip, i
        assert not A.is_periodic(A.transition_matrix(a, 0.1))                               # schon ein kleines Selbstgewicht hebt es auf
        n_bip += bip
    assert 5 <= n_bip < NETS                                                                # beide Fälle kommen vor
    for g in (nx.cycle_graph(6), nx.cycle_graph(7), nx.grid_2d_graph(4, 5)):
        a = nx.to_numpy_array(g)
        assert A.is_periodic(A.transition_matrix(a, 0.0)) == nx.is_bipartite(g)


def test_wisdom_of_the_crowd_variance_matches_a_simulation():
    rng = np.random.default_rng(18)
    for n, extra in ((30, 20), (40, 5)):
        adj = random_connected(rng, n, extra=extra)
        a = A.adjacency_matrix(adj)
        pi = A.stationary_weights(a)
        sigma = 1.7
        truth = 0.4
        draws = truth + sigma * rng.standard_normal((40_000, n))
        consensus = draws @ pi
        assert np.mean(consensus) == pytest.approx(truth, abs=0.05)
        assert np.var(consensus) == pytest.approx(A.consensus_error_variance(pi, sigma), rel=0.04)
        assert 1.0 <= A.effective_sample_size(pi) <= n + 1e-9


def test_invariants_rows_sum_to_one_entries_nonnegative_and_opinions_stay_in_range():
    for _, rng, _, a in _nets(count=60, seed=19):
        s = float(rng.choice([0.0, 0.5, 0.9]))
        w = A.transition_matrix(a, s)
        assert np.allclose(w.sum(axis=1), 1.0) and (w >= 0).all()
        pi = A.stationary_weights(a)
        assert pi.sum() == pytest.approx(1.0) and np.allclose(pi @ w, pi, atol=1e-12)         # π ist stationär
        x0 = rng.random(len(a))
        out = A.degroot_run(w, x0, 1e-8, 300)
        assert out["x"].min() >= x0.min() - 1e-12 and out["x"].max() <= x0.max() + 1e-12
        assert all(out["spans"][i + 1] <= out["spans"][i] + 1e-12 for i in range(len(out["spans"]) - 1)) or s == 0.0     # ohne Selbstgewicht muss die Spanne nicht monoton fallen
        assert np.dot(pi, out["x"]) == pytest.approx(np.dot(pi, x0), abs=1e-10)               # π · x(t) bleibt erhalten


def test_a_hub_carries_its_degree_share_of_the_consensus():
    adj = make_adj(6, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2)])
    a = A.adjacency_matrix(adj)
    pi = A.stationary_weights(a)
    assert A.influence(pi, [0]) == pytest.approx(5 / 12)
    out = A.degroot_run(A.transition_matrix(a, 0.5), [1, 0, 0, 0, 0, 0], 1e-12, 5000)
    assert out["x"][0] == pytest.approx(5 / 12, abs=1e-10)

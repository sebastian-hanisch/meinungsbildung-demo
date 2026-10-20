"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste und größte Regler, alle Netztypen und Modi, Warnungen als Fehler, kleinste Netze in allen Experimenten, Abbildungen."""

import dataclasses
import warnings

import numpy as np
import pytest

import mbd_algorithm as A
import mbd_constants as C
import mbd_evaluation as ev
import mbd_visualization as V


@pytest.mark.parametrize("typ", C.ALL_TYPES)
def test_every_network_type_at_the_slider_corners_runs_without_warnings(typ):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for n, c, s, a, mode in ((C.N_MIN, C.C_MIN, C.S_MIN, C.A_MIN, C.M_RANDOM), (C.N_MIN, C.C_MAX, C.S_MAX, C.A_MAX, C.M_SPLIT), (C.N_MAX, C.C_MAX, C.S_MIN, C.A_MAX, C.M_HUBS), (C.N_MAX, C.C_MIN, C.S_MAX, 99, C.M_SINGLE)):
            an = ev.analyse(ev.Settings(typ, n, c, 3, s, a, mode))
            assert 2 <= an.m <= an.inst.n and ev.verdict_main(an)[0] in ("success", "info", "warning")
            assert all(len(x) == an.m for x in an.frames.values()) and an.pi.sum() == pytest.approx(1.0)
            assert np.all(an.final >= -1e-12) and np.all(an.final <= 1 + 1e-12)                       # alle Meinungen bleiben in [0, 1]
            assert an.consensus == pytest.approx(float(np.dot(an.pi, an.x0))) and 1.0 <= an.n_eff <= an.m + 1e-9


@pytest.mark.parametrize("typ", C.ALL_TYPES)
def test_many_seeds_in_the_smallest_networks_never_break_the_analysis(typ):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for seed in range(8):
            for mode in C.MODES:
                an = ev.analyse(ev.Settings(typ, C.N_MIN, C.C_MIN, seed, 0, 100, mode))
                assert an.m >= 2 and np.isfinite(an.lam2) and np.isfinite(an.run["spans"]).all()


def test_the_step_limit_is_reported_as_not_converged(monkeypatch):
    monkeypatch.setattr(C, "MAX_STEPS", 30)
    an = ev.analyse(ev.Settings(C.T_RING, 100, 4, 3, 50, 100, C.M_RANDOM))
    assert not an.converged and an.steps == 30 and not an.periodic and ev.verdict_main(an)[0] == "warning" and "Obergrenze" in ev.verdict_main(an)[1]
    fj = ev.analyse(ev.Settings(C.T_RING, 100, 4, 3, 50, 99, C.M_RANDOM))
    assert not fj.converged and fj.steps == 30 and ev.verdict_main(fj)[0] == "warning" and np.allclose(fj.final, fj.run["x_star"])


def test_self_weight_and_receptiveness_at_the_extremes():
    full = ev.analyse(ev.Settings(C.T_RANDOM, 100, 6, 3, C.S_MAX, 100, C.M_RANDOM))
    assert full.converged and full.steps > ev.analyse(ev.Settings(C.T_RANDOM, 100, 6, 3, 0, 100, C.M_RANDOM)).steps        # mehr Selbstgewicht, langsamer
    stubborn = ev.analyse(ev.Settings(C.T_RANDOM, 100, 6, 3, 0, C.A_MIN, C.M_RANDOM))
    assert stubborn.fj and stubborn.converged and 0.4 < stubborn.spread_ratio < 0.7                  # Empfänglichkeit 50 %: gut die Hälfte der Vielfalt bleibt


def test_split_mode_on_a_network_without_communities_uses_six_index_blocks():
    an = ev.analyse(ev.Settings(C.T_RANDOM, 120, 6, 3, 0, 100, C.M_SPLIT))
    assert set(np.unique(an.x0).tolist()) == {0.0, 1.0} and abs(float(an.x0.sum()) - an.m / 2) <= an.m / C.SPLIT_GROUPS


def test_hub_mode_puts_the_ones_on_the_nodes_with_most_partners():
    an = ev.analyse(ev.Settings(C.T_BA, 120, 6, 3, 0, 100, C.M_HUBS))
    deg = np.array([len(an.inst.adj[v]) for v in an.nodes])
    assert an.x0.sum() == C.HUBS_K and deg[an.x0 == 1.0].min() >= deg[an.x0 == 0.0].max()


def test_single_node_mode_puts_one_on_exactly_one_node():
    an = ev.analyse(ev.Settings(C.T_RANDOM, 100, 6, 3, 0, 100, C.M_SINGLE))
    assert an.x0.sum() == 1.0 and an.consensus == pytest.approx(an.pi[int(np.argmax(an.x0))])


def test_experiments_run_on_the_smallest_networks_without_warnings():
    seeds = C.LIVE_SEEDS[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e1 = ev.influence_rows(C.N_MIN, C.C_MIN, seeds=seeds)
        e2 = ev.speed_rows(C.N_MIN, C.C_MIN, seeds=seeds)
        e2b = ev.inside_rows(insides=(0.5, 0.99), seeds=seeds)
        e2c = ev.bipartite_rows(C.N_MIN)
        e3 = ev.wisdom_rows(C.N_MIN, C.C_MIN, seeds=seeds, runs=50)
        e4 = ev.stubborn_rows(C.N_MIN, C.C_MIN, seeds=seeds)
        e5 = ev.split_rows(insides=(0.5, 0.99), alphas=(0.8, 0.95), seeds=seeds)
    assert len(e1) == len(C.ALL_TYPES) and len(e2) == 2 * len(C.ALL_TYPES) and len(e2b) == 2 and len(e2c) == 3 and len(e3) == len(C.ALL_TYPES) and len(e4) == len(C.STUBBORN_AS) * len(C.ALL_TYPES) and len(e5) == 4
    for verdict in (ev.verdict_influence(e1), ev.verdict_speed(e2), ev.verdict_inside(e2b), ev.verdict_bipartite(e2c), ev.verdict_wisdom(e3), ev.verdict_stubborn(e4), ev.verdict_split(e5)):
        assert verdict[0] in ("success", "info", "warning") and verdict[1]


def test_a_regular_network_counts_every_head():
    cycle = np.zeros((8, 8))
    for i in range(8):
        cycle[i, (i + 1) % 8] = cycle[(i + 1) % 8, i] = 1.0
    pi = A.stationary_weights(cycle)
    assert A.effective_sample_size(pi) == pytest.approx(8.0) and A.consensus_error_variance(pi, 1.0) == pytest.approx(1 / 8)
    assert A.influence(pi, [0, 1, 2]) == pytest.approx(3 / 8)                                       # bei regulärem Netz zählt jeder gleich


def test_figures_are_built_for_the_analysis_and_the_experiment_rows():
    seeds = C.LIVE_SEEDS[:2]
    for typ, a in ((C.T_RING, 100), (C.T_GRID, 90)):
        an = ev.analyse(ev.Settings(typ, 80, 4, 3, 0, a, C.M_RANDOM))
        for step in (0, max(an.frames)):
            for fig in (V.build_network(an, step), V.build_spans(an, step)):
                assert fig.data and all(ax.fixedrange for ax in (fig.layout.xaxis, fig.layout.yaxis))
    e2 = ev.speed_rows(C.N_MIN, 4, types=(C.T_GRID, C.T_RANDOM), seeds=seeds)
    figs = [V.build_influence(ev.influence_rows(C.N_MIN, 4, seeds=seeds)), V.build_speed(e2, 0.0), V.build_speed(e2, 0.5), V.build_inside(ev.inside_rows(insides=(0.5, 0.9), seeds=seeds)),
            V.build_wisdom(ev.wisdom_rows(C.N_MIN, 4, seeds=seeds, runs=20)), V.build_stubborn(ev.stubborn_rows(C.N_MIN, 4, seeds=seeds)), V.build_split(ev.split_rows(insides=(0.5, 0.9), alphas=(0.8, 0.95), seeds=seeds))]
    for fig in figs:
        assert fig.data and all(ax.fixedrange for ax in (fig.layout.xaxis, fig.layout.yaxis))


def test_the_speed_chart_marks_periodic_cells_by_annotation_not_by_an_empty_bar():
    rows = ev.speed_rows(C.N_MIN, 4, types=(C.T_GRID, C.T_RANDOM), seeds=C.LIVE_SEEDS[:2])
    fig = V.build_speed(rows, 0.0)
    assert [a.text for a in fig.layout.annotations] == ["pendelt"] and list(fig.data[0].x) == [C.T_RANDOM] and all(y is not None for y in fig.data[0].y)
    assert V.build_speed(rows, 0.5).layout.annotations == ()


def test_the_network_figure_colors_follow_the_chosen_step_on_a_fixed_scale():
    an = ev.analyse(ev.Settings(C.T_RANDOM, 80, 6, 3, 0, 100, C.M_RANDOM))
    first, last = V.build_network(an, 0), V.build_network(an, an.steps)
    nodes0, nodes1 = first.data[1], last.data[1]
    assert list(nodes0.marker.color) == pytest.approx(an.x0.tolist()) and list(nodes1.marker.color) == pytest.approx(an.final.tolist())
    assert nodes0.marker.cmin == nodes1.marker.cmin == 0.0 and nodes0.marker.cmax == nodes1.marker.cmax == 1.0 and len(nodes0.x) == an.m
    assert max(nodes0.marker.size) > min(nodes0.marker.size)                                         # Größe = Gewicht im Konsens


def test_the_span_chart_floors_zero_spans_for_the_log_axis():
    an = ev.analyse(ev.Settings(C.T_RANDOM, 60, 6, 3, 0, 100, C.M_RANDOM))
    flat = dataclasses.replace(an, run=dict(an.run, spans=[1.0, 0.0, 0.0]))
    fig = V.build_spans(flat, 0)
    ys = [y for tr in fig.data for y in tr.y]
    assert ys == [1.0, V.FLOOR, V.FLOOR] and fig.layout.yaxis.type == "log"
    xs, ys = V._thin(list(range(1, 5000)))
    assert xs[0] == 0 and xs[-1] == 4998 and len(xs) <= V.MAX_POINTS + 1

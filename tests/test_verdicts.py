"""Verdikte und Messhilfen an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import dataclasses
import math

import numpy as np
import pytest

import mbd_algorithm as A
import mbd_constants as C
import mbd_evaluation as ev

NAN = float("nan")


def _an(m=40, fj=False, periodic=False, converged=True, steps=12, final=0.375, spans_last=5e-7, a=100, x0=None, **over):
    """Analyse mit handgesetzten Kennzahlen auf einem echten kleinen Lauf: Konsens 0,375, einfacher Mittelwert 0,5, |λ₂| 0,75, n_eff 20, Gewicht der zehn Hubs 0,5."""
    real = ev.analyse(ev.Settings(C.T_RANDOM, C.N_MIN, 4, 3))
    x0 = np.array([0.0, 1.0] * (m // 2)) if x0 is None else x0
    finals = np.full(m, final) if np.isscalar(final) else np.asarray(final)
    run = {"steps": steps, "converged": converged, "spans": [1.0, spans_last], "x": finals, "x_star": finals, "frames": {}}
    fields = dict(nodes=tuple(range(m)), x0=x0, run=run, fj=fj, periodic=periodic, lam2=0.75, consensus=0.375, mean0=0.5, n_eff=20.0, hub_share=0.5,
                  settings=dataclasses.replace(real.settings, a=a))
    fields.update(over)
    return dataclasses.replace(real, **fields)


WHO_UNEVEN = "Das Gewicht im Konsens ist ungleich verteilt: die 10 Betriebe mit den meisten Partnern haben zusammen 50,0 % (bei gleichem Gewicht wären es 25,0 %); das Netz zählt wie 20 unabhängige Köpfe von 40."


def test_de_pct_helpers():
    assert ev.de(1.26, 1) == "1,3" and ev.de(0.9487, 3) == "0,949" and ev.de(NAN, 2) == "–" and ev.de(None) == "–" and ev.pct(0.1234, 1) == "12,3" and ev.pct(0.5) == "50"
    assert ev.de(-1.0, 3) == "−1,000"                                                              # typografisches Minus
    assert ev.tol_text(1e-6) == "10⁻⁶" and ev.tol_text(1e-3) == "10⁻³" and ev.tol_text(1e-12) == "10⁻¹²"


def test_main_verdict_degroot_consensus_with_uneven_weights_is_success():
    kind, text = ev.verdict_main(_an())
    assert kind == "success"
    assert text == ("Nach 12 Schritten sind sich alle Betriebe einig (Spanne unter 10⁻⁶): auf 0,375. "
                    "Das ist der mit dem Grad gewichtete Mittelwert der Anfangsmeinungen (0,375), nicht der einfache (0,500). " + WHO_UNEVEN + " Die Geschwindigkeit hängt an |λ₂| = 0,750.")


def test_main_verdict_threshold_for_uneven_weights_is_one_and_a_half_times_the_fair_share():
    assert ev.verdict_main(_an(hub_share=0.375))[0] == "success"                                    # 1,5 · 10 / 40 = 0,375 genau
    kind, text = ev.verdict_main(_an(hub_share=0.37))
    assert kind == "info" and "Das Gewicht im Konsens ist hier fast gleich verteilt: die 10 Betriebe mit den meisten Partnern haben zusammen 37,0 % (bei gleichem Gewicht wären es 25,0 %)" in text


def test_main_verdict_small_networks_use_all_nodes_as_hubs():
    kind, text = ev.verdict_main(_an(m=6, hub_share=1.0, n_eff=4.0))
    assert "die 6 Betriebe mit den meisten Partnern haben zusammen 100,0 % (bei gleichem Gewicht wären es 100,0 %)" in text and kind == "info"   # 1,0 < 1,5 · 6 / 6


def test_main_verdict_periodic_network_pendelt_and_is_a_warning():
    kind, text = ev.verdict_main(_an(periodic=True, converged=False, steps=2000, spans_last=0.5))
    assert kind == "warning"
    assert text == ("Die Meinungen pendeln: ohne Gewicht auf die eigene Meinung hat die Mittelung auf diesem Netz den Eigenwert −1 (das Netz ist bipartit), die Spanne bleibt nach 2000 Schritten bei 0,500. "
                    "Schon ein kleines Selbstgewicht hebt den Eigenwert −1 auf (Experiment 2 zeigt das am Gitter); dann liefe das Netz gegen den Konsens 0,375.")


def test_main_verdict_slow_network_at_the_step_limit_is_a_warning():
    kind, text = ev.verdict_main(_an(converged=False, steps=30000, spans_last=0.0625))
    assert kind == "warning" and text == "Nach 30000 Schritten (Obergrenze) liegt die Spanne der Meinungen noch bei 0,0625; der Konsens wäre 0,375. |λ₂| = 0,750: so langsam mischt dieses Netz."


def test_main_verdict_friedkin_johnsen_names_range_and_remaining_spread():
    final = np.array([0.25, 0.5] * 20)
    kind, text = ev.verdict_main(_an(fj=True, a=95, steps=30, final=final))
    assert kind == "info"
    assert text == ("Nach 30 Schritten hat sich das Netz auf das Gleichgewicht eingependelt (Empfänglichkeit 95 %): die Meinungen bleiben zwischen 0,25 und 0,50, 25,0 % der Anfangsstreuung bleiben erhalten. "
                    "Reine Mittelung käme auf den Konsens 0,375 (einfacher Mittelwert der Anfangsmeinungen 0,500). " + WHO_UNEVEN)
    kind, text = ev.verdict_main(_an(fj=True, a=95, steps=30000, converged=False, final=final))
    assert kind == "warning" and text.startswith("Nach 30000 Schritten (Obergrenze) ist das Gleichgewicht noch nicht auf 10⁻⁶ genau erreicht; gezeigt ist das exakt berechnete Gleichgewicht (Empfänglichkeit 95 %)")


def test_main_verdict_equal_opinions_have_nothing_to_adjust():
    assert ev.verdict_main(_an(x0=np.full(40, 0.3))) == ("info", "Alle 40 Betriebe beginnen mit derselben Meinung (0,30): es gibt nichts abzugleichen.")


# --- E1 bis E5 ------------------------------------------------------------------------------------------------------------------------------------


def test_influence_verdict_names_strong_and_weak_types_and_the_threshold():
    rows = [{"typ": "A", "hubs": 0.375, "random": 0.25, "periphery": 0.0625, "pi_max": 0.1}, {"typ": "B", "hubs": 0.36, "random": 0.25, "periphery": 0.2, "pi_max": 0.1}]
    kind, text = ev.verdict_influence(rows)
    assert kind == "info"
    assert text == ("Haben nur die 10 Betriebe mit den meisten Partnern die Meinung 1, landet der Konsens in „A“ bei 37,5 % (bei 10 zufälligen Betrieben im Mittel 25,0 %, bei den 10 mit den wenigsten Partnern 6,2 %). "
                    "Mindestens das 1,5-Fache der zufälligen Wahl erreichen die Hubs in: „A“. Darunter bleiben sie in: „B“.")        # 6,25 → 6,2: Banker-Rundung der Formatierung
    only_weak = ev.verdict_influence([rows[1]])[1]
    assert "In keinem Netztyp erreichen die Hubs das 1,5-Fache der zufälligen Wahl." in only_weak and "Darunter bleiben sie in: „B“." in only_weak
    only_strong = ev.verdict_influence([rows[0]])[1]
    assert only_strong.endswith("Darunter bleibt kein Netztyp.") and ev.verdict_influence([])[0] == "warning"


def test_speed_verdict_names_fastest_and_the_two_slowest_types():
    rows = [{"typ": "X", "s": 0.5, "lam2": 0.9, "steps": 100.0}, {"typ": "Y", "s": 0.5, "lam2": 0.97, "steps": 400.0}, {"typ": "W", "s": 0.5, "lam2": 0.94, "steps": 200.0},
            {"typ": "X", "s": 0.0, "lam2": 0.8, "steps": 1.0}, {"typ": "V", "s": 0.5, "lam2": 1.0, "steps": NAN}]
    kind, text = ev.verdict_speed(rows)
    assert kind == "info"
    assert text == ("Bei Selbstgewicht 50 % braucht die Mittelung in „X“ 100 Schritte (|λ₂| = 0,900), in „Y“ 400 (|λ₂| = 0,970): 4-mal so lang. "
                    "Je näher |λ₂| an 1 liegt, desto länger; am langsamsten sind „Y“ und „W“.")
    assert ev.verdict_speed([r for r in rows if math.isnan(r["steps"])])[0] == "warning" and ev.verdict_speed([]) == ("warning", "Keine Messung.")
    assert ev.verdict_speed(rows[:1])[1].endswith("Je näher |λ₂| an 1 liegt, desto länger.")       # ein einziger Netztyp: kein zweitlangsamster


def test_inside_verdict_and_bipartite_verdict():
    rows = [{"inside": 0.5, "steps": 40.0, "lam2": 0.878}, {"inside": 0.99, "steps": 1653.0, "lam2": 0.997}]
    assert ev.verdict_inside(rows) == ("info", "Liegen 50 % der Kanten innerhalb der Verbünde, ist die Spanne nach 40 Schritten unter 10⁻³ (|λ₂| = 0,878); bei 99 % sind es 1653 Schritte (|λ₂| = 0,997). Wenige Querkanten sind der Engpass.")
    assert ev.verdict_inside(rows[:1])[0] == "warning"
    bip = [{"c": 4, "s": 0.0, "lam_min": -1.0, "converged": False, "steps": 2000, "span": 0.25}, {"c": 4, "s": 0.1, "lam_min": -0.8, "converged": True, "steps": 670, "span": 1e-7},
           {"c": 8, "s": 0.0, "lam_min": -0.5, "converged": True, "steps": 401, "span": 1e-7}]
    assert ev.verdict_bipartite(bip) == ("info", "Gitter mit 4 Partnern ohne Selbstgewicht: kleinster Eigenwert −1,000, die Meinungen pendeln (Spanne nach 2000 Schritten 0,250). Mit 10 % Selbstgewicht: kleinster Eigenwert −0,800, "
                                               "Konvergenz nach 670 Schritten. Mit 8 Partnern (Diagonalen) ohne Selbstgewicht: kleinster Eigenwert −0,500, Konvergenz nach 401 Schritten.")
    flipped = [dict(bip[0], converged=True, steps=9), dict(bip[1], converged=False), dict(bip[2], converged=False)]
    text = ev.verdict_bipartite(flipped)[1]
    assert "die Meinungen konvergieren (Spanne nach 9 Schritten" in text and text.count("keine Konvergenz") == 2
    assert ev.verdict_bipartite(bip[:2])[0] == "warning"


def test_wisdom_verdict_names_the_worst_and_the_best_type():
    rows = [{"typ": "A", "m": 250.0, "n_eff": 100.0, "factor": 2.5, "sim_factor": 2.75}, {"typ": "B", "m": 250.0, "n_eff": 248.0, "factor": 1.0625, "sim_factor": 1.0}]
    assert ev.verdict_wisdom(rows) == ("info", "Die Fehlervarianz des Konsenses ist das 2,50-Fache des einfachen Mittelwerts in „A“ (das Netz zählt wie 100 von 250 unabhängigen Köpfen; Simulation: 2,75-fach) "
                                              "und nur das 1,06-Fache in „B“ (248 von 250). Je ungleicher die Gewichte π, desto weniger Köpfe zählen.")
    assert ev.verdict_wisdom([])[0] == "warning"


def test_stubborn_and_split_verdicts_use_95_percent_when_present_else_the_largest_value():
    rows = [{"typ": "A", "a": 0.9, "ratio": 0.5}, {"typ": "A", "a": 0.95, "ratio": 0.0625}, {"typ": "B", "a": 0.95, "ratio": 0.125}, {"typ": "B", "a": 0.9, "ratio": 0.25}]
    assert ev.verdict_stubborn(rows) == ("info", "Bei Empfänglichkeit 95 % bleiben in „A“ 6,2 % der Anfangsstreuung erhalten, in „B“ 12,5 %. Je kleiner die Empfänglichkeit, desto mehr Vielfalt bleibt.")
    only = ev.verdict_stubborn([r for r in rows if r["a"] == 0.9])[1]
    assert only.startswith("Bei Empfänglichkeit 90 % bleiben in „B“ 25,0 % der Anfangsstreuung erhalten, in „A“ 50,0 %.")
    srows = [{"inside": 0.5, "a": 0.95, "ratio": 0.125}, {"inside": 0.99, "a": 0.95, "ratio": 0.75}, {"inside": 0.5, "a": 0.8, "ratio": 0.0625}]
    assert ev.verdict_split(srows) == ("info", "Bei Empfänglichkeit 95 % bleiben von der Anfangsspaltung 12,5 % erhalten, wenn 50 % der Kanten innerhalb der Verbünde liegen, und 75,0 %, wenn es 99 % sind. "
                                              "Wenige Querkanten und etwas Sturheit halten die Spaltung entlang der Verbünde.")
    assert ev.verdict_stubborn([])[0] == "warning" and ev.verdict_split([])[0] == "warning"


# --- Messhilfen und Reihen --------------------------------------------------------------------------------------------------------------------------


def test_run_model_picks_degroot_or_friedkin_johnsen_and_the_step_limit():
    w = A.transition_matrix(np.array([[0, 1.0], [1.0, 0]]), 0.5)
    deg = ev.run_model(w, [1.0, 0.0], 1.0)
    assert "x_star" not in deg and deg["converged"] and deg["x"].tolist() == [0.5, 0.5]
    fj = ev.run_model(w, [1.0, 0.0], 0.5)
    assert "x_star" in fj and fj["converged"] and fj["x_star"].tolist() == pytest.approx([0.75, 0.25])      # x1 = 0,5 m + 0,5, x2 = 0,5 m mit m = (x1 + x2) / 2
    swap = A.transition_matrix(np.array([[0, 1.0], [1.0, 0]]), 0.0)
    assert ev.run_model(swap, [1.0, 0.0], 1.0)["steps"] == C.PERIODIC_STEPS                         # Eigenwert −1: die Obergrenze für pendelnde Netze
    assert ev.run_model(swap, [1.0, 0.0], 1.0, periodic=False, max_steps=7)["steps"] == 7


def test_hub_and_periphery_order_break_ties_by_the_smaller_index():
    a = A.adjacency_matrix(((1, 2, 3), (0,), (0, 3), (0, 2)))
    assert ev.hub_order(a).tolist() == [0, 2, 3, 1] and ev.periphery_order(a).tolist() == [1, 2, 3, 0]


def test_influence_rows_hubs_beat_random_beat_periphery_and_random_is_k_over_m():
    rows = ev.influence_rows(80, 5, seeds=C.LIVE_SEEDS[:2])
    assert [r["typ"] for r in rows] == list(C.ALL_TYPES)
    for r in rows:
        assert r["hubs"] >= r["random"] >= r["periphery"] and 0.0 < r["pi_max"] <= 1.0
        assert 0.0 < r["random"] < 1.0
    one = ev.influence_rows(80, 5, types=(C.T_RANDOM,), seeds=C.LIVE_SEEDS[:1])[0]
    _, a, _ = ev.component_of_instance(ev.make_instance(C.T_RANDOM, 80, 5, C.LIVE_SEEDS[0], layout=False))
    assert one["random"] == pytest.approx(C.HUBS_K / len(a))


def test_speed_rows_shape_and_the_periodic_cell():
    rows = ev.speed_rows(C.N_MIN + 20, 4, types=(C.T_GRID, C.T_BA), seeds=C.LIVE_SEEDS[:2])
    assert [(r["typ"], r["s"]) for r in rows] == [(C.T_GRID, 0.0), (C.T_GRID, 0.5), (C.T_BA, 0.0), (C.T_BA, 0.5)]
    assert rows[0]["periodic"] == 2 and math.isnan(rows[0]["steps"]) and rows[0]["lam2"] == pytest.approx(1.0)       # Gitter mit vier Partnern ohne Selbstgewicht: bipartit
    assert rows[1]["periodic"] == 0 and rows[1]["steps"] > 0 and rows[1]["lam2"] < 1.0
    assert ev.speed_cell(rows, C.T_BA, 0.5) is rows[3]


def test_inside_rows_grow_with_the_inside_share():
    rows = ev.inside_rows(insides=(0.5, 0.9), seeds=C.LIVE_SEEDS[:2])
    assert rows[0]["steps"] < rows[1]["steps"] and rows[0]["lam2"] < rows[1]["lam2"] and all(r["s"] == 0.5 and r["nets"] == 2 for r in rows)


def test_bipartite_rows_three_cases():
    rows = ev.bipartite_rows(100)
    assert [(r["c"], r["s"]) for r in rows] == [(4, 0.0), (4, 0.1), (8, 0.0)]
    assert rows[0]["bipartite"] and not rows[0]["converged"] and rows[0]["lam_min"] == pytest.approx(-1.0) and rows[0]["span"] > 1e-3
    assert not rows[1]["bipartite"] and rows[1]["converged"] and rows[1]["lam_min"] == pytest.approx(-0.8, abs=1e-6)
    assert not rows[2]["bipartite"] and rows[2]["converged"]


def test_wisdom_rows_factor_is_at_least_one_and_simulation_agrees():
    rows = ev.wisdom_rows(100, 5, types=(C.T_POWER, C.T_RING), seeds=C.LIVE_SEEDS[:2], runs=2000)
    for r in rows:
        assert r["factor"] >= 1.0 and r["n_eff"] <= r["m"] and r["sim_factor"] == pytest.approx(r["factor"], rel=0.12) and r["runs"] == 2000
    assert rows[0]["factor"] > rows[1]["factor"]                                                    # Potenzgesetz verschwendet mehr als der Ring


def test_stubborn_rows_fall_with_receptiveness_and_split_rows_rise_with_the_inside_share():
    rows = ev.stubborn_rows(80, 5, types=(C.T_RANDOM,), seeds=C.LIVE_SEEDS[:2])
    ratios = [r["ratio"] for r in rows]
    assert [r["a"] for r in rows] == list(C.STUBBORN_AS) and ratios == sorted(ratios, reverse=True) and all(0.0 < x < 1.0 for x in ratios)
    assert ev.stubborn_cell(rows, C.T_RANDOM, 0.95) is rows[3]
    srows = ev.split_rows(insides=(0.5, 0.99), alphas=(0.95,), seeds=C.LIVE_SEEDS[:2])
    assert srows[0]["ratio"] < srows[1]["ratio"] and ev.split_cell(srows, 0.99, 0.95) is srows[1]

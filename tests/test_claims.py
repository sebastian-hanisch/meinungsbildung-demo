"""Jede Zahl aus der README wird hier über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, n = 250 (Gitter 256), c = 6, Median über die Netze; Verbünde-Experimente mit 240 Betrieben, c = 6. Zufall läuft über seedbare
Generatoren (ein Strom je Netz und Zweck) und ist plattformstabil. Schritte stehen auf eine Dezimalstelle (Median über acht Netze kann halbzahlig sein), das Band ist ein Schritt; Eigenwerte,
Gewichte und Verhältnisse stehen auf drei Stellen, ihre Bänder 0,0006 fangen die Rundung."""

from functools import lru_cache

import numpy as np
import pytest

import mbd_algorithm as A
import mbd_constants as C
import mbd_evaluation as ev

SW = C.SWEEP_SEEDS
N, CC = 250, 6
EXPECT = {'speed': [{'typ': 'Zufallsnetz', 's': 0.0, 'lam2': 0.735, 'steps': 40.0, 'periodic': 0, 'nets': 8},
           {'typ': 'Zufallsnetz', 's': 0.5, 'lam2': 0.865, 'steps': 83.0, 'periodic': 0, 'nets': 8},
           {'typ': 'Potenzgesetz-Netz', 's': 0.0, 'lam2': 0.794, 'steps': 55.5, 'periodic': 0, 'nets': 8},
           {'typ': 'Potenzgesetz-Netz', 's': 0.5, 'lam2': 0.896, 'steps': 112.0, 'periodic': 0, 'nets': 8},
           {'typ': 'Skalenfreies Netz (BA)', 's': 0.0, 'lam2': 0.716, 'steps': 36.0, 'periodic': 0, 'nets': 8},
           {'typ': 'Skalenfreies Netz (BA)', 's': 0.5, 'lam2': 0.856, 'steps': 76.5, 'periodic': 0, 'nets': 8},
           {'typ': 'Verbünde', 's': 0.0, 'lam2': 0.879, 'steps': 79.5, 'periodic': 0, 'nets': 8},
           {'typ': 'Verbünde', 's': 0.5, 'lam2': 0.939, 'steps': 163.5, 'periodic': 0, 'nets': 8},
           {'typ': 'Ring mit Abkürzungen', 's': 0.0, 'lam2': 0.977, 'steps': 473.5, 'periodic': 0, 'nets': 8},
           {'typ': 'Ring mit Abkürzungen', 's': 0.5, 'lam2': 0.988, 'steps': 951.5, 'periodic': 0, 'nets': 8},
           {'typ': 'Gitter', 's': 0.0, 'lam2': 0.992, 'steps': 1196.5, 'periodic': 0, 'nets': 8},
           {'typ': 'Gitter', 's': 0.5, 'lam2': 0.996, 'steps': 2397.0, 'periodic': 0, 'nets': 8}],
 'deviation': {'Zufallsnetz': 5.57e-09,
               'Potenzgesetz-Netz': 7.84e-09,
               'Skalenfreies Netz (BA)': 5.24e-09,
               'Verbünde': 1.45e-08,
               'Ring mit Abkürzungen': 2.48e-09,
               'Gitter': 2.41e-11},
 'inside': [{'inside': 0.5, 'lam2': 0.878, 'steps': 38.5, 'nets': 8},
            {'inside': 0.8, 'lam2': 0.935, 'steps': 69.5, 'nets': 8},
            {'inside': 0.9, 'lam2': 0.973, 'steps': 163.5, 'nets': 8},
            {'inside': 0.95, 'lam2': 0.988, 'steps': 307.0, 'nets': 8},
            {'inside': 0.99, 'lam2': 0.999, 'steps': 2893.5, 'nets': 8}],
 'bip': [{'c': 4, 's': 0.0, 'lam_min': -1.0, 'converged': False, 'steps': 2000, 'span': 0.04056, 'bipartite': True},
         {'c': 4, 's': 0.1, 'lam_min': -0.8, 'converged': True, 'steps': 1161, 'span': 9.943e-07, 'bipartite': False},
         {'c': 8, 's': 0.0, 'lam_min': -0.518, 'converged': True, 'steps': 680, 'span': 9.88e-07, 'bipartite': False}],
 'wisdom': [{'typ': 'Zufallsnetz', 'm': 249.0, 'n_eff': 215.7, 'factor': 1.154, 'sim_factor': 1.16, 'pi_max': 0.009},
            {'typ': 'Potenzgesetz-Netz', 'm': 250.0, 'n_eff': 56.7, 'factor': 4.412, 'sim_factor': 4.211, 'pi_max': 0.0893},
            {'typ': 'Skalenfreies Netz (BA)', 'm': 250.0, 'n_eff': 125.4, 'factor': 1.993, 'sim_factor': 2.023, 'pi_max': 0.0346},
            {'typ': 'Verbünde', 'm': 250.0, 'n_eff': 217.4, 'factor': 1.15, 'sim_factor': 1.177, 'pi_max': 0.0087},
            {'typ': 'Ring mit Abkürzungen', 'm': 250.0, 'n_eff': 248.0, 'factor': 1.008, 'sim_factor': 1.011, 'pi_max': 0.0057},
            {'typ': 'Gitter', 'm': 256.0, 'n_eff': 249.2, 'factor': 1.027, 'sim_factor': 1.051, 'pi_max': 0.0043}],
 'influence': [{'typ': 'Zufallsnetz', 'hubs': 0.077, 'random': 0.04, 'periphery': 0.012, 'pi_max': 0.009},
               {'typ': 'Potenzgesetz-Netz', 'hubs': 0.289, 'random': 0.04, 'periphery': 0.013, 'pi_max': 0.0893},
               {'typ': 'Skalenfreies Netz (BA)', 'hubs': 0.199, 'random': 0.04, 'periphery': 0.02, 'pi_max': 0.0346},
               {'typ': 'Verbünde', 'hubs': 0.076, 'random': 0.04, 'periphery': 0.011, 'pi_max': 0.0087},
               {'typ': 'Ring mit Abkürzungen', 'hubs': 0.051, 'random': 0.04, 'periphery': 0.038, 'pi_max': 0.0057},
               {'typ': 'Gitter', 'hubs': 0.043, 'random': 0.039, 'periphery': 0.024, 'pi_max': 0.0043}],
 'stubborn': [{'typ': 'Zufallsnetz', 'a': 0.5, 'ratio': 0.538},
              {'typ': 'Zufallsnetz', 'a': 0.8, 'ratio': 0.245},
              {'typ': 'Zufallsnetz', 'a': 0.9, 'ratio': 0.131},
              {'typ': 'Zufallsnetz', 'a': 0.95, 'ratio': 0.068},
              {'typ': 'Zufallsnetz', 'a': 0.99, 'ratio': 0.014},
              {'typ': 'Potenzgesetz-Netz', 'a': 0.5, 'ratio': 0.542},
              {'typ': 'Potenzgesetz-Netz', 'a': 0.8, 'ratio': 0.257},
              {'typ': 'Potenzgesetz-Netz', 'a': 0.9, 'ratio': 0.142},
              {'typ': 'Potenzgesetz-Netz', 'a': 0.95, 'ratio': 0.076},
              {'typ': 'Potenzgesetz-Netz', 'a': 0.99, 'ratio': 0.016},
              {'typ': 'Skalenfreies Netz (BA)', 'a': 0.5, 'ratio': 0.532},
              {'typ': 'Skalenfreies Netz (BA)', 'a': 0.8, 'ratio': 0.24},
              {'typ': 'Skalenfreies Netz (BA)', 'a': 0.9, 'ratio': 0.127},
              {'typ': 'Skalenfreies Netz (BA)', 'a': 0.95, 'ratio': 0.066},
              {'typ': 'Skalenfreies Netz (BA)', 'a': 0.99, 'ratio': 0.014},
              {'typ': 'Verbünde', 'a': 0.5, 'ratio': 0.534},
              {'typ': 'Verbünde', 'a': 0.8, 'ratio': 0.244},
              {'typ': 'Verbünde', 'a': 0.9, 'ratio': 0.133},
              {'typ': 'Verbünde', 'a': 0.95, 'ratio': 0.071},
              {'typ': 'Verbünde', 'a': 0.99, 'ratio': 0.015},
              {'typ': 'Ring mit Abkürzungen', 'a': 0.5, 'ratio': 0.541},
              {'typ': 'Ring mit Abkürzungen', 'a': 0.8, 'ratio': 0.294},
              {'typ': 'Ring mit Abkürzungen', 'a': 0.9, 'ratio': 0.198},
              {'typ': 'Ring mit Abkürzungen', 'a': 0.95, 'ratio': 0.133},
              {'typ': 'Ring mit Abkürzungen', 'a': 0.99, 'ratio': 0.041},
              {'typ': 'Gitter', 'a': 0.5, 'ratio': 0.543},
              {'typ': 'Gitter', 'a': 0.8, 'ratio': 0.29},
              {'typ': 'Gitter', 'a': 0.9, 'ratio': 0.189},
              {'typ': 'Gitter', 'a': 0.95, 'ratio': 0.123},
              {'typ': 'Gitter', 'a': 0.99, 'ratio': 0.044}],
 'split': [{'inside': 0.5, 'a': 0.8, 'ratio': 0.381},
           {'inside': 0.5, 'a': 0.95, 'ratio': 0.123},
           {'inside': 0.5, 'a': 0.99, 'ratio': 0.027},
           {'inside': 0.8, 'a': 0.8, 'ratio': 0.569},
           {'inside': 0.8, 'a': 0.95, 'ratio': 0.229},
           {'inside': 0.8, 'a': 0.99, 'ratio': 0.055},
           {'inside': 0.9, 'a': 0.8, 'ratio': 0.754},
           {'inside': 0.9, 'a': 0.95, 'ratio': 0.41},
           {'inside': 0.9, 'a': 0.99, 'ratio': 0.121},
           {'inside': 0.95, 'a': 0.8, 'ratio': 0.852},
           {'inside': 0.95, 'a': 0.95, 'ratio': 0.563},
           {'inside': 0.95, 'a': 0.99, 'ratio': 0.212},
           {'inside': 0.99, 'a': 0.8, 'ratio': 0.963},
           {'inside': 0.99, 'a': 0.95, 'ratio': 0.855},
           {'inside': 0.99, 'a': 0.99, 'ratio': 0.547}],
 'identities': {'nets': 12, 'fj_dev': 1.11e-15, 'row_dev': 2.22e-16, 'in_range': True},
 'presets': {'Zufallsnetz: Konsens (Voreinstellung)': {'n': 150,
                                                       'm': 150,
                                                       'edges': 450,
                                                       'steps': 40,
                                                       'converged': True,
                                                       'periodic': False,
                                                       'fj': False,
                                                       'lam2': 0.725,
                                                       'consensus': 0.469,
                                                       'mean0': 0.47,
                                                       'n_eff': 132.2,
                                                       'hub_share': 0.114,
                                                       'spread': 0.0,
                                                       'span_last': 6.56e-07,
                                                       'kind': 'success',
                                                       'verdict': 'Nach 40 Schritten sind sich alle Betriebe einig (Spanne unter 10⁻⁶): auf 0,469. Das ist der mit dem '
                                                                  'Grad gewichtete Mittelwert der Anfangsmeinungen (0,469), nicht der einfache (0,470). Das Gewicht im '
                                                                  'Konsens ist ungleich verteilt: die 10 Betriebe mit den meisten Partnern haben zusammen 11,4 % (bei '
                                                                  'gleichem Gewicht wären es 6,7 %); das Netz zählt wie 132 unabhängige Köpfe von 150. Die '
                                                                  'Geschwindigkeit hängt an |λ₂| = 0,725.'},
             'Hubs bestimmen das Ergebnis': {'n': 150,
                                             'm': 150,
                                             'edges': 444,
                                             'steps': 37,
                                             'converged': True,
                                             'periodic': False,
                                             'fj': False,
                                             'lam2': 0.712,
                                             'consensus': 0.233,
                                             'mean0': 0.067,
                                             'n_eff': 91.0,
                                             'hub_share': 0.233,
                                             'spread': 0.0,
                                             'span_last': 6.396e-07,
                                             'kind': 'success',
                                             'verdict': 'Nach 37 Schritten sind sich alle Betriebe einig (Spanne unter 10⁻⁶): auf 0,233. Das ist der mit dem Grad '
                                                        'gewichtete Mittelwert der Anfangsmeinungen (0,233), nicht der einfache (0,067). Das Gewicht im Konsens ist '
                                                        'ungleich verteilt: die 10 Betriebe mit den meisten Partnern haben zusammen 23,3 % (bei gleichem Gewicht wären '
                                                        'es 6,7 %); das Netz zählt wie 91 unabhängige Köpfe von 150. Die Geschwindigkeit hängt an |λ₂| = 0,712.'},
             'Gitter: der Konsens braucht Zeit': {'n': 144,
                                                  'm': 144,
                                                  'edges': 506,
                                                  'steps': 395,
                                                  'converged': True,
                                                  'periodic': False,
                                                  'fj': False,
                                                  'lam2': 0.971,
                                                  'consensus': 0.478,
                                                  'mean0': 0.473,
                                                  'n_eff': 137.7,
                                                  'hub_share': 0.079,
                                                  'spread': 0.0,
                                                  'span_last': 9.788e-07,
                                                  'kind': 'info',
                                                  'verdict': 'Nach 395 Schritten sind sich alle Betriebe einig (Spanne unter 10⁻⁶): auf 0,478. Das ist der mit dem Grad '
                                                             'gewichtete Mittelwert der Anfangsmeinungen (0,478), nicht der einfache (0,473). Das Gewicht im Konsens ist '
                                                             'hier fast gleich verteilt: die 10 Betriebe mit den meisten Partnern haben zusammen 7,9 % (bei gleichem '
                                                             'Gewicht wären es 6,9 %); das Netz zählt wie 138 unabhängige Köpfe von 144. Die Geschwindigkeit hängt an '
                                                             '|λ₂| = 0,971.'},
             'Gitter ohne Selbstvertrauen: kein Konsens': {'n': 144,
                                                           'm': 144,
                                                           'edges': 264,
                                                           'steps': 2000,
                                                           'converged': False,
                                                           'periodic': True,
                                                           'fj': False,
                                                           'lam2': 1.0,
                                                           'consensus': 0.477,
                                                           'mean0': 0.473,
                                                           'n_eff': 141.1,
                                                           'hub_share': 0.076,
                                                           'spread': 0.085,
                                                           'span_last': 0.04807,
                                                           'kind': 'warning',
                                                           'verdict': 'Die Meinungen pendeln: ohne Gewicht auf die eigene Meinung hat die Mittelung auf diesem Netz den '
                                                                      'Eigenwert −1 (das Netz ist bipartit), die Spanne bleibt nach 2000 Schritten bei 0,048. Schon ein '
                                                                      'kleines Selbstgewicht hebt den Eigenwert −1 auf (Experiment 2 zeigt das am Gitter); dann liefe '
                                                                      'das Netz gegen den Konsens 0,477.'},
             'Sturheit: Vielfalt bleibt': {'n': 150,
                                           'm': 150,
                                           'edges': 450,
                                           'steps': 32,
                                           'converged': True,
                                           'periodic': False,
                                           'fj': True,
                                           'lam2': 0.725,
                                           'consensus': 0.469,
                                           'mean0': 0.47,
                                           'n_eff': 132.2,
                                           'hub_share': 0.114,
                                           'spread': 0.069,
                                           'span_last': 0.09523,
                                           'kind': 'info',
                                           'verdict': 'Nach 32 Schritten hat sich das Netz auf das Gleichgewicht eingependelt (Empfänglichkeit 95 %): die Meinungen '
                                                      'bleiben zwischen 0,41 und 0,51, 6,9 % der Anfangsstreuung bleiben erhalten. Reine Mittelung käme auf den Konsens '
                                                      '0,469 (einfacher Mittelwert der Anfangsmeinungen 0,470). Das Gewicht im Konsens ist ungleich verteilt: die 10 '
                                                      'Betriebe mit den meisten Partnern haben zusammen 11,4 % (bei gleichem Gewicht wären es 6,7 %); das Netz zählt wie '
                                                      '132 unabhängige Köpfe von 150.'},
             'Spaltung entlang der Verbünde': {'n': 150,
                                               'm': 149,
                                               'edges': 450,
                                               'steps': 69,
                                               'converged': True,
                                               'periodic': False,
                                               'fj': True,
                                               'lam2': 0.881,
                                               'consensus': 0.47,
                                               'mean0': 0.503,
                                               'n_eff': 130.1,
                                               'hub_share': 0.119,
                                               'spread': 0.215,
                                               'span_last': 0.343,
                                               'kind': 'info',
                                               'verdict': 'Nach 69 Schritten hat sich das Netz auf das Gleichgewicht eingependelt (Empfänglichkeit 95 %): die Meinungen '
                                                          'bleiben zwischen 0,31 und 0,66, 21,5 % der Anfangsstreuung bleiben erhalten. Reine Mittelung käme auf den '
                                                          'Konsens 0,470 (einfacher Mittelwert der Anfangsmeinungen 0,503). Das Gewicht im Konsens ist ungleich '
                                                          'verteilt: die 10 Betriebe mit den meisten Partnern haben zusammen 11,9 % (bei gleichem Gewicht wären es 6,7 '
                                                          '%); das Netz zählt wie 130 unabhängige Köpfe von 149.'}},
 'power_neff_range': [46.0, 103.2],
 'V_speed': 'Bei Selbstgewicht 50 % braucht die Mittelung in „Skalenfreies Netz (BA)“ 76 Schritte (|λ₂| = 0,856), in „Gitter“ 2397 (|λ₂| = 0,996): 31-mal so lang. Je '
            'näher |λ₂| an 1 liegt, desto länger; am langsamsten sind „Gitter“ und „Ring mit Abkürzungen“.',
 'V_inside': 'Liegen 50 % der Kanten innerhalb der Verbünde, ist die Spanne nach 38 Schritten unter 10⁻³ (|λ₂| = 0,878); bei 99 % sind es 2894 Schritte (|λ₂| = 0,999). '
             'Wenige Querkanten sind der Engpass.',
 'V_bip': 'Gitter mit 4 Partnern ohne Selbstgewicht: kleinster Eigenwert −1,000, die Meinungen pendeln (Spanne nach 2000 Schritten 0,041). Mit 10 % Selbstgewicht: '
          'kleinster Eigenwert −0,800, Konvergenz nach 1161 Schritten. Mit 8 Partnern (Diagonalen) ohne Selbstgewicht: kleinster Eigenwert −0,518, Konvergenz nach 680 '
          'Schritten.',
 'V_wisdom': 'Die Fehlervarianz des Konsenses ist das 4,41-Fache des einfachen Mittelwerts in „Potenzgesetz-Netz“ (das Netz zählt wie 57 von 250 unabhängigen Köpfen; '
             'Simulation: 4,21-fach) und nur das 1,01-Fache in „Ring mit Abkürzungen“ (248 von 250). Je ungleicher die Gewichte π, desto weniger Köpfe zählen.',
 'V_influence': 'Haben nur die 10 Betriebe mit den meisten Partnern die Meinung 1, landet der Konsens in „Potenzgesetz-Netz“ bei 28,9 % (bei 10 zufälligen Betrieben im '
                'Mittel 4,0 %, bei den 10 mit den wenigsten Partnern 1,3 %). Mindestens das 1,5-Fache der zufälligen Wahl erreichen die Hubs in: „Zufallsnetz“, '
                '„Potenzgesetz-Netz“, „Skalenfreies Netz (BA)“, „Verbünde“. Darunter bleiben sie in: „Ring mit Abkürzungen“, „Gitter“.',
 'V_stubborn': 'Bei Empfänglichkeit 95 % bleiben in „Skalenfreies Netz (BA)“ 6,6 % der Anfangsstreuung erhalten, in „Ring mit Abkürzungen“ 13,3 %. Je kleiner die '
               'Empfänglichkeit, desto mehr Vielfalt bleibt.',
 'V_split': 'Bei Empfänglichkeit 95 % bleiben von der Anfangsspaltung 12,3 % erhalten, wenn 50 % der Kanten innerhalb der Verbünde liegen, und 85,5 %, wenn es 99 % '
            'sind. Wenige Querkanten und etwas Sturheit halten die Spaltung entlang der Verbünde.'}


@lru_cache(maxsize=None)
def _speed():
    return ev.speed_rows(N, CC, seeds=SW)


@lru_cache(maxsize=None)
def _inside():
    return ev.inside_rows(seeds=SW)


@lru_cache(maxsize=None)
def _bip():
    return ev.bipartite_rows(N)


@lru_cache(maxsize=None)
def _wisdom():
    return ev.wisdom_rows(N, CC, seeds=SW)


@lru_cache(maxsize=None)
def _influence():
    return ev.influence_rows(N, CC, seeds=SW)


@lru_cache(maxsize=None)
def _stubborn():
    return ev.stubborn_rows(N, CC, seeds=SW)


@lru_cache(maxsize=None)
def _split():
    return ev.split_rows(seeds=SW)


def _band(x, digits=3):
    return pytest.approx(x, abs=0.6 * 10.0**-digits)


# --- E2: Geschwindigkeit --------------------------------------------------------------------------------------------------------------------------


def test_speed_table_eigenvalues_and_steps():
    rows = _speed()
    assert len(rows) == len(EXPECT["speed"])
    for got, exp in zip(rows, EXPECT["speed"]):
        assert (got["typ"], got["s"], got["periodic"], got["nets"]) == (exp["typ"], exp["s"], exp["periodic"], exp["nets"])
        assert got["lam2"] == _band(exp["lam2"]) and got["steps"] == pytest.approx(exp["steps"], abs=1.01), (exp["typ"], exp["s"])
    assert ev.verdict_speed(rows)[1] == EXPECT["V_speed"]


def test_ring_and_grid_are_slow_and_the_others_fast_and_self_weight_slows_everything_down():
    by = {(r["typ"], r["s"]): r for r in _speed()}
    for s in C.SPEED_SS:
        fast = max(by[(t, s)]["steps"] for t in (C.T_RANDOM, C.T_BA))
        assert by[(C.T_RING, s)]["steps"] > 5 * fast and by[(C.T_GRID, s)]["steps"] > 10 * fast and by[(C.T_GRID, s)]["steps"] > by[(C.T_RING, s)]["steps"]
        assert by[(C.T_COMMUNITIES, s)]["steps"] > by[(C.T_RANDOM, s)]["steps"]
    for typ in C.ALL_TYPES:
        assert by[(typ, 0.5)]["steps"] > by[(typ, 0.0)]["steps"] and by[(typ, 0.5)]["lam2"] > by[(typ, 0.0)]["lam2"]
        assert by[(typ, 0.5)]["steps"] == pytest.approx(2 * by[(typ, 0.0)]["steps"], rel=0.1)                    # bei s = 0,5 halbiert sich der Abstand der Eigenwerte zu 1: etwa doppelt so viele Schritte


def test_the_consensus_is_the_degree_weighted_mean_in_every_network_type():
    for typ in C.ALL_TYPES:
        worst = 0.0
        for seed in SW:
            _, a, _ = ev.component_of_instance(ev.make_instance(typ, N, CC, seed, layout=False))
            x0 = ev._rng(seed, 3).random(len(a))
            out = A.degroot_run(A.transition_matrix(a, 0.5), x0, C.TOL, C.MAX_STEPS)
            assert out["converged"]
            worst = max(worst, abs(float(np.mean(out["x"])) - A.consensus_value(A.stationary_weights(a), x0)))
        assert worst <= EXPECT["deviation"][typ] * 1.01 and worst < C.TOL, typ


def test_communities_slow_down_the_agreement_with_the_inside_share():
    rows = _inside()
    assert [r["inside"] for r in rows] == list(C.COMM_INSIDES) == [e["inside"] for e in EXPECT["inside"]]
    for got, exp in zip(rows, EXPECT["inside"]):
        assert got["lam2"] == _band(exp["lam2"]) and got["steps"] == pytest.approx(exp["steps"], abs=1.01) and got["nets"] == exp["nets"]
    steps, lam = [r["steps"] for r in rows], [r["lam2"] for r in rows]
    assert steps == sorted(steps) and lam == sorted(lam) and steps[-1] > 50 * steps[0]
    assert ev.verdict_inside(rows)[1] == EXPECT["V_inside"]


def test_the_bipartite_grid_oscillates_and_a_little_self_weight_or_the_diagonals_cure_it():
    rows = _bip()
    for got, exp in zip(rows, EXPECT["bip"]):
        assert (got["c"], got["s"], got["converged"], got["steps"], got["bipartite"]) == (exp["c"], exp["s"], exp["converged"], exp["steps"], exp["bipartite"])
        assert got["lam_min"] == _band(exp["lam_min"]) and got["span"] == pytest.approx(exp["span"], rel=0.01)
    assert rows[0]["span"] > 0.01 and rows[1]["span"] < C.TOL and rows[2]["span"] < C.TOL
    assert ev.verdict_bipartite(rows)[1] == EXPECT["V_bip"]


# --- E3: Weisheit der Menge ---------------------------------------------------------------------------------------------------------------------------


def test_wisdom_of_the_crowd_table():
    rows = _wisdom()
    for got, exp in zip(rows, EXPECT["wisdom"]):
        assert got["typ"] == exp["typ"] and got["m"] == pytest.approx(exp["m"], abs=0.06) and got["n_eff"] == pytest.approx(exp["n_eff"], abs=0.06)
        assert got["factor"] == _band(exp["factor"]) and got["sim_factor"] == _band(exp["sim_factor"]) and got["pi_max"] == pytest.approx(exp["pi_max"], abs=0.00006), exp["typ"]
        assert got["sim_factor"] == pytest.approx(got["factor"], rel=0.08)                                          # Simulation gegen Formel
        assert got["n_eff"] * got["factor"] == pytest.approx(got["m"], rel=0.12)                                      # m = n_eff · (m Σ π²) je Netz; über Mediane nur näherungsweise
    by = {r["typ"]: r for r in rows}
    assert by[C.T_POWER]["n_eff"] < 0.5 * by[C.T_POWER]["m"] and by[C.T_POWER]["factor"] > 3.0 and by[C.T_BA]["factor"] > 1.5
    assert all(by[t]["factor"] < 1.2 for t in (C.T_RANDOM, C.T_COMMUNITIES, C.T_RING, C.T_GRID))
    assert ev.verdict_wisdom(rows)[1] == EXPECT["V_wisdom"]


def test_the_power_law_networks_differ_a_lot_between_the_eight_fixed_networks():
    neff = []
    for seed in SW:
        _, a, _ = ev.component_of_instance(ev.make_instance(C.T_POWER, N, CC, seed, layout=False))
        neff.append(A.effective_sample_size(A.stationary_weights(a)))
    assert min(neff) == pytest.approx(EXPECT["power_neff_range"][0], abs=0.06) and max(neff) == pytest.approx(EXPECT["power_neff_range"][1], abs=0.06)


# --- E1: Wer bestimmt das Ergebnis? ------------------------------------------------------------------------------------------------------------------


def test_influence_table():
    rows = _influence()
    for got, exp in zip(rows, EXPECT["influence"]):
        assert got["typ"] == exp["typ"] and got["hubs"] == _band(exp["hubs"]) and got["random"] == _band(exp["random"]) and got["periphery"] == _band(exp["periphery"]), exp["typ"]
        assert got["pi_max"] == pytest.approx(exp["pi_max"], abs=0.00006)
    by = {r["typ"]: r for r in rows}
    for typ in (C.T_RANDOM, C.T_POWER, C.T_BA, C.T_COMMUNITIES):
        assert by[typ]["hubs"] >= 1.5 * by[typ]["random"], typ
    for typ in (C.T_RING, C.T_GRID):
        assert by[typ]["hubs"] < 1.5 * by[typ]["random"], typ
    assert by[C.T_POWER]["hubs"] > 5 * by[C.T_POWER]["random"] and by[C.T_BA]["hubs"] > 4 * by[C.T_BA]["random"]
    assert all(r["periphery"] < r["random"] < r["hubs"] for r in rows)
    assert ev.verdict_influence(rows)[1] == EXPECT["V_influence"]


# --- E4: Sturheit -----------------------------------------------------------------------------------------------------------------------------------------


def test_stubborn_table_and_the_diversity_that_remains():
    rows = _stubborn()
    assert len(rows) == len(EXPECT["stubborn"])
    for got, exp in zip(rows, EXPECT["stubborn"]):
        assert (got["typ"], got["a"]) == (exp["typ"], exp["a"]) and got["ratio"] == _band(exp["ratio"]), (exp["typ"], exp["a"])
    for typ in C.ALL_TYPES:
        ratios = [ev.stubborn_cell(rows, typ, a)["ratio"] for a in C.STUBBORN_AS]
        assert ratios == sorted(ratios, reverse=True) and 0.0 < ratios[-1] < 0.06 and ratios[0] > 0.5                    # jede Sturheit hält Vielfalt, mehr Empfänglichkeit weniger
    for slow in (C.T_RING, C.T_GRID):
        for fast in (C.T_RANDOM, C.T_POWER, C.T_BA, C.T_COMMUNITIES):
            assert ev.stubborn_cell(rows, slow, 0.95)["ratio"] > 1.4 * ev.stubborn_cell(rows, fast, 0.95)["ratio"]
    assert ev.verdict_stubborn(rows)[1] == EXPECT["V_stubborn"]


# --- E5: Spaltung in Verbünden -------------------------------------------------------------------------------------------------------------------------


def test_split_table_and_the_split_that_stays():
    rows = _split()
    assert len(rows) == len(EXPECT["split"])
    for got, exp in zip(rows, EXPECT["split"]):
        assert (got["inside"], got["a"]) == (exp["inside"], exp["a"]) and got["ratio"] == _band(exp["ratio"]), (exp["inside"], exp["a"])
    for a in C.COMM_AS:
        ratios = [ev.split_cell(rows, i, a)["ratio"] for i in C.COMM_INSIDES]
        assert ratios == sorted(ratios) and ratios[-1] > 2 * ratios[0] and (a < 0.95 or ratios[-1] > 5 * ratios[0])
    for i in C.COMM_INSIDES:
        by_a = [ev.split_cell(rows, i, a)["ratio"] for a in C.COMM_AS]
        assert by_a == sorted(by_a, reverse=True)
    assert ev.split_cell(rows, 0.99, 0.95)["ratio"] > 0.8 and ev.split_cell(rows, 0.5, 0.95)["ratio"] < 0.15
    assert ev.verdict_split(rows)[1] == EXPECT["V_split"]


# --- Identitäten ------------------------------------------------------------------------------------------------------------------------------------------


def test_identities_fj_equilibrium_row_sums_and_range():
    worst_fj, worst_row, nets = 0.0, 0.0, 0
    for typ in C.ALL_TYPES:
        for seed in SW[:2]:
            _, a, _ = ev.component_of_instance(ev.make_instance(typ, N, CC, seed, layout=False))
            rng = ev._rng(seed, 9)
            w = A.transition_matrix(a, float(rng.choice([0.0, 0.3])))
            x0 = rng.random(len(a))
            alpha = rng.uniform(0.3, 0.95, size=len(a))
            star = A.fj_equilibrium(w, alpha, x0)
            worst_fj = max(worst_fj, float(np.max(np.abs(star - A.fj_iterate(w, alpha, x0, 10_000)))))
            worst_row = max(worst_row, float(np.max(np.abs(w.sum(axis=1) - 1.0))))
            assert x0.min() - 1e-12 <= star.min() and star.max() <= x0.max() + 1e-12
            nets += 1
    assert nets == EXPECT["identities"]["nets"] == 12 and worst_fj < 1e-12 and worst_row < 1e-14


# --- Presets ------------------------------------------------------------------------------------------------------------------------------------------------


def test_presets_numbers():
    assert set(EXPECT["presets"]) == set(C.PRESETS)
    for name, exp in EXPECT["presets"].items():
        an = ev.analyse(ev.Settings(**C.PRESETS[name]))
        assert (an.inst.n, an.m, int(sum(len(x) for x in an.inst.adj) // 2), an.steps, an.converged, an.periodic, an.fj) == (exp["n"], exp["m"], exp["edges"], exp["steps"], exp["converged"], exp["periodic"], exp["fj"]), name
        assert an.lam2 == _band(exp["lam2"]) and an.consensus == _band(exp["consensus"]) and an.mean0 == _band(exp["mean0"]) and an.n_eff == pytest.approx(exp["n_eff"], abs=0.06), name
        assert an.hub_share == _band(exp["hub_share"]) and an.spread_ratio == _band(exp["spread"]) and an.run["spans"][-1] == pytest.approx(exp["span_last"], rel=0.01), name
        assert ev.verdict_main(an) == (exp["kind"], exp["verdict"]), name

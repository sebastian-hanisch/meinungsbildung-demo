"""Netze: gültige Netze, Determinismus (auch über Prozesse), Positionen, größte Gruppe, Gruppen je Betrieb, Blockmodell mit einstellbarem Anteil innen."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mbd_algorithm as A
import mbd_constants as C
import mbd_evaluation as ev
import mbd_scenario as S

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("typ", C.ALL_TYPES)
@pytest.mark.parametrize("n,c", [(C.N_MIN, C.C_MIN), (C.N_MIN, C.C_MAX), (C.DEFAULT_N, C.DEFAULT_C), (C.N_MAX, C.C_MAX)])
def test_every_corner_gives_a_valid_network(typ, n, c):
    inst = S.make_instance(typ, n, c, 35)
    assert inst.n >= 2 and len(inst.pos) == inst.n
    for x, y in inst.pos:
        assert math.isfinite(x) and math.isfinite(y) and -1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9
    for a, nb in enumerate(inst.adj):
        assert list(nb) == sorted(set(nb)) and a not in nb
        for b in nb:
            assert a in inst.adj[b]


def test_instances_are_deterministic_per_seed_and_differ_between_seeds():
    a = S.make_instance(C.T_COMMUNITIES, 80, 6, 5)
    assert a.adj == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).adj and a.pos == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).pos
    assert S.make_instance(C.T_COMMUNITIES, 80, 6, 6, layout=False).adj != a.adj


def test_network_and_run_are_independent_of_the_hash_seed_of_the_process():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import hashlib, mbd_scenario as S, mbd_constants as C, mbd_evaluation as E;"
            "h = hashlib.sha256();"
            "[h.update(repr(S.make_instance(t, 60, 5, 7, layout=False).adj).encode()) for t in C.ALL_TYPES];"
            "[h.update(repr((E.analyse(E.Settings(t, 60, 5, 7, 10, 90, m)).x0.tolist(), E.analyse(E.Settings(t, 60, 5, 7, 10, 90, m)).run['steps'])).encode()) for t in C.ALL_TYPES for m in C.MODES];"
            "print(h.hexdigest())")
    outs = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        outs.append(subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code, str(ROOT)], capture_output=True, text=True, env=env, check=True).stdout.strip())
    assert outs[0] == outs[1] and len(outs[0]) == 64


def test_layout_false_skips_the_force_layout_and_unknown_type_raises():
    inst = S.make_instance(C.T_RANDOM, 60, 4, 1, layout=False)
    assert set(inst.pos) == {(0.5, 0.5)}
    with pytest.raises(ValueError):
        S.make_instance("gibt es nicht", 60, 4, 1)


def test_component_of_instance_is_connected_and_labels_follow_the_communities():
    for typ in C.ALL_TYPES:
        inst = S.make_instance(typ, 90, 3, 11, layout=False)
        nodes, a, groups = ev.component_of_instance(inst)
        assert len(nodes) == len(a) == len(groups) and a.sum(axis=1).min() >= 1 and (a == a.T).all()
        assert len(A.largest_component(ev.matrix_to_adj(a))) == len(nodes)                         # das Teilnetz ist zusammenhängend
        assert groups.min() >= 0 and groups.max() < C.SPLIT_GROUPS
    inst = S.make_instance(C.T_COMMUNITIES, 120, 6, 3, layout=False)
    nodes, _, groups = ev.component_of_instance(inst)
    assert [int(g) for g in groups] == [inst.groups[v] for v in nodes]                           # Verbünde: die Gruppen des Netzes
    plain = S.make_instance(C.T_RANDOM, 120, 6, 3, layout=False)
    _, _, blocks = ev.component_of_instance(plain)
    assert np.bincount(blocks).tolist() == [len(blocks) // C.SPLIT_GROUPS] * C.SPLIT_GROUPS or max(np.bincount(blocks)) - min(np.bincount(blocks)) <= 1


def test_component_of_matrix_drops_isolated_nodes():
    a = np.zeros((6, 6))
    for i, j in ((0, 1), (1, 2), (3, 4)):
        a[i, j] = a[j, i] = 1.0
    nodes, sub = ev.component_of_matrix(a)
    assert nodes == [0, 1, 2] and sub.shape == (3, 3) and sub.sum() == 4
    assert ev.matrix_to_adj(sub) == ((1,), (0, 2), (1,))


@pytest.mark.parametrize("inside", C.COMM_INSIDES)
def test_sbm_net_has_the_requested_inside_share_and_is_deterministic(inside):
    a, lab = ev.sbm_net(C.COMM_N, C.COMM_C, inside, 100000)
    assert a.shape == (C.COMM_N, C.COMM_N) and (a == a.T).all() and np.trace(a) == 0
    same = lab[:, None] == lab[None, :]
    assert (a * same).sum() / a.sum() == pytest.approx(inside, abs=0.05)
    assert a.sum() / C.COMM_N == pytest.approx(C.COMM_C, rel=0.1)                                 # mittlerer Grad
    assert (ev.sbm_net(C.COMM_N, C.COMM_C, inside, 100000)[0] == a).all() and not (ev.sbm_net(C.COMM_N, C.COMM_C, inside, 100001)[0] == a).all()


def test_sbm_net_differs_between_inside_shares_for_the_same_seed():
    assert not (ev.sbm_net(C.COMM_N, C.COMM_C, 0.5, 100000)[0] == ev.sbm_net(C.COMM_N, C.COMM_C, 0.8, 100000)[0]).all()

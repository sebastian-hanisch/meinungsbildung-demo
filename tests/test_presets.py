"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset zeigt, wofür es benannt ist."""

import numpy as np
import pytest

import mbd_constants as C
import mbd_evaluation as ev
import mbd_presets as P


def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["netztyp"] in C.ALL_TYPES and C.N_MIN <= p["n"] <= C.N_MAX and C.C_MIN <= p["c"] <= C.C_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert C.S_MIN <= p["s"] <= C.S_MAX and C.A_MIN <= p["a"] <= C.A_MAX and p["mode"] in C.MODES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[C.DEFAULT_PRESET_NAME]) == ev.Settings()
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET_NAME


def test_setting_specs_cover_every_preset_key_and_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C.PRESETS[C.DEFAULT_PRESET_NAME][key], key


def test_every_preset_differs_from_the_default_in_something_besides_the_network_type():
    default = C.PRESETS[C.DEFAULT_PRESET_NAME]
    for name, p in C.PRESETS.items():
        if name != C.DEFAULT_PRESET_NAME:
            assert any(p[k] != default[k] for k in p if k != "netztyp"), name


def test_each_preset_shows_what_its_name_says():
    an = {name: ev.analyse(_settings(name)) for name in C.PRESETS}
    base = an["Zufallsnetz: Konsens (Voreinstellung)"]
    assert base.converged and not base.periodic and not base.fj and base.spread_ratio < 1e-5 and base.steps < 100
    hubs = an["Hubs bestimmen das Ergebnis"]
    k = C.HUBS_K
    assert hubs.inst.typ == C.T_BA and hubs.hub_share >= 1.5 * k / hubs.m and hubs.consensus >= 2.0 * hubs.mean0 and hubs.converged
    assert hubs.consensus == pytest.approx(hubs.hub_share)                                                         # nur die zehn Hubs haben die Meinung 1: der Konsens ist ihr Gewicht
    slow = an["Gitter: der Konsens braucht Zeit"]
    assert slow.inst.typ == C.T_GRID and slow.converged and not slow.periodic and slow.steps >= 5 * base.steps and slow.lam2 > 0.95
    bip = an["Gitter ohne Selbstvertrauen: kein Konsens"]
    assert bip.inst.typ == C.T_GRID and bip.periodic and not bip.converged and bip.steps == C.PERIODIC_STEPS and bip.settings.s == 0 and bip.run["spans"][-1] > 1e-3
    stubborn = an["Sturheit: Vielfalt bleibt"]
    assert stubborn.fj and stubborn.converged and 0.0 < stubborn.spread_ratio < 0.2 and stubborn.run["spans"][-1] > 0.0
    split = an["Spaltung entlang der Verbünde"]
    assert split.inst.typ == C.T_COMMUNITIES and split.fj and split.settings.mode == C.M_SPLIT and split.spread_ratio > 2.0 * stubborn.spread_ratio
    assert set(np.unique(split.x0).tolist()) == {0.0, 1.0}


def test_every_preset_runs_and_the_verdict_is_a_known_kind():
    for name in C.PRESETS:
        an = ev.analyse(_settings(name))
        kind, text = ev.verdict_main(an)
        assert kind in ("success", "info", "warning") and text
        assert set(an.frames) >= {0, an.steps} and all(len(x) == an.m for x in an.frames.values())

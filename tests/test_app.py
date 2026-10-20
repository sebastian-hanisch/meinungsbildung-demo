"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Slider an seine Grenzen,
prüft Permalink, Netztypen, Modi, den Schritt-Regler und alle fünf Experimente; es darf kein Python-Fehler auftreten."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import mbd_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
EXPERIMENTS = ("influence", "speed", "wisdom", "stubborn", "split")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert [m.label for m in at.metric] == ["Konsens", "Schritte", "|λ₂|", "n_eff", "Streuung übrig"]
    assert _metric(at, "Konsens") == "0,469" and _metric(at, "Schritte") == "40" and _metric(at, "|λ₂|") == "0,725" and _metric(at, "n_eff") == "132" and _metric(at, "Streuung übrig") == "0,0 %"
    assert not at.error and not at.warning
    assert any(s.value.startswith("Nach 40 Schritten sind sich alle Betriebe einig") for s in at.success)


def test_every_preset_click_does_not_raise():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


def test_the_bipartite_preset_shows_a_warning_and_the_fj_preset_a_remaining_spread():
    at = _fresh_app()
    at.sidebar.button(key="preset_Gitter ohne Selbstvertrauen: kein Konsens").click().run(timeout=TIMEOUT)
    assert not at.exception and _metric(at, "Schritte") == str(C.PERIODIC_STEPS) and _metric(at, "|λ₂|") == "1,000" and any("Die Meinungen pendeln" in w.value for w in at.warning)
    at.sidebar.button(key="preset_Sturheit: Vielfalt bleibt").click().run(timeout=TIMEOUT)
    assert not at.exception and not _metric(at, "Streuung übrig").startswith("0,0") and any("Empfänglichkeit 95 %" in i.value for i in at.info)


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels[:4] == ["Betriebe n", "Partner je Betrieb c", "Selbstgewicht (%)", "Empfänglichkeit (%)"], labels
    for index in range(4):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.sidebar.slider[index]
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value)).run(timeout=TIMEOUT)
            assert not at.exception, (labels[index], edge, [str(e) for e in at.exception])


def test_all_sliders_at_their_minimum_and_all_at_their_maximum_do_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        for index in range(len(at.sidebar.slider)):
            slider = at.sidebar.slider[index]
            slider.set_value(int(slider.min if edge == "min" else slider.max))
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("typ", C.ALL_TYPES)
def test_every_network_type_renders(typ):
    at = _fresh_app()
    at.selectbox(key="netztyp_select").set_value(typ).run(timeout=TIMEOUT)
    assert not at.exception, (typ, [str(e) for e in at.exception])


@pytest.mark.parametrize("mode", C.MODES)
def test_every_initial_opinion_mode_renders(mode):
    at = _fresh_app()
    at.selectbox(key="mode_select").set_value(mode).run(timeout=TIMEOUT)
    assert not at.exception, (mode, [str(e) for e in at.exception])


def test_the_step_slider_moves_through_the_frames_and_resets_with_the_settings():
    at = _fresh_app()
    slider = at.select_slider[0]
    assert slider.label == "Schritt" and slider.value == 0 and int(slider.options[-1]) == 40
    slider.set_value(40).run(timeout=TIMEOUT)
    assert not at.exception and at.select_slider[0].value == 40
    at.sidebar.slider(key="s_slider").set_value(50).run(timeout=TIMEOUT)
    assert not at.exception and at.select_slider[0].value == 0 and int(at.select_slider[0].options[-1]) > 40       # neue Einstellungen: neuer Regler, zurück auf Schritt 0


def test_a_new_run_changes_the_seed_and_the_result():
    at = _fresh_app()
    before = (at.number_input(key="seed_input").value, _metric(at, "Konsens"))
    at.button[[b.label for b in at.button].index("🎲 Neuer Lauf")].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before[0]


def test_permalink_values_are_clamped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "5"
    at.query_params["c"] = "99"
    at.query_params["selbst"] = "999"
    at.query_params["empfaenglich"] = "1"
    at.query_params["start"] = "gibt es nicht"
    at.query_params["typ"] = "gibt es nicht"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="n_slider").value == C.N_MIN and at.slider(key="c_slider").value == C.C_MAX
    assert at.slider(key="s_slider").value == C.S_MAX and at.slider(key="a_slider").value == C.A_MIN
    assert at.selectbox(key="mode_select").value == C.DEFAULT_MODE and at.selectbox(key="netztyp_select").value == C.T_RANDOM


def test_permalink_selects_type_and_mode_and_the_address_mirrors_the_settings():
    at = AppTest.from_file(APP_PATH)
    at.query_params["typ"] = C.T_BA
    at.query_params["start"] = C.M_HUBS
    at.run(timeout=TIMEOUT)
    assert not at.exception and at.selectbox(key="netztyp_select").value == C.T_BA and at.selectbox(key="mode_select").value == C.M_HUBS
    assert dict(at.query_params) == {"typ": C.T_BA, "n": "150", "c": "6", "seed": "35", "selbst": "0", "empfaenglich": "100", "start": C.M_HUBS}


def test_every_experiment_runs_and_is_invalidated_by_a_setting_change():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    at.sidebar.slider(key="n_slider").set_value(C.N_MIN).run(timeout=TIMEOUT)         # kleines Netz: schnell
    for name in EXPERIMENTS:
        at.button(key=f"run_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
    texts = " ".join(x.value for x in list(at.info) + list(at.warning) + list(at.success))
    for fragment in ("Haben nur die 10 Betriebe mit den meisten Partnern", "Bei Selbstgewicht 50 % braucht die Mittelung", "Liegen 50 % der Kanten innerhalb der Verbünde", "Gitter mit 4 Partnern ohne Selbstgewicht",
                     "Die Fehlervarianz des Konsenses", "Bei Empfänglichkeit 95 % bleiben in", "Bei Empfänglichkeit 95 % bleiben von der Anfangsspaltung"):
        assert fragment in texts, fragment
    at.sidebar.slider(key="n_slider").set_value(C.N_MIN + 10).run(timeout=TIMEOUT)
    assert not at.exception
    assert sum("neu starten" in i.value for i in at.info) >= len(EXPERIMENTS) - 1       # die Experimente 1 bis 4 hängen von n ab (die Spaltung in Verbünden nicht)

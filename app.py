"""Meinungsbildung: wer bestimmt das Ergebnis? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Reihe „Netzwerkanalyse“ und Wurzel des Anpassungs-Arms: Betriebe gleichen ihre Einschätzung mit der ihrer Partner ab (DeGroot) und halten einen Teil ihrer Anfangsmeinung fest
(Friedkin–Johnsen). Die Demo zeigt, wohin das Netz läuft, wie schnell, wer das Ergebnis bestimmt (der Konsens ist ein mit dem Grad gewichteter Mittelwert), wie viel die „Weisheit der Menge“ dann noch
wert ist und wie viel Vielfalt bei etwas Sturheit und in Verbünden bleibt. Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mbd_constants as C
from mbd_evaluation import (
    Settings,
    analyse,
    bipartite_rows,
    de,
    inside_rows,
    influence_rows,
    pct,
    speed_rows,
    split_rows,
    stubborn_rows,
    verdict_bipartite,
    verdict_influence,
    verdict_inside,
    verdict_main,
    verdict_speed,
    verdict_split,
    verdict_stubborn,
    verdict_wisdom,
    wisdom_rows,
)
from mbd_layout import dunkles_design, kopf_css
from mbd_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mbd_visualization import (
    build_influence,
    build_inside,
    build_network,
    build_speed,
    build_spans,
    build_split,
    build_stubborn,
    build_wisdom,
)

st.set_page_config(page_title="Meinungsbildung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _influence(n, c):
    return influence_rows(n, c)


@st.cache_data(show_spinner=False)
def _speed(n, c):
    return speed_rows(n, c), inside_rows(), bipartite_rows(n)


@st.cache_data(show_spinner=False)
def _wisdom(n, c):
    return wisdom_rows(n, c)


@st.cache_data(show_spinner=False)
def _stubborn(n, c):
    return stubborn_rows(n, c)


@st.cache_data(show_spinner=False)
def _split():
    return split_rows()


def experiment(name, label, key, compute, show, hint):
    """Teure Zusatzansicht erst auf Klick; das Ergebnis bleibt nur für genau diese Einstellungen stehen (Session State)."""
    state_key = f"exp_{name}"
    if st.button(label, key=f"run_{name}", help=hint):
        with st.spinner("Rechne …"):
            result = compute()
        st.session_state[state_key] = (key, result)
    state = st.session_state.get(state_key)
    if state is None:
        st.caption(f"Noch nicht gerechnet. {hint}")
        return
    if state[0] != key:
        st.info("Die Einstellungen haben sich seit der letzten Rechnung geändert. Bitte neu starten.")
        return
    show(state[1])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mbd_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🗳️ Meinungsbildung: wer bestimmt das Ergebnis?")
st.markdown(
    """
**Neuntes Stück** der Reihe „Netzwerkanalyse“ und Wurzel des **Anpassungs-Arms**: in den Ausbreitungs-Stücken 5 bis 8 und 13 ändert sich der Zustand eines Betriebs nur in eine Richtung (gesund, dann befallen). Hier **gleichen sich die Betriebe an** und können zurückkehren, sich zersplittern oder erstarren. In einem Netz von Subunternehmern und Lieferanten schätzt jeder Betrieb eine Zahl (zum Beispiel „wie lange dauert die Umstellung auf das neue Verfahren“, eine Zahl zwischen 0 und 1) und gleicht sie laufend mit der seiner Partner ab. Die Demo stellt das **DeGroot-Modell** (jeder ersetzt seine Meinung durch den Mittelwert seiner Partner) dem **Friedkin–Johnsen-Modell** (ein Teil der Anfangsmeinung bleibt) gegenüber. Die Verbindung zur [Zentralität](https://sebastianhanisch-centrality-demo.streamlit.app/): das Gewicht, mit dem ein Betrieb in den Konsens eingeht, ist eine stationäre Verteilung wie beim PageRank, nur ohne Dämpfung.

Die Kernfrage: **Wo landet das Netz, wie schnell, und wer bestimmt den Wert?** Der Hook: der Konsens ist **kein Mittelwert der Köpfe, sondern ein mit dem Grad gewichteter**; wenige Hubs bestimmen den Wert, und die „Weisheit der Menge“ schrumpft entsprechend. Dazu: Verbünde bremsen die Einigung, und schon etwas Sturheit lässt einen Rest der Vielfalt stehen, in Verbünden besonders. Die Ergebnisse sind **Modellbefunde**, keine Aussagen über die Wirklichkeit.

Weiter unten: fünf Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Meinungsbildung", expanded=True):
    st.markdown(
        """
1. **DeGroot:** in jedem Schritt ersetzt jeder Betrieb seine Meinung durch den **Mittelwert seiner Partner**. Mit dem **Selbstgewicht** *s* zählt die eigene Meinung zu *s* mit, die Partner teilen sich den Rest zu gleichen Teilen.
2. **Konsens:** in einem zusammenhängenden Netz laufen alle Meinungen auf **denselben** Wert. Er ist der **mit dem Grad gewichtete** Mittelwert der Anfangsmeinungen: ein Betrieb mit doppelt so vielen Partnern zählt doppelt.
3. **Friedkin–Johnsen:** ein Betrieb mit der **Empfänglichkeit** *a* überlässt den Partnern den Anteil *a* seiner Meinung und hält den Rest (die **Sturheit** 1 − *a*) an seiner Anfangsmeinung fest. Das Netz läuft dann auf ein **Gleichgewicht**, in dem nicht alle gleich denken.
4. **Ausgewertet** wird die größte zusammenhängende Gruppe; isolierte Betriebe bleiben bei ihrer Meinung und erscheinen nicht.
        """
    )
    st.markdown(
        """
| Größe | Bedeutung hier |
|---|---|
| **Selbstgewicht** *s* | Anteil der eigenen Meinung in der nächsten Runde; 0 ist die reine Partner-Mittelung |
| **Empfänglichkeit** *a* | Anteil, den der Betrieb den Partnern überlässt; 100 % ist reines DeGroot |
| **Konsens** | Grenzwert der Meinungen: Σ *d*ᵢ *x*ᵢ(0) / Σ *d*ᵢ mit dem Grad *d*ᵢ |
| **\\|λ₂\\|** | zweitgrößter Eigenwertbetrag der Mittelungsmatrix: je näher an 1, desto langsamer die Einigung |
| **n_eff** | so viele unabhängige Köpfe ist das Netz für den Konsens wert: 1 / Σ πᵢ² |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um einen Lauf zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    typ = st.selectbox("Netztyp", C.ALL_TYPES, key="netztyp_select", help="Welches Netz erzeugt wird.")
    n = st.slider("Betriebe n", *bounds("n_slider"), key="n_slider", help="Zahl der Betriebe (beim Gitter auf ein Quadrat gerundet; ausgewertet wird die größte zusammenhängende Gruppe).")
    c = st.slider("Partner je Betrieb c", *bounds("c_slider"), key="c_slider", help="Was c bedeutet, hängt vom Netztyp ab (siehe darunter).")
    st.caption(C.C_MEANING[typ])
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt das Netz und die zufälligen Anfangsmeinungen.")
    st.button("🎲 Neuer Lauf", width="stretch", on_click=randomize_seed)
    st.markdown("**Meinungsbildung**")
    s = st.slider("Selbstgewicht (%)", *bounds("s_slider"), key="s_slider", help="Anteil der eigenen Meinung in der nächsten Runde. Bei 0 mittelt jeder nur über die Partner; auf bipartiten Netzen (zum Beispiel dem Gitter mit vier Partnern) pendeln die Meinungen dann.")
    a = st.slider("Empfänglichkeit (%)", *bounds("a_slider"), key="a_slider", help="Anteil, den der Betrieb den Partnern überlässt. 100 % ist reines DeGroot (Konsens); darunter hält jeder Betrieb den Rest an seiner Anfangsmeinung fest (Friedkin–Johnsen).")
    mode = st.selectbox("Anfangsmeinungen", C.MODES, key="mode_select", help="Wer mit welcher Meinung beginnt.")
    st.caption(C.MODE_HELP[mode])

sync_query_params({"netztyp_select": typ, "n_slider": int(n), "c_slider": int(c), "seed_input": int(seed), "s_slider": int(s), "a_slider": int(a), "mode_select": mode})

settings = Settings(typ, int(n), int(c), int(seed), int(s), int(a), mode)
with st.spinner("Lasse die Meinungen laufen …"):
    an = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wohin laufen die Meinungen?")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Konsens", de(an.consensus, 3), delta=f"ungewichtet {de(an.mean0, 3)}", delta_color="off")
m2.metric("Schritte", f"{an.steps}", delta=("Obergrenze" if not an.converged else "bis 10⁻⁶"), delta_color="off")
m3.metric("|λ₂|", de(an.lam2, 3), delta="pendelt" if an.periodic and not an.fj else None, delta_color="off")
m4.metric("n_eff", de(an.n_eff, 0), delta=f"von {an.m} Köpfen", delta_color="off")
m5.metric("Streuung übrig", f"{pct(an.spread_ratio, 1)} %", help="Streuung (Standardabweichung) des Endzustands in Prozent der Streuung der Anfangsmeinungen: 0 % ist ein Konsens, 100 % hat sich nichts verändert.")
st.caption("Konsens: Wert, auf den die reine Mittelung läuft (ungewichtet: einfacher Mittelwert der Anfangsmeinungen). n_eff: Zahl unabhängiger Köpfe, die das Netz für den Konsens wert ist. Streuung übrig: Streuung des Endzustands in Prozent der Anfangsstreuung.")
steps = sorted(an.frames)
step_key = "step_" + "_".join(str(v) for v in (typ, n, c, seed, s, a, mode))
step = st.select_slider("Schritt", options=steps, value=steps[0], key=step_key, help="Blättert durch die Meinungsbildung: ganz links die Anfangsmeinungen, ganz rechts der Endzustand.")
st.markdown("**Das Netz im Lauf** (Farbe: Meinung im gewählten Schritt; Größe: Gewicht des Betriebs im Konsens)")
if an.m < an.inst.n:
    st.caption(f"Ausgewertet wird die größte zusammenhängende Gruppe mit {an.m} von {an.inst.n} Betrieben.")
st.plotly_chart(build_network(an, step), width="stretch", key="net_chart")
st.markdown("**Spanne der Meinungen über die Schritte** (logarithmisch; Spanne = größte minus kleinste Meinung)")
st.plotly_chart(build_spans(an, step), width="stretch", key="span_chart")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wer bestimmt das Ergebnis?")
kind, text = verdict_main(an)
getattr(st, kind)(text)
st.caption("Ein einzelner Lauf ist ein Zufallsbeispiel; die Experimente rechnen über feste Netze.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

n_exp, c_exp = settings.n, settings.c
st.subheader("🔬 Experiment 1: Wer bestimmt das Ergebnis?")
st.caption(f"Sechs Netztypen, n = {n_exp}, c = {c_exp}: nur zehn Betriebe haben die Meinung 1, alle anderen 0. Gerechnet wird der Konsenswert (Median über {len(C.LIVE_SEEDS)} feste Netze); zehn zufällige Betriebe geben im Mittel genau 10 / m.")


def show_influence(rows):
    st.plotly_chart(build_influence(rows), width="stretch", key="influence_chart")
    kind_, text_ = verdict_influence(rows)
    getattr(st, kind_)(text_)


experiment("influence", "Einfluss messen", (n_exp, c_exp), lambda: _influence(n_exp, c_exp), show_influence, "Rechnet sechs Netztypen, weniger als eine Sekunde.")

st.markdown("---")

st.subheader("🔬 Experiment 2: Wie schnell einigen sich die Betriebe?")
st.caption(f"Schritte bis die Spanne unter 10⁻⁶ fällt, je Netztyp bei Selbstgewicht 0 und 50 % (n = {n_exp}, c = {c_exp}, Median über {len(C.LIVE_SEEDS)} feste Netze); dazu Verbünde mit einstellbarem Anteil der Kanten innen ({C.COMM_N} Betriebe) und der Sonderfall des bipartiten Gitters.")


def show_speed(result):
    rows, inside, bip = result
    for s_ in C.SPEED_SS:
        st.markdown(f"**Schritte bei Selbstgewicht {pct(s_)} %** (logarithmisch)")
        st.plotly_chart(build_speed(rows, s_), width="stretch", key=f"speed_chart_{int(100 * s_)}")
    kind_, text_ = verdict_speed(rows)
    getattr(st, kind_)(text_)
    st.markdown("**Verbünde: wie bremsen wenige Querkanten?** (Selbstgewicht 50 %, Spanne unter 10⁻³)")
    st.plotly_chart(build_inside(inside), width="stretch", key="inside_chart")
    kind_, text_ = verdict_inside(inside)
    getattr(st, kind_)(text_)
    st.markdown("**Sonderfall: das bipartite Gitter**")
    st.markdown("| Gitter | Selbstgewicht | kleinster Eigenwert | Ergebnis |\n|---|---|---|---|\n" + "\n".join(
        f"| {r['c']} Partner | {pct(r['s'])} % | {de(r['lam_min'], 3)} | " + (f"konvergiert nach {r['steps']} Schritten" if r["converged"] else f"pendelt (Spanne {de(r['span'], 3)} nach {r['steps']} Schritten)") + " |" for r in bip))
    kind_, text_ = verdict_bipartite(bip)
    getattr(st, kind_)(text_)


experiment("speed", "Geschwindigkeit messen", (n_exp, c_exp), lambda: _speed(n_exp, c_exp), show_speed, "Rechnet sechs Netztypen zweimal, dazu fünf Verbünde und drei Gitter, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 3: Die Weisheit der Menge")
st.caption(f"Jeder Betrieb beginnt mit dem wahren Wert plus unabhängigem Rauschen. Wie viel schlechter als der einfache Mittelwert aller Köpfe ist der Konsens? Formel m · Σ πᵢ² gegen eine Simulation mit {C.WISDOM_RUNS} Wiederholungen je Netz (n = {n_exp}, c = {c_exp}, Median über {len(C.LIVE_SEEDS)} feste Netze).")


def show_wisdom(rows):
    st.plotly_chart(build_wisdom(rows), width="stretch", key="wisdom_chart")
    kind_, text_ = verdict_wisdom(rows)
    getattr(st, kind_)(text_)


experiment("wisdom", "Weisheit messen", (n_exp, c_exp), lambda: _wisdom(n_exp, c_exp), show_wisdom, "Rechnet sechs Netztypen mit je 1000 Wiederholungen, wenige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 4: Wie viel Vielfalt hält die Sturheit?")
st.caption(f"Friedkin–Johnsen bei Selbstgewicht 0 und zufälligen Anfangsmeinungen: Streuung des Gleichgewichts in % der Anfangsstreuung über die Empfänglichkeit {', '.join(pct(x) for x in C.STUBBORN_AS)} % (n = {n_exp}, c = {c_exp}, Median über {len(C.LIVE_SEEDS)} feste Netze).")


def show_stubborn(rows):
    st.plotly_chart(build_stubborn(rows), width="stretch", key="stubborn_chart")
    kind_, text_ = verdict_stubborn(rows)
    getattr(st, kind_)(text_)


experiment("stubborn", "Sturheit messen", (n_exp, c_exp), lambda: _stubborn(n_exp, c_exp), show_stubborn, "Löst 120 Gleichungssysteme, wenige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 5: Spaltung in Verbünden")
st.caption(f"Sechs Verbünde ({C.COMM_N} Betriebe, c = {C.COMM_C}), drei starten bei +1, drei bei −1. Wie viel der Spaltung bleibt im Gleichgewicht, wenn der Anteil der Kanten innerhalb der Verbünde von {pct(C.COMM_INSIDES[0])} % auf {pct(C.COMM_INSIDES[-1])} % steigt? Median über {len(C.LIVE_SEEDS)} feste Netze.")


def show_split(rows):
    st.plotly_chart(build_split(rows), width="stretch", key="split_chart")
    kind_, text_ = verdict_split(rows)
    getattr(st, kind_)(text_)


experiment("split", "Spaltung messen", "fest", _split, show_split, "Löst 60 Gleichungssysteme, einige Sekunden.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gleiches Gewicht je Partner, ungerichtetes Netz** | Jeder Betrieb hört auf alle Partner gleich. Mit unterschiedlichem Vertrauen oder einseitigen Beziehungen ist das Gewicht im Konsens nicht mehr der Grad; es ist der linke Eigenvektor der neuen Matrix. | – |
| **Selbstgewicht als Kunstgriff** | Ohne eigenes Gewicht pendeln Meinungen auf bipartiten Netzen; ein Selbstgewicht macht das Modell stabil, ist aber eine Annahme, die die Geschwindigkeit verändert. | – |
| **Gleiche Empfänglichkeit für alle** | Die Experimente setzen für alle Betriebe dieselbe Sturheit. Verschieden sture Betriebe (ein Betrieb ganz stur) verschieben das Gleichgewicht; als Rechnung ist das im Kern geprüft, in der Demo nicht einstellbar. | – |
| **Mittelwert statt Zufall** | Die Betriebe mitteln, statt die Meinung eines Partners zu übernehmen. Beim Übernehmen entscheidet der Zufall über den Gewinner. | Stück 10 (Voter-Modell) |
| **Vertrauen ohne Grenze** | Jeder hört auf jeden Partner, egal wie weit dessen Meinung entfernt ist. Wer nur auf ähnliche Meinungen hört, zersplittert bei kleinem Vertrauensradius. | Stück 11 (begrenztes Vertrauen) |
| **Festes Netz** | Das Netz ändert sich nicht mit den Meinungen. Wer sich nach Gleichgesinnten umsieht, verändert die Struktur. | Stück 12 (Schelling) |
| **Größte Gruppe** | Isolierte Betriebe und kleine Inseln bleiben unberücksichtigt. | – |
| **Stichproben** | Die Experimente nehmen Mediane über wenige feste Netze (in der App vier, in der README acht) bei n = 250 bzw. 240; die Streuung zwischen den Netzen ist nur für n_eff im Potenzgesetz-Netz ausgewiesen (README). | – |
| **Modellbefund** | Die Demo zeigt Verhalten im Modell, nicht die Wirkung in der Wirklichkeit. Wie weit Meinungsmodelle echte Daten treffen, ist offen (Flache u. a. 2017). | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Mittelung (DeGroot).** Mit der Nachbarschaftsmatrix $A$, der Gradmatrix $D$ und dem Selbstgewicht $s$ ist die Mittelungsmatrix

$$W = s\,I + (1 - s)\,D^{-1}A$$

und die Meinungen entwickeln sich als

$$x(t+1) = W\,x(t)$$

**Konsens.** Die Matrix $W$ ist zeilenstochastisch. Ihr linker Eigenvektor zum Eigenwert 1 ist bei ungerichtetem Netz

$$\pi_i = \frac{d_i}{\sum_j d_j}$$

Ist das Netz zusammenhängend und $W$ nicht periodisch (nicht bipartit oder $s > 0$), dann gilt $x(t) \to x_\infty \mathbf{1}$ mit

$$x_\infty = \sum_i \pi_i\, x_i(0)$$

**Geschwindigkeit.** Der Abstand zum Konsens fällt wie die Potenz des zweitgrößten Eigenwertbetrags $|\lambda_2|$ von $W$:

$$\lVert x(t) - x_\infty \mathbf{1} \rVert \;\sim\; |\lambda_2|^{\,t}$$

Bei $s = 0$ auf einem bipartiten Netz hat $W$ den Eigenwert $-1$; dann konvergiert die Mittelung nicht.

**Weisheit der Menge** (Golub und Jackson 2010). Sind die Anfangsmeinungen der wahre Wert plus unabhängiges Rauschen mit Varianz $\sigma^2$, dann hat der Konsens die Varianz

$$\mathrm{Var}(x_\infty) = \sigma^2 \sum_i \pi_i^{2}$$

und das Netz zählt wie

$$n_{\mathrm{eff}} = \frac{1}{\sum_i \pi_i^{2}}$$

unabhängige Köpfe (beim einfachen Mittelwert aller $m$ Köpfe wäre die Varianz $\sigma^2/m$).

**Friedkin–Johnsen.** Mit der Empfänglichkeit $\Lambda = \mathrm{diag}(a_i)$ ist die Iteration

$$x(t+1) = \Lambda W\,x(t) + (I - \Lambda)\,x(0)$$

und für $a_i < 1$ gibt es das eindeutige Gleichgewicht

$$x^{*} = (I - \Lambda W)^{-1} (I - \Lambda)\,x(0)$$

Die Demo löst dieses lineare Gleichungssystem und prüft es gegen die Iteration.

**Literatur.**
- DeGroot, M. H. (1974): *Reaching a Consensus.* Journal of the American Statistical Association 69(345), 118–121.
- Friedkin, N. E., Johnsen, E. C. (1990): *Social Influence and Opinions.* Journal of Mathematical Sociology 15(3–4), 193–205.
- Golub, B., Jackson, M. O. (2010): *Naïve Learning in Social Networks and the Wisdom of Crowds.* American Economic Journal: Microeconomics 2(1), 112–149.
- Flache, A. u. a. (2017): *Models of Social Influence: Towards the Next Frontiers.* Journal of Artificial Societies and Social Simulation 20(4).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

"""Kennzahlen, Messreihen und Verdikte der Demo „Meinungsbildung“. Verdikte sind reine Funktionen (Art, Text); jede Zahl im Text kommt aus der Rechnung.

Modell: DeGroot (x ← W x mit W = s I + (1 − s) D⁻¹ A) und, bei Empfänglichkeit unter 100 %, Friedkin–Johnsen (x ← Λ W x + (I − Λ) x(0)). Ausgewertet wird die größte zusammenhängende Gruppe;
isolierte Betriebe bleiben bei ihrer Meinung und sind nicht Teil der Anzeige. Jede Messreihe mittelt (Median) über feste Netze (`LIVE_SEEDS` in der App, `SWEEP_SEEDS` für die README-Zahlen)."""

import math
import os
from dataclasses import dataclass

# BLAS-Threads: siehe mbd_algorithm.py (muss vor dem ersten Import von numpy stehen).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

import mbd_algorithm as A
import mbd_constants as C
from mbd_scenario import Instance, make_instance

NAN = float("nan")


def de(x, digits=1) -> str:
    """Dezimalzahl mit Komma (deutsche Texte); eine nicht definierte Zahl erscheint als Gedankenstrich."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x, digits=0) -> str:
    return de(100.0 * x, digits)


def tol_text(tol) -> str:
    """Abbruchschwelle als Zehnerpotenz, zum Beispiel 10⁻⁶."""
    return "10⁻" + "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in str(int(round(-math.log10(tol)))))


def _rng(*parts):
    return np.random.default_rng([int(p) for p in parts])


def _median(values):
    return float(np.median(values)) if len(values) else NAN


# --- Netz und Lauf -------------------------------------------------------------------------------------------------------------------------------------


def matrix_to_adj(a):
    """Nachbarlisten aus einer 0/1-Nachbarschaftsmatrix."""
    return tuple(tuple(np.flatnonzero(row).tolist()) for row in a)


def component_of_instance(inst: Instance):
    """(Knoten der größten Zusammenhangskomponente, Nachbarschaftsmatrix des Teilnetzes, Gruppe je Knoten). Gruppe: der Verbund bei Verbünden, sonst sechs gleich große Nummernblöcke."""
    nodes = A.largest_component(inst.adj)
    a = A.induced_matrix(inst.adj, nodes)
    if inst.groups is not None:
        groups = np.array([inst.groups[v] for v in nodes])
    else:
        groups = np.arange(len(nodes)) * C.SPLIT_GROUPS // len(nodes)
    return nodes, a, groups


def component_of_matrix(a):
    """Wie `component_of_instance` für eine Nachbarschaftsmatrix (ohne Gruppen). Rückgabe (Knoten, Teilmatrix)."""
    adj = matrix_to_adj(a)
    nodes = A.largest_component(adj)
    return nodes, A.induced_matrix(adj, nodes)


def run_model(w, x0, a_val, tol=C.TOL, max_steps=None, periodic=None, frames_wanted=None):
    """Ein Lauf: DeGroot bei a ≥ 1, sonst Friedkin–Johnsen. Wenn W den Eigenwert −1 hat (`periodic`), gilt die Obergrenze `PERIODIC_STEPS`."""
    if max_steps is None:
        max_steps = C.PERIODIC_STEPS if (periodic if periodic is not None else A.is_periodic(w)) else C.MAX_STEPS
    if a_val >= 1.0:
        return A.degroot_run(w, x0, tol, max_steps, frames_wanted)
    return A.fj_run(w, a_val, x0, tol, max_steps, frames_wanted)


# --- Hauptanalyse ----------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    netztyp: str = C.T_RANDOM
    n: int = C.DEFAULT_N
    c: int = C.DEFAULT_C
    seed: int = C.DEFAULT_SEED
    s: int = C.DEFAULT_S           # Selbstgewicht in Prozent
    a: int = C.DEFAULT_A           # Empfänglichkeit in Prozent (100 = reines DeGroot)
    mode: str = C.DEFAULT_MODE


@dataclass(frozen=True, eq=False)
class Analysis:
    inst: Instance
    settings: Settings
    nodes: tuple            # Betriebe der ausgewerteten Gruppe (Nummern im Gesamtnetz)
    pi: object              # Gewichte π = d / Σ d je Betrieb der Gruppe
    x0: object              # Anfangsmeinungen der Gruppe
    run: dict               # Ergebnis von `degroot_run` bzw. `fj_run`
    fj: bool                # Friedkin–Johnsen (Empfänglichkeit unter 100 %)
    periodic: bool          # W hat den Eigenwert −1
    lam2: float             # zweitgrößter Eigenwertbetrag von W
    consensus: float        # π · x(0): Konsens der reinen Mittelung
    mean0: float            # einfacher Mittelwert der Anfangsmeinungen
    n_eff: float            # 1 / Σ π²
    hub_share: float        # Gewicht der HUBS_K Betriebe mit den meisten Partnern
    frames: dict            # Schritt -> Meinungen

    @property
    def m(self) -> int:
        return len(self.nodes)

    @property
    def final(self):
        """Endzustand: das Gleichgewicht (Friedkin–Johnsen) bzw. der Zustand am Ende der Iteration (DeGroot)."""
        return self.run["x_star"] if self.fj else self.run["x"]

    @property
    def steps(self) -> int:
        return self.run["steps"]

    @property
    def converged(self) -> bool:
        return self.run["converged"]

    @property
    def spread_ratio(self) -> float:
        """Streuung des Endzustands durch die Streuung der Anfangsmeinungen (0 = Konsens, 1 = nichts hat sich verändert)."""
        s0 = float(np.std(self.x0))
        return float(np.std(self.final)) / s0 if s0 > 0 else NAN


def hub_order(a):
    """Betriebe nach Partnerzahl absteigend; bei Gleichstand die kleinere Nummer."""
    return np.argsort(-a.sum(axis=1), kind="stable")


def periphery_order(a):
    """Betriebe nach Partnerzahl aufsteigend; bei Gleichstand die kleinere Nummer."""
    return np.argsort(a.sum(axis=1), kind="stable")


def analyse(settings: Settings) -> Analysis:
    inst = make_instance(settings.netztyp, settings.n, settings.c, settings.seed, layout=True)
    nodes, a, groups = component_of_instance(inst)
    s, a_val = settings.s / 100.0, settings.a / 100.0
    w = A.transition_matrix(a, s)
    pi = A.stationary_weights(a)
    x0 = A.initial_opinions(settings.mode, a, groups, _rng(settings.seed, 2), C.HUBS_K)
    lam2, lam_min = A.spectrum_summary(w)
    periodic = lam_min < -1.0 + 1e-9
    first = run_model(w, x0, a_val, periodic=periodic)
    wanted = A.frame_schedule(first["steps"], C.FRAMES)
    run = run_model(w, x0, a_val, periodic=periodic, frames_wanted=wanted)
    k = min(C.HUBS_K, len(nodes))
    return Analysis(inst, settings, tuple(nodes), pi, x0, run, a_val < 1.0, periodic, lam2, A.consensus_value(pi, x0), float(np.mean(x0)), A.effective_sample_size(pi),
                    A.influence(pi, hub_order(a)[:k]), run["frames"])


def verdict_main(an: Analysis) -> tuple[str, str]:
    """Verdikt dieses Laufs: wohin laufen die Meinungen, wie schnell, und wer bestimmt das Ergebnis?"""
    s0 = float(np.std(an.x0))
    k = min(C.HUBS_K, an.m)
    uneven = an.hub_share >= 1.5 * k / an.m
    who = (("Das Gewicht im Konsens ist ungleich verteilt: " if uneven else "Das Gewicht im Konsens ist hier fast gleich verteilt: ")
           + f"die {k} Betriebe mit den meisten Partnern haben zusammen {pct(an.hub_share, 1)} % (bei gleichem Gewicht wären es {pct(k / an.m, 1)} %); "
           f"das Netz zählt wie {de(an.n_eff, 0)} unabhängige Köpfe von {an.m}.")
    if s0 == 0:
        return "info", f"Alle {an.m} Betriebe beginnen mit derselben Meinung ({de(float(an.x0[0]), 2)}): es gibt nichts abzugleichen."
    if an.fj:
        lo, hi = float(np.min(an.final)), float(np.max(an.final))
        head = (f"Nach {an.steps} Schritten hat sich das Netz auf das Gleichgewicht eingependelt" if an.converged
                else f"Nach {an.steps} Schritten (Obergrenze) ist das Gleichgewicht noch nicht auf {tol_text(C.TOL)} genau erreicht; gezeigt ist das exakt berechnete Gleichgewicht")
        text = (f"{head} (Empfänglichkeit {an.settings.a} %): die Meinungen bleiben zwischen {de(lo, 2)} und {de(hi, 2)}, {pct(an.spread_ratio, 1)} % der Anfangsstreuung bleiben erhalten. "
                f"Reine Mittelung käme auf den Konsens {de(an.consensus, 3)} (einfacher Mittelwert der Anfangsmeinungen {de(an.mean0, 3)}). ")
        return ("info" if an.converged else "warning"), text + who
    if an.periodic and not an.converged:
        return "warning", (f"Die Meinungen pendeln: ohne Gewicht auf die eigene Meinung hat die Mittelung auf diesem Netz den Eigenwert −1 (das Netz ist bipartit), die Spanne bleibt nach {an.steps} Schritten bei {de(an.run['spans'][-1], 3)}. "
                           f"Schon ein kleines Selbstgewicht hebt den Eigenwert −1 auf (Experiment 2 zeigt das am Gitter); dann liefe das Netz gegen den Konsens {de(an.consensus, 3)}.")
    if not an.converged:
        return "warning", (f"Nach {an.steps} Schritten (Obergrenze) liegt die Spanne der Meinungen noch bei {de(an.run['spans'][-1], 4)}; der Konsens wäre {de(an.consensus, 3)}. "
                           f"|λ₂| = {de(an.lam2, 3)}: so langsam mischt dieses Netz.")
    kind = "success" if uneven else "info"
    return kind, (f"Nach {an.steps} Schritten sind sich alle Betriebe einig (Spanne unter {tol_text(C.TOL)}): auf {de(an.final.mean(), 3)}. "
                  + f"Das ist der mit dem Grad gewichtete Mittelwert der Anfangsmeinungen ({de(an.consensus, 3)}), nicht der einfache ({de(an.mean0, 3)}). " + who
                  + f" Die Geschwindigkeit hängt an |λ₂| = {de(an.lam2, 3)}.")


# --- E1: Wer bestimmt das Ergebnis? ------------------------------------------------------------------------------------------------------------------


def influence_rows(n, c, types=C.ALL_TYPES, seeds=C.LIVE_SEEDS, k=C.HUBS_K) -> list[dict]:
    """Je Netztyp: Konsenswert, wenn nur k Betriebe die Meinung 1 haben (alle anderen 0): die k mit den meisten Partnern, k zufällige (exakter Erwartungswert k / m), die k mit den wenigsten
    (Median über die festen Netze), dazu das größte Einzelgewicht π_max."""
    rows = []
    for typ in types:
        hubs, rand, per, pmax = [], [], [], []
        for seed in seeds:
            _, a, _ = component_of_instance(make_instance(typ, n, c, seed, layout=False))
            pi = A.stationary_weights(a)
            kk = min(k, len(a))
            hubs.append(A.influence(pi, hub_order(a)[:kk]))
            per.append(A.influence(pi, periphery_order(a)[:kk]))
            rand.append(kk / len(a))                                                      # E[Σ π über kk zufällige Betriebe] = kk / m, weil der Mittelwert von π gleich 1 / m ist
            pmax.append(float(pi.max()))
        rows.append({"typ": typ, "hubs": _median(hubs), "random": _median(rand), "periphery": _median(per), "pi_max": _median(pmax)})
    return rows


def verdict_influence(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E1: wo bestimmen wenige Hubs das Ergebnis, wo nicht?"""
    if not rows:
        return "warning", "Keine Messung."
    strong = [r["typ"] for r in rows if r["hubs"] >= 1.5 * r["random"]]
    weak = [r["typ"] for r in rows if r["hubs"] < 1.5 * r["random"]]
    top = max(rows, key=lambda r: r["hubs"] / r["random"])
    text = (f"Haben nur die {C.HUBS_K} Betriebe mit den meisten Partnern die Meinung 1, landet der Konsens in „{top['typ']}“ bei {pct(top['hubs'], 1)} % (bei {C.HUBS_K} zufälligen Betrieben im Mittel {pct(top['random'], 1)} %, "
            f"bei den {C.HUBS_K} mit den wenigsten Partnern {pct(top['periphery'], 1)} %). ")
    text += ("Mindestens das 1,5-Fache der zufälligen Wahl erreichen die Hubs in: " + ", ".join(f"„{t}“" for t in strong) + ". ") if strong else "In keinem Netztyp erreichen die Hubs das 1,5-Fache der zufälligen Wahl. "
    text += ("Darunter bleiben sie in: " + ", ".join(f"„{t}“" for t in weak) + ".") if weak else "Darunter bleibt kein Netztyp."
    return "info", text


# --- E2: Geschwindigkeit -------------------------------------------------------------------------------------------------------------------------------


def _speed_cell(a, s, tol, rng):
    w = A.transition_matrix(a, s)
    lam2, lam_min = A.spectrum_summary(w)
    periodic = lam_min < -1.0 + 1e-9
    out = run_model(w, rng.random(len(a)), 1.0, tol=tol, periodic=periodic)
    return lam2, (out["steps"] if out["converged"] else NAN), periodic


def speed_rows(n, c, types=C.ALL_TYPES, ss=C.SPEED_SS, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Je Netztyp und Selbstgewicht: |λ₂| und Schritte bis die Spanne unter `TOL` fällt (Median über die festen Netze; Schritte NaN, wenn eines der Netze nicht konvergiert), dazu die Zahl pendelnder Netze."""
    rows = []
    for typ in types:
        nets = [component_of_instance(make_instance(typ, n, c, seed, layout=False))[1] for seed in seeds]
        for s in ss:
            cells = [_speed_cell(a, s, C.TOL, _rng(seed, 3)) for seed, a in zip(seeds, nets)]
            steps = [x[1] for x in cells]
            rows.append({"typ": typ, "s": s, "lam2": _median([x[0] for x in cells]), "steps": NAN if any(math.isnan(v) for v in steps) else _median(steps), "periodic": sum(1 for x in cells if x[2]), "nets": len(cells)})
    return rows


def speed_cell(rows, typ, s):
    return next(r for r in rows if r["typ"] == typ and r["s"] == s)


def inside_rows(insides=C.COMM_INSIDES, n=C.COMM_N, c=C.COMM_C, s=0.5, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Verbünde mit einstellbarem Anteil der Kanten innen: |λ₂| und Schritte bis die Spanne unter `COMM_TOL` fällt (Median über feste Netze)."""
    rows = []
    for inside in insides:
        cells = []
        for seed in seeds:
            _, a = component_of_matrix(sbm_net(n, c, inside, seed)[0])
            cells.append(_speed_cell(a, s, C.COMM_TOL, _rng(seed, 3)))
        rows.append({"inside": inside, "s": s, "lam2": _median([x[0] for x in cells]), "steps": _median([x[1] for x in cells]), "nets": len(cells)})
    return rows


def sbm_net(n, c, inside, seed):
    """Blockmodell mit sechs Verbünden (Zufall nur über den Seed). Rückgabe (Nachbarschaftsmatrix, Gruppe je Knoten)."""
    return A.sbm_matrix(n, C.BLOCKS, c, inside, 7 * int(seed) + int(round(100 * inside)))


def bipartite_rows(n=C.DEFAULT_N) -> list[dict]:
    """Das Gitter: mit vier Partnern bipartit (Eigenwert −1 bei Selbstgewicht 0), mit acht nicht. Je Fall der kleinste Eigenwert, ob die Mittelung konvergiert, und die Spanne am Ende."""
    cases = ((4, 0.0), (4, 0.1), (8, 0.0))
    rows = []
    for c, s in cases:
        _, a, _ = component_of_instance(make_instance(C.T_GRID, n, c, 0, layout=False))
        w = A.transition_matrix(a, s)
        lam2, lam_min = A.spectrum_summary(w)
        out = run_model(w, _rng(0, 3).random(len(a)), 1.0, periodic=lam_min < -1.0 + 1e-9)
        rows.append({"c": c, "s": s, "lam_min": lam_min, "converged": out["converged"], "steps": out["steps"], "span": out["spans"][-1], "bipartite": lam_min < -1.0 + 1e-9})
    return rows


def verdict_speed(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E2: wer mischt schnell, wer langsam?"""
    s = C.SPEED_SS[-1]
    mine = [r for r in rows if r["s"] == s]
    if not mine:
        return "warning", "Keine Messung."
    ok = [r for r in mine if not math.isnan(r["steps"])]
    if not ok:
        return "warning", "In keinem Netztyp ist die Mittelung innerhalb der Obergrenze fertig."
    ranked = sorted(ok, key=lambda r: r["steps"])
    fast, slow = ranked[0], ranked[-1]
    tail = " Je näher |λ₂| an 1 liegt, desto länger" + (f"; am langsamsten sind „{slow['typ']}“ und „{ranked[-2]['typ']}“." if len(ranked) >= 2 else ".")
    return "info", (f"Bei Selbstgewicht {pct(s)} % braucht die Mittelung in „{fast['typ']}“ {de(fast['steps'], 0)} Schritte (|λ₂| = {de(fast['lam2'], 3)}), in „{slow['typ']}“ {de(slow['steps'], 0)} (|λ₂| = {de(slow['lam2'], 3)}): "
                    f"{de(slow['steps'] / fast['steps'], 0)}-mal so lang." + tail)


def verdict_inside(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E2 (Verbünde): wie bremst ein dichter Verbund die Einigung?"""
    if len(rows) < 2:
        return "warning", "Zu wenige Messpunkte."
    a, b = rows[0], rows[-1]
    return "info", (f"Liegen {pct(a['inside'])} % der Kanten innerhalb der Verbünde, ist die Spanne nach {de(a['steps'], 0)} Schritten unter {tol_text(C.COMM_TOL)}"
                    + f" (|λ₂| = {de(a['lam2'], 3)}); bei {pct(b['inside'])} % sind es {de(b['steps'], 0)} Schritte (|λ₂| = {de(b['lam2'], 3)}). Wenige Querkanten sind der Engpass.")


def verdict_bipartite(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E2 (Sonderfall): das bipartite Gitter pendelt, ein wenig Selbstgewicht oder Diagonalen heben das auf."""
    by = {(r["c"], r["s"]): r for r in rows}
    bad, fixed, diag = by.get((4, 0.0)), by.get((4, 0.1)), by.get((8, 0.0))
    if not (bad and fixed and diag):
        return "warning", "Keine Messung."
    return "info", (f"Gitter mit 4 Partnern ohne Selbstgewicht: kleinster Eigenwert {de(bad['lam_min'], 3)}, "
                    + ("die Meinungen pendeln" if not bad["converged"] else "die Meinungen konvergieren") + f" (Spanne nach {bad['steps']} Schritten {de(bad['span'], 3)}). Mit 10 % Selbstgewicht: kleinster Eigenwert {de(fixed['lam_min'], 3)}, "
                    + (f"Konvergenz nach {fixed['steps']} Schritten" if fixed["converged"] else "keine Konvergenz") + f". Mit 8 Partnern (Diagonalen) ohne Selbstgewicht: kleinster Eigenwert {de(diag['lam_min'], 3)}, "
                    + (f"Konvergenz nach {diag['steps']} Schritten." if diag["converged"] else "keine Konvergenz."))


# --- E3: Weisheit der Menge ---------------------------------------------------------------------------------------------------------------------------


def wisdom_rows(n, c, types=C.ALL_TYPES, seeds=C.LIVE_SEEDS, runs=C.WISDOM_RUNS) -> list[dict]:
    """Je Netztyp: effektive Zahl unabhängiger Köpfe n_eff = 1 / Σ π², Faktor der Fehlervarianz gegenüber dem einfachen Mittelwert (m Σ π²), und derselbe Faktor aus einer Simulation:
    Anfangsmeinung = Wahrheit + unabhängiges Rauschen, `runs` Wiederholungen je Netz (Median über die festen Netze)."""
    rows = []
    for typ in types:
        neff, factor, sim, size, pmax = [], [], [], [], []
        for seed in seeds:
            _, a, _ = component_of_instance(make_instance(typ, n, c, seed, layout=False))
            pi = A.stationary_weights(a)
            m = len(a)
            draws = _rng(seed, 4).standard_normal((runs, m))
            neff.append(A.effective_sample_size(pi))
            factor.append(m * A.consensus_error_variance(pi, 1.0))
            sim.append(m * float(np.var(draws @ pi)))                                      # Wahrheit 0 und σ = 1; Vergleichswert des einfachen Mittelwerts: 1 / m
            size.append(m)
            pmax.append(float(pi.max()))
        rows.append({"typ": typ, "m": _median(size), "n_eff": _median(neff), "factor": _median(factor), "sim_factor": _median(sim), "pi_max": _median(pmax), "runs": runs})
    return rows


def verdict_wisdom(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E3: wo verschwenden Hubs Genauigkeit?"""
    if not rows:
        return "warning", "Keine Messung."
    worst = max(rows, key=lambda r: r["factor"])
    best = min(rows, key=lambda r: r["factor"])
    return "info", (f"Die Fehlervarianz des Konsenses ist das {de(worst['factor'], 2)}-Fache des einfachen Mittelwerts in „{worst['typ']}“ (das Netz zählt wie {de(worst['n_eff'], 0)} von {de(worst['m'], 0)} unabhängigen Köpfen; "
                    f"Simulation: {de(worst['sim_factor'], 2)}-fach) und nur das {de(best['factor'], 2)}-Fache in „{best['typ']}“ ({de(best['n_eff'], 0)} von {de(best['m'], 0)}). Je ungleicher die Gewichte π, desto weniger Köpfe zählen.")


# --- E4: Sturheit ---------------------------------------------------------------------------------------------------------------------------------------------


def stubborn_rows(n, c, types=C.ALL_TYPES, alphas=C.STUBBORN_AS, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Je Netztyp und Empfänglichkeit a (Selbstgewicht 0, Anfangsmeinung zufällig): Streuung des Friedkin–Johnsen-Gleichgewichts durch die Streuung der Anfangsmeinung (Median über die festen Netze)."""
    rows = []
    for typ in types:
        nets = [component_of_instance(make_instance(typ, n, c, seed, layout=False))[1] for seed in seeds]
        for al in alphas:
            vals = []
            for seed, a in zip(seeds, nets):
                x0 = _rng(seed, 5).random(len(a))
                star = A.fj_equilibrium(A.transition_matrix(a, 0.0), al, x0)
                vals.append(float(np.std(star) / np.std(x0)))
            rows.append({"typ": typ, "a": al, "ratio": _median(vals)})
    return rows


def stubborn_cell(rows, typ, a):
    return next(r for r in rows if r["typ"] == typ and r["a"] == a)


def verdict_stubborn(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E4: wie viel Vielfalt bleibt bei wenig Sturheit?"""
    if not rows:
        return "warning", "Keine Messung."
    ref = 0.95 if any(r["a"] == 0.95 for r in rows) else max(r["a"] for r in rows)
    mine = sorted((r for r in rows if r["a"] == ref), key=lambda r: r["ratio"])
    lo, hi = mine[0], mine[-1]
    return "info", (f"Bei Empfänglichkeit {pct(ref)} % bleiben in „{lo['typ']}“ {pct(lo['ratio'], 1)} % der Anfangsstreuung erhalten, in „{hi['typ']}“ {pct(hi['ratio'], 1)} %. "
                    "Je kleiner die Empfänglichkeit, desto mehr Vielfalt bleibt.")


# --- E5: Spaltung in Verbünden ------------------------------------------------------------------------------------------------------------------------


def split_rows(insides=C.COMM_INSIDES, alphas=C.COMM_AS, n=C.COMM_N, c=C.COMM_C, seeds=C.LIVE_SEEDS) -> list[dict]:
    """Verbünde (sechs, Anteil `inside` der Kanten innen), drei Verbünde starten bei +1, drei bei −1: Streuung des Gleichgewichts durch die Anfangsstreuung (Median über feste Netze)."""
    rows = []
    for inside in insides:
        nets = []
        for seed in seeds:
            a_full, lab = sbm_net(n, c, inside, seed)
            nodes, a = component_of_matrix(a_full)
            nets.append((a, np.array([lab[v] for v in nodes])))
        for al in alphas:
            vals = []
            for a, groups in nets:
                x0 = np.where(groups < (groups.max() + 1) / 2.0, 1.0, -1.0)
                star = A.fj_equilibrium(A.transition_matrix(a, 0.0), al, x0)
                vals.append(float(np.std(star) / np.std(x0)))
            rows.append({"inside": inside, "a": al, "ratio": _median(vals)})
    return rows


def split_cell(rows, inside, a):
    return next(r for r in rows if r["inside"] == inside and r["a"] == a)


def verdict_split(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E5: halten dichte Verbünde die Spaltung?"""
    if not rows:
        return "warning", "Keine Messung."
    al = 0.95 if any(r["a"] == 0.95 for r in rows) else max(r["a"] for r in rows)
    mine = sorted((r for r in rows if r["a"] == al), key=lambda r: r["inside"])
    lo, hi = mine[0], mine[-1]
    return "info", (f"Bei Empfänglichkeit {pct(al)} % bleiben von der Anfangsspaltung {pct(lo['ratio'], 1)} % erhalten, wenn {pct(lo['inside'])} % der Kanten innerhalb der Verbünde liegen, "
                    f"und {pct(hi['ratio'], 1)} %, wenn es {pct(hi['inside'])} % sind. Wenige Querkanten und etwas Sturheit halten die Spaltung entlang der Verbünde.")

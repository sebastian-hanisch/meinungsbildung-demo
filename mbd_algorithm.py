"""Kern: Meinungsbildung nach DeGroot und Friedkin–Johnsen auf ungerichteten Netzen (numpy, dichte Matrizen für n ≤ 250).

DeGroot: jeder Betrieb ersetzt seine Meinung durch den gewichteten Mittelwert seiner Partner und seiner selbst, x(t + 1) = W x(t) mit W = s I + (1 − s) D⁻¹ A (Selbstgewicht s, Rest gleichmäßig auf die
Partner). W ist zeilenstochastisch; bei zusammenhängendem, nicht bipartitem Netz (oder s > 0) konvergiert x(t) gegen den Konsens π · x(0) mit dem linken Perron-Vektor π = d / Σ d.
Friedkin–Johnsen: x(t + 1) = Λ W x(t) + (I − Λ) x(0) mit der Empfänglichkeit Λ = diag(a); das Gleichgewicht ist x* = (I − Λ W)⁻¹ (I − Λ) x(0).

Netze sind Nachbarlisten `adj` (Tupel von sortierten Tupeln, Knoten 0 bis n - 1, ungerichtet). Zufall nur über `numpy.random.default_rng(Seed)`."""

import os

# Dichte Matrizen bis 250 × 250: ein BLAS-Thread genügt. Mehrere Threads machen die kleine Lösung auf ausgelasteten Rechnern hundertfach langsamer (gemessen: 300 ms gegen 1 ms).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

# --- Netze ----------------------------------------------------------------------------------------------------------------------------------------


def adjacency_matrix(adj):
    n = len(adj)
    a = np.zeros((n, n))
    for i, nb in enumerate(adj):
        for j in nb:
            a[i, j] = 1.0
    return a


def largest_component(adj):
    """Knoten der größten Zusammenhangskomponente, aufsteigend (Gleichstand: die mit dem kleinsten Knoten)."""
    n = len(adj)
    seen = [False] * n
    best = []
    for s in range(n):
        if seen[s]:
            continue
        comp = [s]
        seen[s] = True
        for x in comp:
            for w in adj[x]:
                if not seen[w]:
                    seen[w] = True
                    comp.append(w)
        if len(comp) > len(best):
            best = comp
    return sorted(best)


def induced_matrix(adj, nodes):
    """Nachbarschaftsmatrix des Teilnetzes auf `nodes` (neue Nummern = Position in `nodes`)."""
    idx = {v: i for i, v in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)))
    for v in nodes:
        for w in adj[v]:
            if w in idx:
                a[idx[v], idx[w]] = 1.0
    return a


def sbm_matrix(n, q, c, inside, seed):
    """Blockmodell: q gleich große Gruppen (Knoten i gehört zu Gruppe i · q // n), mittlerer Grad c, Anteil `inside` der Kanten innerhalb der Gruppen. Rückgabe (Nachbarschaftsmatrix, Gruppe je Knoten)."""
    lab = np.arange(n) * q // n
    same = lab[:, None] == lab[None, :]
    n_in = (same.sum() - n) / 2.0
    n_out = n * (n - 1) / 2.0 - n_in
    m = c * n / 2.0
    p_in = min(1.0, inside * m / n_in) if n_in > 0 else 0.0
    p_out = min(1.0, (1.0 - inside) * m / n_out) if n_out > 0 else 0.0
    prob = np.where(same, p_in, p_out)
    u = np.random.default_rng(int(seed)).random((n, n))
    upper = np.triu(u < prob, 1).astype(float)
    return upper + upper.T, lab


# --- DeGroot --------------------------------------------------------------------------------------------------------------------------------------


def transition_matrix(a, s=0.0):
    """W = s I + (1 − s) D⁻¹ A. Jeder Betrieb braucht mindestens einen Partner (sonst Fehler), 0 ≤ s ≤ 1."""
    if not 0.0 <= s <= 1.0:
        raise ValueError("Selbstgewicht muss in [0, 1] liegen")
    d = a.sum(axis=1)
    if (d <= 0).any():
        raise ValueError("Betrieb ohne Partner")
    return s * np.eye(len(a)) + (1.0 - s) * a / d[:, None]


def stationary_weights(a):
    """Linker Perron-Vektor von D⁻¹ A (und von W für jedes s): π = d / Σ d."""
    d = a.sum(axis=1)
    return d / d.sum()


def consensus_value(pi, x0):
    return float(np.dot(pi, x0))


def span(x):
    return float(np.max(x) - np.min(x))


def frame_schedule(max_steps, frames):
    """Schritte, deren Meinungen aufgehoben werden: 0, 1, 2, … und danach geometrisch bis `max_steps` (höchstens etwa `frames` Stück)."""
    steps = {0, 1}
    steps |= {int(round(v)) for v in np.geomspace(1, max(1, max_steps), frames)}
    return sorted(s for s in steps if 0 <= s <= max_steps)


def degroot_run(w, x0, tol, max_steps, frames=None):
    """Iteration x ← W x, bis die Spanne unter `tol` fällt oder `max_steps` erreicht sind. Rückgabe Wörterbuch: steps (Zahl der Schritte), converged, spans (Spanne nach 0 bis steps Schritten),
    frames (Schritt -> Meinungen, nur für die Schritte aus `frames` und der letzte), x (Endzustand)."""
    x = np.array(x0, dtype=float)
    wanted = set(frames or ())
    spans = [span(x)]
    kept = {0: x.copy()} if 0 in wanted else {}
    steps = 0
    while spans[-1] >= tol and steps < max_steps:
        x = w @ x
        steps += 1
        spans.append(span(x))
        if steps in wanted:
            kept[steps] = x.copy()
    kept[steps] = x.copy()
    return {"steps": steps, "converged": spans[-1] < tol, "spans": spans, "frames": kept, "x": x}


def second_eigenvalue_modulus(w):
    """Zweitgrößter Eigenwertbetrag von W (dichte Rechnung)."""
    ev = np.sort(np.abs(np.linalg.eigvals(w)))[::-1]
    return float(ev[1]) if len(ev) > 1 else 0.0


def min_eigenvalue(w):
    """Kleinster (reeller) Eigenwert der symmetrisierbaren Matrix W; −1 heißt: die Meinungen pendeln (bipartites Netz ohne Selbstgewicht)."""
    ev = np.linalg.eigvals(w)
    return float(np.min(ev.real))


def spectrum_summary(w):
    """(zweitgrößter Eigenwertbetrag, kleinster reeller Eigenwert) aus einer einzigen Eigenwertrechnung."""
    ev = np.linalg.eigvals(w)
    mods = np.sort(np.abs(ev))[::-1]
    return (float(mods[1]) if len(mods) > 1 else 0.0), float(np.min(ev.real))


def is_periodic(w, tol=1e-9):
    """W hat den Eigenwert −1: die reine Mittelung konvergiert nicht."""
    return min_eigenvalue(w) < -1.0 + tol


# --- Friedkin–Johnsen -------------------------------------------------------------------------------------------------------------------------


def _alpha(a_val, n):
    return np.full(n, float(a_val)) if np.isscalar(a_val) else np.asarray(a_val, dtype=float)


def fj_equilibrium(w, a_val, x0):
    """x* = (I − Λ W)⁻¹ (I − Λ) x(0) mit Λ = diag(a); a ist eine Zahl in [0, 1) oder ein Vektor. Für a = 1 (alle) gibt es kein eindeutiges Gleichgewicht."""
    n = len(w)
    alpha = _alpha(a_val, n)
    if (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("Empfänglichkeit muss in [0, 1] liegen")
    if (alpha >= 1).all():
        raise ValueError("bei Empfänglichkeit 1 für alle ist das Gleichgewicht nicht eindeutig (reines DeGroot)")
    lam = np.diag(alpha)
    return np.linalg.solve(np.eye(n) - lam @ w, (np.eye(n) - lam) @ np.asarray(x0, dtype=float))


def fj_iterate(w, a_val, x0, steps):
    """Orakel: x ← Λ W x + (I − Λ) x(0), `steps` Schritte ab x(0)."""
    alpha = _alpha(a_val, len(w))
    x0 = np.asarray(x0, dtype=float)
    x = x0.copy()
    for _ in range(steps):
        x = alpha * (w @ x) + (1.0 - alpha) * x0
    return x


def fj_run(w, a_val, x0, tol, max_steps, frames=None):
    """Iteration bis zum Gleichgewicht (größte Abweichung unter `tol`) oder `max_steps`. Rückgabe wie `degroot_run` und das Gleichgewicht `x_star`; `spans` ist die Spanne, `gaps` die Abweichung vom Gleichgewicht."""
    x0 = np.asarray(x0, dtype=float)
    star = fj_equilibrium(w, a_val, x0)
    alpha = _alpha(a_val, len(w))
    x = x0.copy()
    wanted = set(frames or ())
    spans = [span(x)]
    gaps = [float(np.max(np.abs(x - star)))]
    kept = {0: x.copy()} if 0 in wanted else {}
    steps = 0
    while gaps[-1] >= tol and steps < max_steps:
        x = alpha * (w @ x) + (1.0 - alpha) * x0
        steps += 1
        spans.append(span(x))
        gaps.append(float(np.max(np.abs(x - star))))
        if steps in wanted:
            kept[steps] = x.copy()
    kept[steps] = x.copy()
    return {"steps": steps, "converged": gaps[-1] < tol, "spans": spans, "gaps": gaps, "frames": kept, "x": x, "x_star": star}


# --- Weisheit der Menge und Einfluss ------------------------------------------------------------------------------------------------------------


def effective_sample_size(pi):
    """1 / Σ π²: so viele unabhängige Köpfe wäre das Netz bei gleichem Gewicht wert."""
    return float(1.0 / np.sum(np.asarray(pi) ** 2))


def consensus_error_variance(pi, sigma=1.0):
    """Varianz des Konsenses, wenn die Anfangsmeinungen wahrer Wert + unabhängiges Rauschen der Varianz σ² sind: σ² Σ π²."""
    return float(sigma**2 * np.sum(np.asarray(pi) ** 2))


def influence(pi, nodes):
    """Konsenswert, wenn nur die Betriebe `nodes` die Meinung 1 haben und alle anderen 0: die Summe ihrer Gewichte."""
    return float(np.sum(np.asarray(pi)[list(nodes)]))


def initial_opinions(mode, a, groups, rng, hubs=10):
    """Anfangsmeinungen: „Zufällig“ (gleichverteilt in [0, 1]), „Spaltung nach Gruppen“ (Gruppen 0 bis 2 bei 1, die übrigen bei 0), „Die Hubs dafür“ (die `hubs` Betriebe mit größtem Grad bei 1, Gleichstand:
    kleinere Nummer), „Ein einzelner Betrieb“ (ein zufälliger bei 1)."""
    n = len(a)
    if mode == "Zufällig":
        return rng.random(n)
    if mode == "Spaltung nach Gruppen":
        groups = np.asarray(groups)
        return np.where(groups < (groups.max() + 1) / 2.0, 1.0, 0.0)
    if mode == "Die Hubs dafür":
        order = np.argsort(-a.sum(axis=1), kind="stable")
        x = np.zeros(n)
        x[order[: min(hubs, n)]] = 1.0
        return x
    if mode == "Ein einzelner Betrieb":
        x = np.zeros(n)
        x[int(rng.integers(n))] = 1.0
        return x
    raise ValueError(f"unbekannter Modus: {mode}")

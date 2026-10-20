"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import mbd_constants as C
from mbd_evaluation import de, pct

MAX_POINTS = 600          # Verlaufslinien werden auf so viele Punkte ausgedünnt
FLOOR = 1e-12             # Untergrenze der logarithmischen Achse (eine Spanne von genau 0 hat keinen Logarithmus)


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, bottom=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=bottom), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _pct_axis(fig, title):
    fig.update_yaxes(title_text=title, ticksuffix=" %", rangemode="tozero")
    return fig


def _edges(adj, keep):
    """Kanten zwischen Betrieben aus `keep` als Linienzug mit Lücken (Plotly-Muster)."""
    return [(a, b) for a in keep for b in adj[a] if a < b and b in keep]


def build_network(an, step):
    """Netz der ausgewerteten Gruppe; Farbe = Meinung im gewählten Schritt (feste Skala 0 bis 1), Größe = Gewicht π im Konsens."""
    inst = an.inst
    keep = set(an.nodes)
    pos = inst.pos
    xs, ys = [], []
    for a, b in _edges(inst.adj, keep):
        xs += [pos[a][0], pos[b][0], None]
        ys += [pos[a][1], pos[b][1], None]
    x = an.frames[step]
    pmax = float(np.max(an.pi))
    size = 0.75 if an.m > 200 else 1.0
    sizes = [size * (4.0 + 14.0 * math.sqrt(float(p) / pmax)) for p in an.pi]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_EDGE, width=0.7), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[v][0] for v in an.nodes], y=[pos[v][1] for v in an.nodes], mode="markers", showlegend=False,
        marker=dict(size=sizes, color=[float(v) for v in x], colorscale="Viridis", cmin=0.0, cmax=1.0, line=dict(color="rgba(40,40,40,0.45)", width=0.4),
                    colorbar=dict(title=dict(text="Meinung"), thickness=12, len=0.8)),
        hovertext=[f"Betrieb {v}: Meinung {de(float(m), 3)}, {len(inst.adj[v])} Partner, Gewicht {pct(float(p), 2)} %" for v, m, p in zip(an.nodes, x, an.pi)], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[-0.04, 1.04], y=[-0.04, 1.04], mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, autorange=True)
    fig.update_yaxes(visible=False, autorange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _thin(values):
    """(Schritte, Werte) höchstens `MAX_POINTS` Punkte, erster und letzter bleiben."""
    n = len(values)
    stride = max(1, -(-n // MAX_POINTS))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx, [max(values[i], FLOOR) for i in idx]


def build_spans(an, step):
    """Spanne der Meinungen (größte minus kleinste) über die Schritte, logarithmisch; bei Friedkin–Johnsen zusätzlich der Abstand zum Gleichgewicht. Senkrechte Linie = gewählter Schritt."""
    fig = go.Figure()
    xs, ys = _thin(an.run["spans"])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Spanne der Meinungen", line=dict(color=C.COLOR_A, width=3)))
    if an.fj:
        gx, gy = _thin(an.run["gaps"])
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", name="Abstand zum Gleichgewicht", line=dict(color=C.COLOR_B, width=3, dash="dash")))
    fig.add_vline(x=step, line=dict(color="rgba(120,120,120,0.9)", width=1.5, dash="dot"))
    fig.update_xaxes(title_text="Schritt")
    fig.update_yaxes(title_text="größte minus kleinste Meinung", type="log", exponentformat="power")
    return _base(fig, height=300, bottom=50)


def build_influence(rows):
    """Konsenswert in %, wenn nur zehn Betriebe die Meinung 1 haben: die Hubs, zufällige Betriebe, die Randbetriebe."""
    fig = go.Figure()
    xs = [r["typ"] for r in rows]
    for label, key, color in (("die 10 mit den meisten Partnern", "hubs", C.COLOR_B), ("10 zufällige", "random", "rgba(120,120,120,0.9)"), ("die 10 mit den wenigsten Partnern", "periphery", C.COLOR_A)):
        ys = [100.0 * r[key] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=ys, name=label, marker_color=color, text=[de(y, 1) for y in ys], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-20)
    _pct_axis(fig, "Konsenswert bei Meinung 1 in zehn Betrieben")
    return _base(fig, height=380, bottom=110)


def build_speed(rows, s):
    """Schritte bis die Spanne unter 10⁻⁶ fällt, je Netztyp bei Selbstgewicht `s` (logarithmisch); ein Netz, das nicht konvergiert (pendelt), bekommt statt des Balkens die Beschriftung „pendelt“."""
    mine = [r for r in rows if r["s"] == s]
    ok = [r for r in mine if not math.isnan(r["steps"])]
    fig = go.Figure(go.Bar(x=[r["typ"] for r in ok], y=[r["steps"] for r in ok], marker_color=[C.TYPE_COLORS[r["typ"]] for r in ok], text=[de(r["steps"], 0) for r in ok], textposition="outside"))
    for r in mine:
        if math.isnan(r["steps"]):
            fig.add_annotation(x=r["typ"], y=0.04, yref="paper", text="pendelt", showarrow=False, font=dict(color=C.COLOR_B))
    fig.update_xaxes(tickangle=-20, categoryorder="array", categoryarray=[r["typ"] for r in mine])
    fig.update_yaxes(title_text="Schritte", type="log", exponentformat="power")
    return _base(fig, height=340, bottom=110)


def build_inside(rows):
    """Verbünde: Schritte bis die Spanne unter 10⁻³ fällt, über den Anteil der Kanten innerhalb der Verbünde (logarithmisch)."""
    xs = [100.0 * r["inside"] for r in rows]
    fig = go.Figure(go.Scatter(x=xs, y=[r["steps"] for r in rows], mode="lines+markers+text", line=dict(color=C.COLOR_C, width=3), text=[de(r["steps"], 0) for r in rows], textposition="top center"))
    fig.update_xaxes(title_text="Kanten innerhalb der Verbünde (%)", tickvals=xs, ticktext=[pct(r["inside"]) for r in rows])
    fig.update_yaxes(title_text="Schritte", type="log", exponentformat="power")
    return _base(fig, height=320, bottom=50)


def build_wisdom(rows):
    """Faktor der Fehlervarianz gegenüber dem einfachen Mittelwert: Formel m · Σ π² (Balken) und Simulation (Raute)."""
    xs = [r["typ"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["factor"] for r in rows], name="Formel m · Σ π²", marker_color=[C.TYPE_COLORS[r["typ"]] for r in rows], text=[de(r["factor"], 2) for r in rows], textposition="outside"))
    fig.add_trace(go.Scatter(x=xs, y=[r["sim_factor"] for r in rows], mode="markers", name="Simulation", marker=dict(symbol="diamond-open", size=11, color="rgba(30,30,30,0.9)", line=dict(width=2))))
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.9)", dash="dash"))
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(title_text="Fehlervarianz gegenüber dem einfachen Mittelwert (×)", rangemode="tozero")
    return _base(fig, height=380, bottom=110)


def build_stubborn(rows):
    """Streuung des Gleichgewichts in % der Anfangsstreuung über die Empfänglichkeit, je Netztyp."""
    fig = go.Figure()
    for typ in dict.fromkeys(r["typ"] for r in rows):
        mine = [r for r in rows if r["typ"] == typ]
        fig.add_trace(go.Scatter(x=[100.0 * r["a"] for r in mine], y=[100.0 * r["ratio"] for r in mine], mode="lines+markers", name=typ, line=dict(color=C.TYPE_COLORS[typ], width=3)))
    alphas = sorted({r["a"] for r in rows})
    fig.update_xaxes(title_text="Empfänglichkeit (%)", tickvals=[100.0 * a for a in alphas], ticktext=[pct(a) for a in alphas])
    _pct_axis(fig, "übrige Streuung (% der Anfangsstreuung)")
    return _base(fig, height=360, bottom=60)


def build_split(rows):
    """Streuung des Gleichgewichts in % der Anfangsstreuung über den Anteil der Kanten innerhalb der Verbünde, je Empfänglichkeit."""
    fig = go.Figure()
    colors = (C.COLOR_A, C.COLOR_B, C.COLOR_C)
    for color, al in zip(colors, sorted({r["a"] for r in rows})):
        mine = [r for r in rows if r["a"] == al]
        fig.add_trace(go.Scatter(x=[100.0 * r["inside"] for r in mine], y=[100.0 * r["ratio"] for r in mine], mode="lines+markers", name=f"Empfänglichkeit {pct(al)} %", line=dict(color=color, width=3)))
    insides = sorted({r["inside"] for r in rows})
    fig.update_xaxes(title_text="Kanten innerhalb der Verbünde (%)", tickvals=[100.0 * x for x in insides], ticktext=[pct(x) for x in insides])
    _pct_axis(fig, "übrige Spaltung (% der Anfangsstreuung)")
    return _base(fig, height=340, bottom=60)

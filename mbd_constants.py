"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Netztypen -----------------------------------------------------------------------------------------------------------------------------------
T_RANDOM = "Zufallsnetz"
T_POWER = "Potenzgesetz-Netz"
T_BA = "Skalenfreies Netz (BA)"
T_COMMUNITIES = "Verbünde"
T_RING = "Ring mit Abkürzungen"
T_GRID = "Gitter"
ALL_TYPES = (T_RANDOM, T_POWER, T_BA, T_COMMUNITIES, T_RING, T_GRID)
BLOCKS = 6

C_MEANING = {
    T_RANDOM: "mittlerer Grad des Zufallsnetzes",
    T_POWER: "mittlerer Grad (Gradfolge mit Potenzgesetz-Exponent 2,5, Konfigurationsmodell)",
    T_BA: "zwei Kanten je neuem Knoten, also m = c / 2 gerundet (jeder Knoten hat mindestens m Nachbarn)",
    T_COMMUNITIES: "mittlerer Grad; 80 % der Kanten bleiben im eigenen Verbund (6 Verbünde)",
    T_RING: "Nachbarn auf dem Ring (c gerade gerundet), dazu Abkürzungen mit 5 % je Ringkante",
    T_GRID: "Nachbarschaft: unter 6 vier Nachbarn (bipartit!), 6 bis 7 mit einer Diagonale, ab 8 mit beiden Diagonalen",
}

N_MIN, N_MAX, DEFAULT_N = 60, 250, 150
C_MIN, C_MAX, DEFAULT_C = 3, 12, 6
SEED_MAX, DEFAULT_SEED = 999999, 35
S_MIN, S_MAX, DEFAULT_S = 0, 90, 0                 # Selbstgewicht in Prozent
A_MIN, A_MAX, DEFAULT_A = 50, 100, 100             # Empfänglichkeit in Prozent (100 = reines DeGroot)
SWEEP_SEEDS = tuple(range(100000, 100008))         # feste Netze für alle Sweeps: nie ein einzelner Seed als Befund
LIVE_SEEDS = SWEEP_SEEDS[:4]                       # so viele feste Netze je Zelle rechnet die App live (die README-Zahlen nutzen alle acht)

# --- Anfangsmeinungen ----------------------------------------------------------------------------------------------------------------------------
M_RANDOM, M_SPLIT, M_HUBS, M_SINGLE = "Zufällig", "Spaltung nach Gruppen", "Die Hubs dafür", "Ein einzelner Betrieb"
MODES = (M_RANDOM, M_SPLIT, M_HUBS, M_SINGLE)
DEFAULT_MODE = M_RANDOM
MODE_HELP = {
    M_RANDOM: "Jeder Betrieb beginnt mit einer zufälligen Zahl zwischen 0 und 1.",
    M_SPLIT: "Drei der sechs Gruppen beginnen bei 1, die anderen bei 0 (bei Verbünden die Verbünde, sonst sechs gleich große Nummernblöcke).",
    M_HUBS: "Die zehn Betriebe mit den meisten Partnern beginnen bei 1, alle anderen bei 0.",
    M_SINGLE: "Ein zufälliger Betrieb beginnt bei 1, alle anderen bei 0.",
}
HUBS_K = 10                                         # Zahl der Betriebe in Experiment 1 und im Modus „Die Hubs dafür“
SPLIT_GROUPS = 6

# --- Messungen -----------------------------------------------------------------------------------------------------------------------------------
TOL = 1e-6                                          # Abbruch der Iteration: Spanne der Meinungen
MAX_STEPS = 30000                                   # Obergrenze der Iteration (Gitter bei Selbstgewicht 90 % braucht Zehntausende Schritte)
PERIODIC_STEPS = 2000                               # Obergrenze, wenn W den Eigenwert −1 hat (die Meinungen pendeln; mehr Schritte ändern nichts)
FRAMES = 120                                        # Zahl der Bilder im Schritt-Regler
SPEED_SS = (0.0, 0.5)                               # Selbstgewichte in Experiment 2
INFLUENCE_TIE = "bei Gleichstand die kleinere Nummer"
STUBBORN_AS = (0.5, 0.8, 0.9, 0.95, 0.99)
COMM_INSIDES = (0.5, 0.8, 0.9, 0.95, 0.99)
COMM_N, COMM_C = 240, 6
COMM_AS = (0.8, 0.95, 0.99)
COMM_TOL = 1e-3
WISDOM_RUNS = 1000

# --- Darstellung ---------------------------------------------------------------------------------------------------------------------------------
COLOR_EDGE = "rgba(120,120,120,0.30)"
COLOR_A = "#2F6B65"
COLOR_B = "#c4623b"
COLOR_C = "#6b5b95"
TYPE_COLORS = {T_RANDOM: "#2F6B65", T_POWER: "#c4623b", T_BA: "#6b5b95", T_COMMUNITIES: "#9d755d", T_RING: "#4c78a8", T_GRID: "#8c8c8c"}

PRESETS = {
    "Zufallsnetz: Konsens (Voreinstellung)": {"netztyp": T_RANDOM, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "s": DEFAULT_S, "a": DEFAULT_A, "mode": DEFAULT_MODE},
    "Hubs bestimmen das Ergebnis": {"netztyp": T_BA, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "s": DEFAULT_S, "a": DEFAULT_A, "mode": M_HUBS},
    "Gitter: der Konsens braucht Zeit": {"netztyp": T_GRID, "n": DEFAULT_N, "c": 8, "seed": DEFAULT_SEED, "s": DEFAULT_S, "a": DEFAULT_A, "mode": DEFAULT_MODE},
    "Gitter ohne Selbstvertrauen: kein Konsens": {"netztyp": T_GRID, "n": DEFAULT_N, "c": 4, "seed": DEFAULT_SEED, "s": DEFAULT_S, "a": DEFAULT_A, "mode": DEFAULT_MODE},
    "Sturheit: Vielfalt bleibt": {"netztyp": T_RANDOM, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "s": DEFAULT_S, "a": 95, "mode": DEFAULT_MODE},
    "Spaltung entlang der Verbünde": {"netztyp": T_COMMUNITIES, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "s": DEFAULT_S, "a": 95, "mode": M_SPLIT},
}
PRESET_HELP = {
    "Zufallsnetz: Konsens (Voreinstellung)": "Zufällige Anfangsmeinungen im Zufallsnetz: alle nähern sich dem gradgewichteten Mittelwert.",
    "Hubs bestimmen das Ergebnis": "Im skalenfreien Netz haben die zehn Hubs bei 1 viel mehr Gewicht als zehn zufällige Betriebe.",
    "Gitter: der Konsens braucht Zeit": "Im Gitter mit Diagonalen laufen die Partner-Mittelungen nur langsam zusammen.",
    "Gitter ohne Selbstvertrauen: kein Konsens": "Das Gitter mit vier Partnern ist bipartit: ohne Gewicht auf die eigene Meinung pendeln die Meinungen, statt sich anzunähern.",
    "Sturheit: Vielfalt bleibt": "Mit nur 95 % Empfänglichkeit bleibt ein Rest der Anfangsvielfalt für immer.",
    "Spaltung entlang der Verbünde": "Drei Verbünde starten bei 1, drei bei 0: mit Sturheit und wenigen Querkanten bleibt die Spaltung bestehen.",
}
DEFAULT_PRESET_NAME = "Zufallsnetz: Konsens (Voreinstellung)"

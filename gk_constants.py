"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Garg-Könemann: Näherung ohne LP-Löser"."""

# --- Regler Distributionsnetz (wie in den Vorgänger-Demos) ------------------------------------------------------------------------
P_MIN, P_MAX, DEFAULT_P = 2, 6, 3            # Werke
D_MIN, D_MAX, DEFAULT_D = 2, 6, 3            # Verteilzentren
S_MIN, S_MAX, DEFAULT_S = 3, 12, 8           # Filialen
DENSITY_MIN, DENSITY_MAX, DEFAULT_DENSITY = 20, 100, 60   # Anteil vorhandener Lanes in ganzen Prozent, Schritt 10
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0, 100, 50       # Streuung der Lane-Breiten in ganzen Prozent, Schritt 25
LOAD_MIN, LOAD_MAX, DEFAULT_LOAD = 40, 160, 90            # Gesamtnachfrage in Prozent der Werkskapazität, Schritt 10
DEFAULT_SEED = 155
SEED_MAX = 2_000_000_000

# --- Regler Streckennetz ------------------------------------------------------------------------------------------------------------
GW_MIN, GW_MAX, DEFAULT_GW = 3, 8, 5                      # Breite des Gitters
GH_MIN, GH_MAX, DEFAULT_GH = 3, 6, 4                      # Höhe des Gitters
GDENSITY_MIN, GDENSITY_MAX, DEFAULT_GDENSITY = 40, 100, 70   # Anteil der Gitterkanten (über den Spannbaum hinaus), Schritt 10
GCAP_MIN, GCAP_MAX, DEFAULT_GCAP = 1, 4, 2                # größte Kapazität je Kante und Richtung
GDEM_MIN, GDEM_MAX, DEFAULT_GDEM = 1, 6, 4                # größte Menge je Gut
DEFAULT_GSEED = 7

K_MIN, K_MAX, DEFAULT_K = 2, 5, 3            # Zahl der Güter

NETS = {
    "random": "Distributionsnetz (wie im Vorgänger)",
    "grid": "Streckennetz (Gitter mit Start-Ziel-Aufträgen)",
    "preis": "Preis-Netz: zwei Güter, eine Engstelle",
    "gap": "Frachtnetz mit Bruch (2 Güter, LP gebrochen)",
}
DEFAULT_NET = "random"
FIXED_NETS = ("preis", "gap")

EPSS = (0.5, 0.3, 0.2, 0.1)
DEFAULT_EPS = 0.3
METHODS = {"gk": "Garg–Könemann (ein Orakelaufruf je Gut und Schritt)", "fleischer": "Fleischer (Phasen mit Schwelle, weniger Aufrufe)"}
DEFAULT_METHOD = "gk"
BUDGETS = (0.0, 1.0, 0.9, 0.75)                      # Kostenbudget in Anteil der Kosten des LP-Optimums; 0 = keine Kostenzeile
DEFAULT_BUDGET = 0.0
VARIANTS = {"mult": "Wie beschrieben (multiplikative Längen, Skalierung)", "add": "Negativkontrolle: additive statt multiplikative Längen", "noscale": "Negativkontrolle: ohne Skalierung am Ende"}
DEFAULT_VARIANT = "mult"

# --- feste Seed-Mengen (dieselben wie in den Vorgänger-Demos; unabhängig vom Nutzer-Seed) ------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
QUALITY_SEEDS = DIST_SEEDS[:40]                     # Verteilung der Güte (Rechenzeit: eps = 0,1 dauert je Netz eine halbe Sekunde)
EPS_SEEDS = DIST_SEEDS[:20]                         # eps-Tabelle und Negativkontrollen
SIZE_SEEDS = DIST_SEEDS[:4]
SCALE_SIZES = ((5, 4), (8, 6), (10, 6), (12, 8), (16, 10))   # Experiment: Größe (volle Gitter, K = 5)
MAX_PUSHES = 400000

COLORS = {"flow": "#1f77b4", "length": "rgba(255,127,14,0.35)", "new": "#111111", "faint": "rgba(150,150,150,0.45)", "node": "#111111", "optimal": "#d62728", "dual": "#2ca02c"}

# --- Presets -----------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", k=DEFAULT_K, p=DEFAULT_P, d=DEFAULT_D, s=DEFAULT_S, density=DEFAULT_DENSITY, spread=DEFAULT_SPREAD, load=DEFAULT_LOAD, seed=DEFAULT_SEED,
             gw=DEFAULT_GW, gh=DEFAULT_GH, gdensity=DEFAULT_GDENSITY, gcap=DEFAULT_GCAP, gdem=DEFAULT_GDEM, gseed=DEFAULT_GSEED,
             eps=DEFAULT_EPS, method=DEFAULT_METHOD, budget=DEFAULT_BUDGET, variant=DEFAULT_VARIANT)
PRESETS = {
    "🚚 Zufallsnetz": {**_BASE},
    "🗺️ Streckennetz": {**_BASE, "net": "grid", "k": 4},
    "🔀 Preis-Wende": {**_BASE, "net": "preis", "k": 2},
    "🧩 Frachtnetz mit Bruch": {**_BASE, "net": "gap", "k": 2},
    "⚡ Fleischer": {**_BASE, "method": "fleischer"},
    "🎯 Feine Näherung": {**_BASE, "eps": 0.1},
    "💰 Kostenbudget 75 %": {**_BASE, "budget": 0.75},
    "🚫 Ohne Skalierung": {**_BASE, "variant": "noscale"},
}
PRESET_HELP = {
    "🚚 Zufallsnetz": "Das Netz der Vorgänger-Demos (Seed 155, drei Güter), ε = 0,3: etwa 780 Schiebeschritte und 2 337 Kürzeste-Wege-Aufrufe, kein LP - der Fluss liefert 66,5 von 67 Einheiten (99 %), bewiesen mindestens 91 %; die Garantie wäre 49 %.",
    "🗺️ Streckennetz": "Gitter 5 × 4 mit vier Gütern: 313 Schritte, 6,5 von 7 Einheiten (93 %), bewiesen mindestens 88 %. Die orange Unterlage zeigt, welche Kanten durch das multiplikative Wachstum lang geworden sind - dort weicht der Fluss aus.",
    "🔀 Preis-Wende": "Zwei Güter von A nach D, die Direktstrecke trägt nur eine Einheit: das Verfahren schiebt zuerst beide direkt, die Direktstrecke wird lang, und der Umweg kommt dran. 43 Schritte, 1,96 von 2 Einheiten.",
    "🧩 Frachtnetz mit Bruch": "Das LP-Optimum ist gebrochen (1,5 Einheiten); das Verfahren liefert 1,42 (95 %) mit drei Pfaden - die Näherung ist wie das LP teilbar, ganzzahlig wären es nur 1,0.",
    "⚡ Fleischer": "Dasselbe Netz mit dem Verfahren von Fleischer: fast dieselben Schritte (780), aber nur 906 statt 2 337 Kürzeste-Wege-Aufrufe (2,6-fach weniger), gleiche Güte (99,9 %).",
    "🎯 Feine Näherung": "ε = 0,1: 7 480 Schritte und 22 440 Aufrufe (etwa das Zehnfache von ε = 0,3), dafür 99,8 % des Optimums, bewiesen mindestens 99,5 % - die Garantie wäre 81 %.",
    "💰 Kostenbudget 75 %": "Zusätzliche Kostenzeile mit 75 % der Kosten des LP-Optimums (1 161): das Verfahren liefert 52,4 Einheiten bei genau diesen Kosten, das LP mit Budgetzeile 56,25 (93 %). Ohne Budget wäre der Fluss fast optimal, aber 23 % teurer als das Kostenminimum.",
    "🚫 Ohne Skalierung": "Negativkontrolle: ohne Teilen durch die größte Auslastung liefert die Pfadsumme 2 935 statt höchstens 67 Einheiten und überlastet die Kanten bis zum 44-Fachen ihrer Grenze - der Fluss ist unzulässig.",
}

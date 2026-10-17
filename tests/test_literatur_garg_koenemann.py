"""Literatur-Referenzwerte für Garg-Könemann (ORACLE-PLAYBOOK §5.1). Quelle am 2026-10-17 eingesehen; die Formeln stehen in der Quelle, die Zahlen unten sind aus diesen Formeln von Hand gerechnet.

Quelle: Debmalya Panigrahi (Dozent), Kevin Sun (Mitschrift), "COMPSCI 638: Graph Algorithms, Lecture 4" (Duke University, 6. September 2019), Vorlesungsskript zu Garg und Könemann, "Faster and simpler algorithms for multicommodity
flow and other fractional packing problems", SIAM J. Comput. 37(2):630-652, 2007, https://courses.cs.duke.edu//fall19/compsci638/fall19_notes/lecture4.pdf, eingesehen 2026-10-17 (PDF, Seiten 2 bis 5):
- Algorithm 1 (Seite 2): Start Länge delta je Kante, Schritt: kürzester Weg p, gamma = kleinste Kapazität auf p, Länge je Kante auf p mal (1 + eps * gamma / u(e)).
- Lemma 1 (Seite 2 bis 3): höchstens m * log_{1+eps}((1 + eps) / delta) Iterationen.
- Lemma 2 (Seite 3): die Skalierung des Flusses durch 1 / log_{1+eps}((1 + eps) / delta) liefert einen zulässigen Fluss (je Kante f_e <= u_e * log_{1+eps}((1 + eps) / delta)).
- Theorem 3 (Seite 3 bis 5): der skalierte Fluss hat mindestens den Wert (1 - eps) mal dem Maximum (mit der Näherung (1 + eps)^Delta ~ 1 + Delta * eps bewiesen) bei delta = (1 + eps)^(1 - 1/eps) * m^(-1/eps) (Seite 5).
m ist die Zahl der Packungszeilen (Kanten); die Demo benutzt dieselben Formeln mit R Zeilen (gemeinsame Kapazität und Gut-Obergrenzen, optional Budgetzeile). Die Demo bricht bei D(y) >= 1 ab (Garg-Könemann), das Skript bei "Länge dual zulässig"
(alle Wege >= 1); die Lemmata 1 und 2 gelten für beide Abbruchregeln, weil jede Zeilenlänge vor dem letzten Schritt unter 1 / b_r bleibt und danach höchstens um (1 + eps) wächst.
Nicht eingesehen und deshalb nicht getestet: die Originalarbeit (SIAM J. Comput. 2007; nur Zusammenfassung und eine verlinkte Kopie ohne Abruf), die genaue Fassung der Garantie (1 - eps)^2 der Demo, Fleischers Phasenverfahren (Fleischer 2000).
"""

import math

import pytest

import gk_algorithm as alg
import gk_edge_lp as el
import gk_model as md
import gk_scenario as sc


def _single_path(arc_cap, demand):
    """Ein Gut, ein Weg S -> A -> B -> T: drei Packungszeilen (innere Kante, Angebots- und Nachfragekante des Guts)."""
    return md._terminals(3, [(0, 1, demand)], 1, [(2, 3, arc_cap, 1, sc.K_OTHER)], [(50, 96), (50, 4), (10, 50), (90, 50)], ["Quelle S", "Senke T", "A", "B"], ["S", "T", "A", "B"], "pairs")


def _delta(eps, m):
    return (1 + eps) ** (1 - 1 / eps) * m ** (-1 / eps)               # Seite 5 des Skripts


def _scale(eps, m):
    return math.log((1 + eps) / _delta(eps, m)) / math.log(1 + eps)     # Lemma 2


def test_delta_and_scaling_factor_by_hand_for_three_rows():
    """Von Hand: m = 3, eps = 1/2: delta = 1,5^(1 - 2) * 3^(-2) = 1 / (1,5 * 9) = 2/27; (1 + eps) / delta = 20,25 = 4,5^2, also log_{1,5}(20,25) = 2 ln 4,5 / ln 1,5 = 7,4192."""
    r = alg.garg_koenemann(_single_path(3, 2), 0.5, "gk")
    assert r.rows == 3
    assert r.delta == pytest.approx(2 / 27, rel=1e-12)
    assert r.theory_scale == pytest.approx(2 * math.log(4.5) / math.log(1.5), rel=1e-12) and r.theory_scale == pytest.approx(7.4192, abs=5e-4)


@pytest.mark.parametrize("eps", [0.5, 0.3, 0.2, 0.1])
@pytest.mark.parametrize("which", ["path", "distribution"])
def test_delta_and_scaling_follow_the_lecture_formulas(eps, which):
    mcf = _single_path(3, 2) if which == "path" else md.generate_mcf(2, 2, 3, 60, 50, 90, 4, 1)
    r = alg.garg_koenemann(mcf, eps, "gk")
    assert r.delta == pytest.approx(_delta(eps, r.rows), rel=1e-12)
    assert r.theory_scale == pytest.approx(_scale(eps, r.rows), rel=1e-9)


def test_length_update_after_the_first_push_follows_algorithm_1():
    """Algorithm 1, Zeile 8: l_e <- (1 + eps * gamma / u(e)) * l_e. Ein Weg, innere Kapazität 3, Nachfrage 2: gamma = 2 (Engpass), die innere Kante hat danach die Länge delta / 3 * (1 + eps * 2 / 3)."""
    eps = 0.5
    mcf = _single_path(3, 2)
    r = alg.garg_koenemann(mcf, eps, "gk")
    first = r.frames[0]
    assert first.push == 1 and first.amount == pytest.approx(2.0)
    assert first.y[0] == pytest.approx(_delta(eps, 3) / 3 * (1 + eps * 2 / 3), rel=1e-12)
    # Engpass = innere Kante (Kapazität 3 < Nachfrage 4): gamma = u(e), Faktor genau 1 + eps
    r = alg.garg_koenemann(_single_path(3, 4), eps, "gk")
    assert r.frames[0].amount == pytest.approx(3.0) and r.frames[0].y[0] == pytest.approx(_delta(eps, 3) / 3 * (1 + eps), rel=1e-12)


def _nets():
    out = [md.generate_mcf(2, 2, 4, 60, 50, 90, s, 1 + s % 3) for s in range(6)]
    out += [md.generate_grid(3, 2, 70, 2, 2, 1 + s % 3, 100 + s) for s in range(4)]
    out += [md.generate_pairs(5, 60, 2, 200 + s, 2, 2) for s in range(3)]
    return out


@pytest.mark.parametrize("method", ["gk", "fleischer"])
@pytest.mark.parametrize("eps", [0.5, 0.3])
def test_lemma_1_and_2_bounds_and_theorem_3_factor(method, eps):
    for mcf in _nets():
        opt = el.solve_lp(mcf).total_delivered
        if opt < 1e-9:
            continue
        r = alg.garg_koenemann(mcf, eps, method)
        m = r.rows
        scale = _scale(eps, m)
        assert r.pushes <= m * scale + 1e-9                        # Lemma 1: höchstens m log_{1+eps}((1+eps)/delta) Iterationen
        assert r.congestion <= scale + 1e-9                         # Lemma 2: je Zeile ist der ungeskalierte Fluss höchstens scale mal die Grenze, der geteilte Fluss zulässig
        assert r.primal_theory <= opt + 1e-6 and r.primal >= r.primal_theory - 1e-9
        assert r.primal >= (1 - eps) * opt - 1e-6                   # Theorem 3 (1 - eps) (die Skriptfassung beweist es mit einer Näherung, die Demo erreicht es mit Abstand: kleinstes Verhältnis 0,83 bei eps = 0,5)

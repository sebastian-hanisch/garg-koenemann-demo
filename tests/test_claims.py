"""Jede Zahl, die README und Hilfetexte nennen, ist hier belegt (Standardeinstellungen, feste Netze, Seeds ab 100000).
Das Verfahren ist deterministisch und braucht keinen LP-Löser; das Vergleichs-LP (Optimum) liefert HiGHS. Anteile, Schritte und Aufrufe werden mit Bändern geprüft (Gleitkomma-Summen können auf anderen Plattformen um einen Schritt abweichen)."""

import math

import pytest

import gk_constants as C
import gk_evaluation as ev

P = ev.DEFAULT_PARAMS
GRID = P._replace(net="grid", k=4)
GK = ev.Opts(0.3, "gk", 0.0, "mult")
FL = ev.Opts(0.3, "fleischer", 0.0, "mult")


@pytest.fixture(scope="module")
def eps_rows():
    return ev.eps_table(P)


def test_quality_distribution_default_net():
    """Distributionsnetz (drei Güter, ε = 0,3, Garg-Könemann, 40 feste Netze): Fluss im Mittel etwa 95 % des Optimums (schlechtestes Netz etwa 90 %), bewiesen im Mittel etwa 87 %, mit der Skalierung der Theorie
    etwa 77 %; Garantie 49 %; etwa 760 Schritte, dreimal so viele Aufrufe; die obere Schranke ist in allen Netzen gültig."""
    d = ev.distribution(P, GK)
    assert d["n_seeds"] == 40 and d["share_dual_valid"] == 1.0
    assert 0.94 <= d["ratio_mean"] <= 0.965 and 0.89 <= d["ratio_min"] <= 0.93 and 0.85 <= d["certified_mean"] <= 0.90 and 0.75 <= d["theory_mean"] <= 0.80 and d["theory_min"] >= 0.49
    assert 700 <= d["pushes_mean"] <= 830 and d["calls_mean"] == pytest.approx(3 * d["pushes_mean"]) and d["pushes_mean"] < 0.3 * d["bound_mean"]
    assert 40 <= d["congestion_mean"] <= 48


def test_quality_distribution_grid():
    """Streckennetz (5x4, vier Güter, ε = 0,3): etwa 96 % des Optimums (schlechtestes etwa 91 %), etwa 270 Schritte, etwa 1 080 Aufrufe."""
    d = ev.distribution(GRID, GK)
    assert d["share_dual_valid"] == 1.0 and 0.95 <= d["ratio_mean"] <= 0.975 and 0.89 <= d["ratio_min"] <= 0.93 and 250 <= d["pushes_mean"] <= 300 and d["calls_mean"] == pytest.approx(4 * d["pushes_mean"])


def test_fleischer_saves_calls_without_losing_quality():
    """Fleischer: gleiche Schritte, gleiche Güte (etwa 95 %), aber etwa 890 statt 2 290 Aufrufe (2,6-fach weniger); die bewiesene Güte ist etwas geringer (etwa 84 % gegen 87 %), weil die Schranke nur am Phasenende bekannt ist."""
    g, f = ev.distribution(P, GK), ev.distribution(P, FL)
    assert abs(f["pushes_mean"] - g["pushes_mean"]) < 0.03 * g["pushes_mean"] and abs(f["ratio_mean"] - g["ratio_mean"]) < 0.01
    assert 2.3 <= g["calls_mean"] / f["calls_mean"] <= 3.0 and 800 <= f["calls_mean"] <= 980 and f["certified_mean"] < g["certified_mean"] and f["share_dual_valid"] == 1.0
    gg, gf = ev.distribution(GRID, GK), ev.distribution(GRID, FL)
    assert 2.2 <= gg["calls_mean"] / gf["calls_mean"] <= 2.8


def test_eps_table(eps_rows):
    """20 feste Netze: tatsächliche Güte für ε = 0,5 / 0,3 / 0,2 / 0,1 etwa 91 / 95 / 97 / 98,6 % gegen die Garantie 25 / 49 / 64 / 81 %; mit der Skalierung der Theorie etwa 59 / 77 / 85 / 93 %;
    bewiesen etwa 75 / 88 / 93 / 97 %; Schritte etwa 230 / 780 / 1 850 / 7 660, wachsen mit Steigung etwa 2,2 gegen 1/ε (Theorie: 2); die Schritte liegen weit unter der Schranke."""
    gk = [r for r in eps_rows if r["method"] == "gk"]
    assert [r["eps"] for r in gk] == [0.5, 0.3, 0.2, 0.1] and all(r["dual_valid"] and r["feasible"] for r in eps_rows)
    assert [round(100 * r["guarantee"]) for r in gk] == [25, 49, 64, 81]
    assert all(lo <= r["ratio"] <= hi for r, (lo, hi) in zip(gk, ((0.89, 0.93), (0.94, 0.965), (0.96, 0.98), (0.975, 0.995))))
    assert all(lo <= r["theory"] <= hi for r, (lo, hi) in zip(gk, ((0.55, 0.63), (0.75, 0.80), (0.83, 0.87), (0.91, 0.945))))
    assert all(lo <= r["certified"] <= hi for r, (lo, hi) in zip(gk, ((0.71, 0.79), (0.85, 0.90), (0.90, 0.95), (0.95, 0.985))))
    assert all(r["ratio"] > r["theory"] > r["guarantee"] for r in gk) and all(r["pushes"] < 0.5 * r["bound"] for r in gk)
    slope = math.log(gk[3]["pushes"] / gk[0]["pushes"]) / math.log(0.5 / 0.1)
    assert 2.0 <= slope <= 2.4 and 200 <= gk[0]["pushes"] <= 270 and 6900 <= gk[3]["pushes"] <= 8400


def test_fleischer_calls_across_eps(eps_rows):
    """Aufrufe Garg-Könemann etwa 690 / 2 340 / 5 560 / 22 990, Fleischer etwa 270 / 900 / 2 140 / 8 810: in jedem ε etwa 2,6-fach weniger."""
    gk = [r for r in eps_rows if r["method"] == "gk"]
    fl = [r for r in eps_rows if r["method"] == "fleischer"]
    assert all(2.3 <= g["calls"] / f["calls"] <= 3.0 for g, f in zip(gk, fl)) and 20000 <= gk[3]["calls"] <= 26000 and 8000 <= fl[3]["calls"] <= 9800
    assert all(abs(g["ratio"] - f["ratio"]) < 0.03 for g, f in zip(gk, fl))


def test_negative_controls():
    """ε = 0,3, 20 Netze: wie beschrieben etwa 95 % des Optimums; additive Längen brechen nach etwa 6 Schritten mit etwa 49 % ab; ohne Skalierung ist der Fluss etwa das 42-Fache des Optimums und die Kanten sind bis zum
    etwa 44-Fachen überlastet."""
    rows = {r["variant"]: r for r in ev.control_table(P)}
    assert 0.94 <= rows["mult"]["ratio"] <= 0.965 and rows["mult"]["feasible"] == 1.0
    assert 0.40 <= rows["add"]["ratio"] <= 0.58 and rows["add"]["ratio_min"] < 0.5 and 3 <= rows["add"]["pushes"] <= 12 and rows["add"]["feasible"] == 1.0
    assert 38 <= rows["noscale"]["ratio"] <= 46 and 40 <= rows["noscale"]["congestion"] <= 48 and rows["noscale"]["feasible"] == 0.0


def test_budget_table():
    """Seed 155, ε = 0,2: Kostenbudget 100 / 90 / 75 / 50 % der Kosten des LP-Optimums (1548): LP-Optimum 67 / 63,3 / 56,3 / 43,0, das Verfahren etwa 96 % davon bei genau diesen Kosten; ohne Budget etwa 99,5 % zu 23 % höheren Kosten
    als das Kostenminimum."""
    rows = ev.budget_table(P)
    assert [r["frac"] for r in rows] == [0.0, 1.0, 0.9, 0.75, 0.5] and rows[0]["opt_cost"] == 1936.0
    assert [round(r["opt"], 2) for r in rows[1:]] == [67.0, 63.26, 56.25, 43.04]
    assert all(r["feasible"] and 0.94 <= r["ratio"] <= 0.99 for r in rows[1:]) and all(abs(r["cost"] - r["budget"]) < 0.02 * r["budget"] for r in rows[2:])
    assert rows[0]["ratio"] >= 0.99 and 1.20 <= rows[0]["cost"] / 1548.0 <= 1.26


def test_size_table():
    """Volle Gitter mit fünf Gütern (Fleischer, ε = 0,3, 4 feste Netze je Größe): Zeilen 72 / 174 / 218 / 354 / 598, Schritte etwa 390 auf 980, Aufrufe etwa 620 auf 1 330 - 20- bis 30-mal die Aufrufe der Column Generation
    (etwa 26 bis 205); Güte etwa 95 bis 97 %; in Sekunden etwa 10- bis 19-mal langsamer als HiGHS auf dem Kanten-LP."""
    rows = ev.size_table()
    assert [r["rows"] for r in rows] == [72, 174, 218, 354, 598]
    assert 340 <= rows[0]["pushes"] <= 440 and 900 <= rows[-1]["pushes"] <= 1060 and 550 <= rows[0]["calls"] <= 700 and 1200 <= rows[-1]["calls"] <= 1450
    assert all(r["calls"] > 5 * r["cg_calls"] for r in rows) and rows[0]["calls"] > 15 * rows[0]["cg_calls"] and all(0.93 <= r["ratio"] <= 0.99 for r in rows)
    assert all(r["t_gk"] > 3 * r["t_lp"] for r in rows) and rows[-1]["t_gk"] / rows[-1]["t_lp"] > rows[0]["t_gk"] / rows[0]["t_lp"] * 0.8      # nur zur Information; ein Kreuzungspunkt existiert nicht


def test_seed_155_numbers():
    """Beispielnetz: 67 von 76 Einheiten, Garg-Könemann ε = 0,3: 66,5 Einheiten (99 %), bewiesen mindestens 91 %, 56 Zeilen, etwa 780 Schritte, 3 mal so viele Aufrufe; Kosten etwa das 1,23-Fache des Kostenminimums.
    Fleischer: 906 Aufrufe; ε = 0,1: etwa 7 480 Schritte, 99,8 %, bewiesen 99,5 %."""
    d = ev.verdict(ev.analyse(P))[2]
    assert d["opt"] == 67.0 and d["rows"] == 56 and 0.985 <= d["ratio"] <= 0.998 and 0.88 <= d["certified"] <= 0.93 and 740 <= d["pushes"] <= 820 and 1.20 <= d["cost_ratio"] <= 1.26
    f = ev.verdict(ev.analyse(P, ev.DEFAULT_OPTS._replace(method="fleischer")))[2]
    assert 850 <= f["calls"] <= 960 and 2.4 <= d["calls"] / f["calls"] <= 2.8 and f["ratio"] > 0.995
    fine = ev.verdict(ev.analyse(P, ev.DEFAULT_OPTS._replace(eps=0.1)))[2]
    assert 7200 <= fine["pushes"] <= 7800 and fine["ratio"] > 0.995 and fine["certified"] > 0.99 and 0.93 <= fine["theory_ratio"] <= 0.96


def test_teaching_nets_and_grid_preset():
    """Streckennetz (Seed 7): 6,5 von 7 Einheiten (93 %), bewiesen 88 %, etwa 313 Schritte; Preis-Wende: 1,96 von 2, 43 Schritte; Frachtnetz mit Bruch: 1,42 von 1,5, drei Pfade."""
    d = ev.verdict(ev.analyse(GRID))[2]
    assert d["opt"] == 7.0 and 0.91 <= d["ratio"] <= 0.95 and 0.85 <= d["certified"] <= 0.92 and 290 <= d["pushes"] <= 340 and d["rows"] == 64
    pr = ev.verdict(ev.analyse(ev.normalise(P._replace(net="preis"))))[2]
    assert pr["opt"] == 2.0 and 1.9 <= pr["primal"] <= 2.0 and 38 <= pr["pushes"] <= 50 and pr["rows"] == 7
    gp = ev.verdict(ev.analyse(ev.normalise(P._replace(net="gap"))))[2]
    assert gp["opt"] == 1.5 and 1.38 <= gp["primal"] <= 1.46 and gp["columns"] == 3 and gp["rows"] == 13

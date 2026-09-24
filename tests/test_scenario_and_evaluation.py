"""Auswertung: Netzbau, Urteil, Verteilungen, eps-, Kontroll-, Budget- und Größentabelle, Anzeige-Helfer."""

import pytest

import gk_algorithm as alg
import gk_constants as C
import gk_evaluation as ev
import gk_model as md

P = ev.DEFAULT_PARAMS
GRID = P._replace(net="grid", k=4)


def test_build_dispatches_on_the_net_and_ignores_the_random_settings_for_fixed_nets():
    assert ev.build(P._replace(net="gap")) == md.gap_net() and ev.build(P._replace(net="preis")) == md.preis_net()
    assert ev.build(P) == md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3) and ev.build(GRID) == md.generate_grid(5, 4, 70, 2, 4, 4, 7)
    assert ev.normalise(P._replace(net="gap", k=5, seed=9, gw=7)) == ev.normalise(P._replace(net="gap"))


def test_with_seed_changes_only_the_seed_of_the_shown_vehicle():
    assert ev.with_seed(P, 5).seed == 5 and ev.with_seed(P, 5).gseed == P.gseed
    assert ev.with_seed(GRID, 5).gseed == 5 and ev.with_seed(GRID, 5).seed == P.seed


def test_verdict_codes_and_data():
    lvl, code, d = ev.verdict(ev.analyse(P))
    assert (lvl, code) == ("success", "ok") and d["opt"] == 67.0 and d["primal"] <= d["opt"] + 1e-6 and d["dual_valid"] and d["ratio"] <= 1 + 1e-9 and d["calls"] == 3 * d["pushes"]
    assert d["certified"] <= 1 + 1e-9 and d["pushes"] <= d["bound"] and d["guarantee"] == pytest.approx(0.49) and d["opt_cost"] >= d["lp_cost"]
    lvl, code, d = ev.verdict(ev.analyse(P, ev.DEFAULT_OPTS._replace(variant="noscale")))
    assert (lvl, code) == ("error", "noscale") and d["primal"] > 10 * d["opt"] and d["congestion"] > 10
    lvl, code, d = ev.verdict(ev.analyse(P, ev.DEFAULT_OPTS._replace(variant="add")))
    assert (lvl, code) == ("warning", "bad") and d["ratio"] < 0.9
    lvl, code, d = ev.verdict(ev.analyse(P, ev.DEFAULT_OPTS._replace(budget=0.75)))
    assert code == "ok" and d["budget"] == pytest.approx(0.75 * 1548.0) and d["cost"] <= d["budget"] + 1e-6 and d["opt"] == pytest.approx(56.25)
    lvl, code, d = ev.verdict(ev.analyse(P._replace(p=2, d=6, s=3, density=20, seed=8)))
    assert code == "nothing" and d["opt"] == 0


def test_distribution_is_consistent_with_single_runs():
    seeds = tuple(range(4))
    dist = ev.distribution(P, seeds=seeds)
    for i, seed in enumerate(seeds):
        d = ev.verdict(ev.analyse(ev.with_seed(P, seed)))[2]
        assert dist["cols"]["pushes"][i] == d["pushes"] and dist["cols"]["calls"][i] == d["calls"] and dist["cols"]["ratio"][i] == pytest.approx(d["ratio"])
    assert dist["n_seeds"] == 4 and dist["share_dual_valid"] == 1.0 and dist["ratio_min"] <= dist["ratio_mean"] <= 1 + 1e-9
    assert ev.distribution(GRID, seeds=seeds)["n_seeds"] == 4


def test_eps_table_has_two_rows_per_eps_in_the_documented_order():
    rows = ev.eps_table(P, seeds=C.EPS_SEEDS[:2])
    assert [(r["eps"], r["method"]) for r in rows] == [(e, m) for e in C.EPSS for m in ("gk", "fleischer")]
    assert all(r["dual_valid"] and r["feasible"] and r["theory"] >= r["guarantee"] - 1e-9 and r["ratio"] >= r["theory"] - 1e-9 for r in rows)
    gk = [r for r in rows if r["method"] == "gk"]
    assert [r["pushes"] for r in gk] == sorted(r["pushes"] for r in gk) and all(f["calls"] < g["calls"] for g, f in zip(gk, [r for r in rows if r["method"] == "fleischer"]))


def test_control_table_shows_what_each_ingredient_does():
    rows = {r["variant"]: r for r in ev.control_table(P, seeds=C.EPS_SEEDS[:4])}
    assert set(rows) == set(C.VARIANTS)
    assert rows["mult"]["feasible"] == 1.0 and rows["add"]["feasible"] == 1.0 and rows["noscale"]["feasible"] == 0.0
    assert rows["noscale"]["ratio"] > 10 and rows["add"]["ratio"] < rows["mult"]["ratio"] - 0.2 and rows["add"]["pushes"] < 50


def test_budget_table_stays_inside_the_budget_and_below_the_lp():
    rows = ev.budget_table(ev.normalise(P._replace(net="preis")))
    assert [r["frac"] for r in rows] == [0.0, 1.0, 0.9, 0.75, 0.5] and rows[0]["budget"] is None
    assert all(r["feasible"] and r["primal"] <= r["opt"] + 1e-6 and r["dual"] >= r["opt"] - 1e-6 for r in rows)
    assert [r["opt"] for r in rows[1:]] == sorted((r["opt"] for r in rows[1:]), reverse=True)


def test_size_table_shapes():
    rows = ev.size_table(sizes=((3, 3), (4, 3)), K=3, seeds=C.SIZE_SEEDS[:2])
    assert [r["size"] for r in rows] == [(3, 3), (4, 3)] and all(r["calls"] > r["cg_calls"] and 0.7 < r["ratio"] <= 1 + 1e-9 for r in rows) and rows[0]["rows"] < rows[1]["rows"]


def test_frame_ratio_and_pushes_bound():
    a = ev.analyse(P)
    cert, act = ev.frame_ratio(a.gk.frames[-1], a.opt)
    assert act == pytest.approx(a.gk.primal / a.opt) and cert == pytest.approx(a.gk.primal / a.gk.dual)
    assert ev.pushes_bound(a.gk) == a.gk.rows * a.gk.theory_scale


def test_path_label():
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    r = ev.analyse(P).gk
    label = ev.path_label(mcf, r.columns[0][1])
    assert label.startswith("W") and label.split(" → ")[-1].startswith("F")
    grid = md.generate_grid(3, 3, 100, 2, 3, 2, 1)
    assert all(part.startswith("(") for part in ev.path_label(grid, alg.garg_koenemann(grid, 0.5).columns[0][1]).split(" → "))

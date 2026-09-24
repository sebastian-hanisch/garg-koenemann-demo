"""Kern: Garg-Könemann und Fleischer gegen unabhängige Gegenproben (Kanten-LP mit HiGHS): Zulässigkeit des skalierten Flusses, gültige obere Schranke, Theorie-Garantie, Schrittschranke, Aufrufe,
Kostenbudget gegen das LP mit Budgetzeile, Negativkontrollen, Lehrnetze, Aufnahmepunkte."""

import random

import networkx as nx
import numpy as np
import pytest

import gk_algorithm as alg
import gk_cg as cg
import gk_constants as C
import gk_edge_lp as el
import gk_lp
import gk_model as md
import gk_scenario as sc
import gk_ssp as ssp
from gk_scenario import SplitMix64

TOL = 1e-6


def _dist(count):
    rng = random.Random(31)
    sizes = ((2, 2, 3), (3, 3, 6), (3, 3, 8), (4, 3, 5), (2, 4, 9))
    for i in range(count):
        p, d, s = sizes[i % len(sizes)]
        yield md.generate_mcf(p, d, s, rng.choice((40, 60, 80, 100)), rng.choice((0, 50, 100)), rng.choice((60, 90, 130)), 6000 + i, 2 + i % 3)


def _grids(count):
    rng = random.Random(32)
    for i in range(count):
        yield md.generate_grid(rng.choice((3, 4, 5)), rng.choice((3, 4)), rng.choice((50, 70, 100)), rng.choice((1, 2, 3)), rng.choice((2, 4)), 2 + i % 3, 7000 + i)


def _check_feasible(mcf, x, tol=TOL):
    """Erhaltung je Gut, gemeinsame Kapazität, gutspezifische Obergrenzen, Nichtnegativität des skalierten Kantenflusses."""
    net = mcf.net
    assert (x >= -tol).all()
    for k in range(mcf.K):
        bal = np.zeros(net.n)
        for e, (u, v, _, _, _) in enumerate(net.arcs):
            bal[u] -= x[k, e]
            bal[v] += x[k, e]
            assert x[k, e] <= min(mcf.ub[k][e], 10 ** 6) + tol
        assert all(abs(bal[v]) <= tol for v in range(net.n) if v not in (net.s, net.t))
    for e, (_, _, cap, _, _) in enumerate(net.arcs):
        if mcf.joint[e]:
            assert x[:, e].sum() <= cap + tol


def test_splitmix64_reference_vector():
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]


def test_the_ssp_copy_reproduces_the_predecessor_numbers():
    """Wache: SSP 53 907 durchsuchte Kanten (Mittel 539,07) über die 100 festen Netze der Vorgänger-Demos, Kosten = networkx."""
    scans = 0
    for seed in C.DIST_SEEDS:
        net = sc.generate(3, 3, 8, 60, 50, 90, seed)
        r = ssp.ssp(net, keep_trace=False)
        scans += r.scanned_total
        if seed < C.DIST_SEEDS[0] + 20:
            g = nx.DiGraph()
            for u, v, c, k, _ in net.arcs:
                g.add_edge(u, v, capacity=c, weight=k)
            flow = nx.max_flow_min_cost(g, net.s, net.t)
            assert (r.value, r.total) == (sum(flow[net.s].values()), nx.cost_of_flow(g, flow))
    assert scans == 53907


def test_the_predecessor_copies_reproduce_their_numbers():
    """Kanten-LP (Seed 155, drei Güter): 67 von 76 Einheiten für 1548, 114 Variablen; Column Generation des Vorgängers gleich dem LP; Frachtnetz mit Bruch 1,5 für 24."""
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    lp = el.solve_lp(mcf)
    assert (lp.total_delivered, mcf.total_demand(), lp.cost, lp.n_vars) == (67.0, 76, 1548.0, 114)
    assert abs(cg.column_generation(mcf).obj - lp.objective) < TOL
    gap = md.gap_net()
    assert (el.solve_lp(gap).total_delivered, el.solve_lp(gap).cost) == (1.5, 24.0)


def test_the_rows_are_those_of_the_column_generation_master():
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    joint_rows, ub_rows, b = alg.rows_of(mcf)
    master = cg.Master(mcf)
    assert joint_rows == master.joint_rows and ub_rows == master.ub_rows and (b == master.b).all()
    res = alg.garg_koenemann(mcf, 0.5)
    assert res.rows == len(b) == 56 and alg.garg_koenemann(mcf, 0.5, budget=1000.0).rows == 57


def test_max_delivery_lp_agrees_with_the_lexicographic_lp_and_shrinks_with_the_budget():
    for mcf in list(_dist(6)) + list(_grids(6)):
        lp = el.solve_lp(mcf)
        opt, cost, x = gk_lp.solve_max_delivery(mcf)
        assert abs(opt - lp.total_delivered) < TOL and cost >= lp.cost - TOL
        _check_feasible(mcf, x)
        prev = opt
        for frac in (1.0, 0.9, 0.75, 0.5):
            o, c, xb = gk_lp.solve_max_delivery(mcf, frac * lp.cost)
            assert o <= prev + TOL and c <= frac * lp.cost + TOL
            _check_feasible(mcf, xb)
            prev = o
        assert abs(gk_lp.solve_max_delivery(mcf, lp.cost)[0] - lp.total_delivered) < TOL              # das billigste Optimum passt ins Budget 100 %


# --- Zulässigkeit, Schranken, Garantie -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gk", "fleischer"])
@pytest.mark.parametrize("eps", [0.5, 0.3, 0.2])
def test_the_scaled_flow_is_feasible_and_the_bounds_enclose_the_optimum(method, eps):
    for mcf in list(_dist(8)) + list(_grids(8)):
        opt = el.solve_lp(mcf).total_delivered
        r = alg.garg_koenemann(mcf, eps, method)
        _check_feasible(mcf, r.x)
        assert r.stopped == "D>=1" and abs(r.primal - r.amounts.sum()) < TOL and abs(r.primal - r.x[:, [e for e in range(mcf.m) if mcf.reward[e]]].sum()) < TOL
        assert r.primal <= opt + TOL                                       # zulässig: nie mehr als das Optimum
        assert r.dual >= opt - TOL                                         # bewiesene obere Schranke
        assert r.primal_theory >= (1 - eps) ** 2 * opt - TOL               # die Garantie der Theorie
        assert r.primal >= r.primal_theory - TOL                           # die Skalierung durch die größte Auslastung ist nie schlechter als die der Theorie
        assert r.pushes <= r.rows * r.theory_scale + 1                     # beweisbare Schranke der Schritte
        assert 0 < r.gap <= 1 + TOL


def test_garg_koenemann_asks_every_good_every_step_and_fleischer_asks_less():
    for mcf in list(_dist(6)) + list(_grids(6)):
        g, f = alg.garg_koenemann(mcf, 0.3, "gk"), alg.garg_koenemann(mcf, 0.3, "fleischer")
        assert g.calls == mcf.K * g.pushes
        assert f.calls < g.calls and f.phases > 0 and g.phases == 0
        assert abs(f.pushes - g.pushes) <= 0.1 * g.pushes + 2              # dieselbe Größenordnung an Schritten


def test_the_theory_scale_and_delta_follow_the_formulas():
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    r = alg.garg_koenemann(mcf, 0.3)
    delta = 1.3 * (1.3 * 56) ** (-1 / 0.3)
    assert abs(r.delta - delta) < 1e-15 * max(1, 1 / delta) and abs(r.theory_scale - np.log(1.3 / delta) / np.log(1.3)) < 1e-9


def test_smaller_eps_needs_more_steps_and_gives_a_better_flow():
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    res = [alg.garg_koenemann(mcf, eps, "fleischer") for eps in (0.5, 0.3, 0.2, 0.1)]
    assert [r.pushes for r in res] == sorted(r.pushes for r in res) and res[-1].pushes > 20 * res[0].pushes
    assert res[-1].primal > res[0].primal and res[-1].gap > res[0].gap


def test_the_run_is_deterministic():
    mcf = md.generate_grid(5, 4, 70, 2, 4, 4, 7)
    a, b = alg.garg_koenemann(mcf, 0.3, "gk"), alg.garg_koenemann(mcf, 0.3, "gk")
    assert (a.pushes, a.calls, a.primal, a.dual) == (b.pushes, b.calls, b.primal, b.dual) and a.columns == b.columns and (a.amounts == b.amounts).all()


def test_the_frames_are_consecutive_feasible_and_the_last_is_the_result():
    for mcf in list(_dist(4)) + list(_grids(4)):
        opt = el.solve_lp(mcf).total_delivered
        r = alg.garg_koenemann(mcf, 0.3, "gk")
        pushes = [f.push for f in r.frames]
        assert pushes == sorted(pushes) and pushes[:8] == list(range(1, 9)) and len(r.frames) <= 62
        for f in r.frames:
            _check_feasible(mcf, f.x)
            assert f.primal <= opt + TOL and abs(f.primal - f.x[:, [e for e in range(mcf.m) if mcf.reward[e]]].sum()) < TOL
        duals = [f.dual for f in r.frames if f.dual not in (0.0, float("inf"))]
        assert all(b <= a + 1e-9 for a, b in zip(duals, duals[1:])) and all(d >= opt - TOL for d in duals)
        last = r.frames[-1]
        assert abs(last.primal - r.primal) < TOL and last.dual == r.dual and (last.x == r.x).all()


def test_fleischer_frames_carry_a_bound_only_after_a_complete_phase():
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    opt = el.solve_lp(mcf).total_delivered
    r = alg.garg_koenemann(mcf, 0.3, "fleischer")
    assert all(f.dual >= opt - TOL for f in r.frames if f.dual not in (0.0, float("inf"))) and r.dual >= opt - TOL


# --- Kostenbudget -------------------------------------------------------------------------------------------------------------------------------

def test_the_budget_row_keeps_the_cost_and_stays_below_the_lp_with_budget():
    for mcf in list(_dist(5)) + list(_grids(5)):
        lp = el.solve_lp(mcf)
        for frac in (1.0, 0.75, 0.5):
            budget = frac * lp.cost
            opt = gk_lp.solve_max_delivery(mcf, budget)[0]
            for method in ("gk", "fleischer"):
                r = alg.garg_koenemann(mcf, 0.3, method, budget=budget)
                _check_feasible(mcf, r.x)
                assert r.cost <= budget + TOL and r.primal <= opt + TOL and r.dual >= opt - TOL and r.primal_theory >= 0.49 * opt - TOL


def test_without_a_budget_the_cost_is_not_minimised():
    """Das Verfahren maximiert nur die Menge: der Fluss ist im Mittel teurer als das Kostenminimum des LP-Optimums."""
    ratios = []
    for mcf in _dist(10):
        lp = el.solve_lp(mcf)
        if lp.cost > 0:
            ratios.append(alg.garg_koenemann(mcf, 0.2, "gk").cost / lp.cost)
    assert sum(ratios) / len(ratios) > 1.05


# --- Negativkontrollen und Lehrnetze -------------------------------------------------------------------------------------------------------------

def test_without_the_final_scaling_the_flow_is_infeasible():
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    opt = el.solve_lp(mcf).total_delivered
    r = alg.garg_koenemann(mcf, 0.3, "gk", scale=False)
    assert r.congestion > 10 and r.primal > 10 * opt and r.x[:, [e for e in range(mcf.m) if mcf.joint[e]]].sum(axis=0).max() > 10 * max(a[2] for a in mcf.net.arcs) / 10


def test_additive_lengths_stop_early_with_a_poor_flow():
    add, mult = [], []
    for mcf in _dist(8):
        opt = el.solve_lp(mcf).total_delivered
        r = alg.garg_koenemann(mcf, 0.3, "gk", update="add")
        _check_feasible(mcf, r.x)
        assert r.pushes < 50
        add.append(r.primal / opt if opt > 0 else 1)
        mult.append(alg.garg_koenemann(mcf, 0.3, "gk").primal / opt if opt > 0 else 1)
    assert sum(add) / len(add) < 0.7 < 0.9 < sum(mult) / len(mult)


def test_the_teaching_nets():
    """Preis-Wende: Optimum 2 (eine Einheit direkt, eine über den Umweg); Frachtnetz mit Bruch: Optimum 1,5, gebrochene Pfadmengen."""
    pr = md.preis_net()
    r = alg.garg_koenemann(pr, 0.1, "gk")
    assert r.primal >= 1.95 and r.primal <= 2.0 + TOL and r.dual >= 2.0 - TOL and len({c for c in r.columns}) == 4
    gp = md.gap_net()
    g = alg.garg_koenemann(gp, 0.1, "gk")
    assert 1.4 <= g.primal <= 1.5 + TOL and g.dual >= 1.5 - TOL and any(abs(a - round(a)) > 1e-6 for a in g.amounts)


def test_a_net_without_a_path_stops_at_once():
    mcf = md.generate_mcf(2, 6, 3, 20, 50, 90, 8, 3)
    assert el.solve_lp(mcf).total_delivered == 0
    r = alg.garg_koenemann(mcf, 0.3)
    assert r.primal == 0 and r.pushes == 0


def test_max_pushes_stops_the_run_honestly():
    mcf = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    r = alg.garg_koenemann(mcf, 0.1, "gk", max_pushes=100)
    assert r.stopped == "max_pushes" and r.pushes == 100
    _check_feasible(mcf, r.x)

"""Eigenschaftstests (metamorph) für Garg-Könemann und Fleischer: Aussagen, die für das (1 - eps)-Näherungsverfahren und das Packungs-LP gelten müssen, ohne dass man den genauen Lauf kennt.

Je Eigenschaft 12 bis 20 Zufallsinstanzen mit festem Seed (kleine Distributions-, Strecken- und Frachtnetze, 1 bis 3 Güter, eps 0,5 bis 0,15, mit und ohne Kostenbudget; mehr Instanzen sprengen die Zehn-Sekunden-Grenze, weil jeder Lauf
hunderte Dijkstra-Aufrufe braucht) plus Randfälle. Das Optimum liefert HiGHS (Kanten-LP der Demo,
`scipy.optimize.linprog`), die Garantie steht in README und `gk_algorithm`; Belege nach ORACLE-PLAYBOOK §5.2 (Optimierung):

1. Alle Kapazitäten (und das Kostenbudget) mal c = 2 oder 4: der Lauf ist derselbe (Schritte, Aufrufe, Pfade), Lieferung und obere Schranke wachsen mit c (alle Längen skalieren mit 1/c, nur Vergleiche und Verhältnisse kommen vor).
2. Kosten und Budget gemeinsam mal c: unveränderter Lauf, unveränderte Lieferung.
3. Schwache Dualität für beliebige positive Längen: OPT <= D(y) / alpha(y) (alpha = kürzester Weg über alle Güter, D = Summe b_r y_r).
4. Umnummerieren der Knoten und Umordnen der Güter: dasselbe LP-Optimum; die Lieferung liegt im Band [(1 - eps)^2 OPT, OPT] und zwei Läufe unterscheiden sich höchstens um die Bandbreite.
5. Nach der Skalierung durch die größte Auslastung ist mindestens eine Zeile genau voll (Auslastung 1), keine über 1.
6. Ein Gut gegen Max-Flow (networkx): Lieferung zulässig, im Band (1 - eps)^2 F bis F, obere Schranke >= F, auch für kleine eps.
7. Randfälle: Netz ohne Weg (nichts wird geschoben), ein einziger Weg (Lieferung genau die Engpasskapazität), Obergrenze der Nachfrage kleiner als die Kapazität."""

import dataclasses
import math
import random

import networkx as nx
import numpy as np
import pytest

import gk_algorithm as alg
import gk_edge_lp as el
import gk_model as md
import gk_scenario as sc

TOL = 1e-6
BIG = md.BIG


def _instances(seed, count):
    """Gemischte kleine Instanzen: Distributions-, Strecken- und Frachtnetze mit 1 bis 3 Gütern."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        kind = i % 3
        if kind == 0:
            out.append(md.generate_mcf(rng.choice((2, 3)), rng.choice((2, 3)), rng.choice((3, 4, 6)), rng.choice((40, 60, 100)), rng.choice((0, 50, 100)), rng.choice((60, 90, 130)), rng.randrange(10 ** 6), 1 + i % 3))
        elif kind == 1:
            out.append(md.generate_grid(rng.choice((2, 3)), rng.choice((2, 3)), rng.choice((40, 70, 100)), rng.choice((1, 2, 3)), rng.choice((1, 2, 4)), 1 + i % 3, rng.randrange(10 ** 6)))
        else:
            out.append(md.generate_pairs(rng.randint(3, 6), rng.choice((40, 60, 90)), 1 + i % 3, rng.randrange(10 ** 6), rng.choice((1, 2, 3)), rng.choice((1, 2, 3))))
    return out


def _scale_caps(mcf, c):
    arcs = tuple((u, v, cap * c, cost, kind) for u, v, cap, cost, kind in mcf.net.arcs)
    ub = tuple(tuple(x if x >= BIG else x * c for x in row) for row in mcf.ub)
    return dataclasses.replace(mcf, net=dataclasses.replace(mcf.net, arcs=arcs), ub=ub)


def _scale_costs(mcf, c):
    arcs = tuple((u, v, cap, cost * c, kind) for u, v, cap, cost, kind in mcf.net.arcs)
    return dataclasses.replace(mcf, net=dataclasses.replace(mcf.net, arcs=arcs))


def _relabel(mcf, perm):
    n = mcf.net.n
    names, labels, pos = [None] * n, [None] * n, [None] * n
    for i in range(n):
        names[perm[i]], labels[perm[i]], pos[perm[i]] = mcf.net.names[i], mcf.net.labels[i], mcf.net.pos[i]
    arcs = tuple((perm[u], perm[v], cap, cost, kind) for u, v, cap, cost, kind in mcf.net.arcs)
    net = sc.Net(tuple(names), tuple(labels), tuple(pos), arcs, perm[mcf.net.s], perm[mcf.net.t], mcf.net.logistic)
    return dataclasses.replace(mcf, net=net)


def _permute_goods(mcf, order):
    return dataclasses.replace(mcf, names=tuple(mcf.names[k] for k in order), factors=tuple(mcf.factors[k] for k in order), ub=tuple(mcf.ub[k] for k in order))


def _opt(mcf):
    return el.solve_lp(mcf).total_delivered


def _utilisations(mcf, res, budget=None):
    """Auslastung aller Packungszeilen des skalierten Flusses: gemeinsame Kapazität, Gut-Obergrenzen, Budget."""
    util = []
    for e, (_, _, cap, _, _) in enumerate(mcf.net.arcs):
        if mcf.joint[e] and cap > 0:
            util.append(res.x[:, e].sum() / cap)
    for k in range(mcf.K):
        for e in range(mcf.m):
            if 0 < mcf.ub[k][e] < BIG:
                util.append(res.x[k, e] / mcf.ub[k][e])
    if budget is not None:
        util.append(sum(mcf.cost(k, e) * res.x[k, e] for k in range(mcf.K) for e in range(mcf.m) if not mcf.reward[e]) / budget)
    return util


@pytest.mark.parametrize("method", ["gk", "fleischer"])
def test_scaling_capacities_by_a_power_of_two_gives_the_same_run_and_scaled_amounts(method):
    checked = 0
    for i, mcf in enumerate(_instances(4201, 16)):
        opt = _opt(mcf)
        if opt < 1e-9:
            continue
        lp = el.solve_lp(mcf)
        budget = 0.75 * lp.cost if i % 2 and lp.cost > 1e-9 else None
        for c in (2, 4):
            a = alg.garg_koenemann(mcf, 0.5, method, budget=budget)
            b = alg.garg_koenemann(_scale_caps(mcf, c), 0.5, method, budget=None if budget is None else budget * c)
            assert (a.rows, a.pushes, a.calls, a.columns) == (b.rows, b.pushes, b.calls, b.columns)
            assert b.primal == pytest.approx(c * a.primal, rel=1e-9) and b.dual == pytest.approx(c * a.dual, rel=1e-9)
            assert b.delivered == pytest.approx(tuple(c * d for d in a.delivered), rel=1e-9, abs=1e-12)
        checked += 1
    assert checked >= 15


@pytest.mark.parametrize("method", ["gk", "fleischer"])
def test_scaling_costs_and_budget_together_changes_nothing(method):
    checked = 0
    for mcf in _instances(4202, 16):
        lp = el.solve_lp(mcf)
        if lp.total_delivered < 1e-9 or lp.cost < 1e-9:
            continue
        budget = 0.6 * lp.cost
        for c in (2, 5):
            a = alg.garg_koenemann(mcf, 0.5, method, budget=budget)
            b = alg.garg_koenemann(_scale_costs(mcf, c), 0.5, method, budget=budget * c)
            assert (a.rows, a.pushes, a.calls, a.columns) == (b.rows, b.pushes, b.calls, b.columns)
            assert b.primal == pytest.approx(a.primal, rel=1e-9) and b.dual == pytest.approx(a.dual, rel=1e-9)
            assert b.cost == pytest.approx(c * a.cost, rel=1e-9)
        checked += 1
    assert checked >= 12


def test_weak_duality_for_arbitrary_positive_lengths():
    """OPT <= D(y) / alpha(y) für jede positive Längenfunktion y auf den Packungszeilen (alpha = kürzester Weg über alle Güter)."""
    rng = random.Random(4203)
    checked = 0
    for mcf in _instances(4203, 20):
        opt = _opt(mcf)
        joint_rows, ub_rows, b = alg.rows_of(mcf)
        adj = [alg.paths.adjacency(mcf, k) for k in range(mcf.K)]
        for _ in range(4):
            scale = 10 ** rng.uniform(-3, 2)
            yj = np.zeros(mcf.m)
            yu = np.zeros((mcf.K, mcf.m))
            D = 0.0
            for e, i in joint_rows.items():
                yj[e] = rng.random() * scale + 1e-6
                D += yj[e] * b[i]
            for (k, e), i in ub_rows.items():
                yu[k, e] = rng.random() * scale + 1e-6
                D += yu[k, e] * b[i]
            alpha = min(alg._dijkstra(mcf, adj[k], k, yj, yu, 0.0)[0] for k in range(mcf.K))
            if alpha == math.inf:
                assert opt < 1e-9
                continue
            assert opt <= D / alpha + 1e-7 * max(1.0, D / alpha)
            checked += 1
    assert checked >= 60


@pytest.mark.parametrize("method", ["gk", "fleischer"])
def test_relabelling_nodes_and_reordering_goods_keep_the_optimum_and_the_guarantee(method):
    rng = random.Random(4204)
    eps = 0.4
    band = 1 - (1 - eps) ** 2
    checked = 0
    for mcf in _instances(4204, 12):
        opt = _opt(mcf)
        if opt < 1e-9:
            continue
        perm = list(range(mcf.net.n))
        rng.shuffle(perm)
        order = list(range(mcf.K))
        rng.shuffle(order)
        variants = [mcf, _relabel(mcf, perm), _permute_goods(mcf, order), _permute_goods(_relabel(mcf, perm), order)]
        results = []
        for v in variants:
            assert _opt(v) == pytest.approx(opt, abs=TOL)
            r = alg.garg_koenemann(v, eps, method)
            assert (1 - eps) ** 2 * opt - TOL <= r.primal <= opt + TOL and r.dual >= opt - TOL
            results.append(r.primal)
        assert max(results) - min(results) <= band * opt + TOL
        checked += 1
    assert checked >= 10


@pytest.mark.parametrize("method", ["gk", "fleischer"])
def test_scaling_by_the_congestion_makes_exactly_one_row_full(method):
    checked = 0
    for i, mcf in enumerate(_instances(4205, 16)):
        lp = el.solve_lp(mcf)
        if lp.total_delivered < 1e-9:
            continue
        budget = 0.7 * lp.cost if i % 2 and lp.cost > 1e-9 else None
        r = alg.garg_koenemann(mcf, 0.5, method, budget=budget)
        util = _utilisations(mcf, r, budget)
        assert max(util) == pytest.approx(1.0, abs=1e-9) and all(u <= 1 + 1e-9 for u in util)
        assert r.congestion >= 1 - 1e-9                       # ungeskaliert war mindestens eine Zeile ausgelastet (D >= 1 beendet den Lauf erst nach Überlast)
        checked += 1
    assert checked >= 15


@pytest.mark.parametrize("eps", [0.5, 0.3, 0.15])
def test_single_good_against_networkx_max_flow(eps):
    rng = random.Random(4206)
    checked = 0
    for i in range(12):
        if i % 2:
            mcf = md.generate_mcf(rng.choice((2, 3)), rng.choice((2, 3)), rng.choice((3, 5)), rng.choice((40, 70, 100)), rng.choice((0, 100)), rng.choice((60, 100, 130)), rng.randrange(10 ** 6), 1)
        else:
            mcf = md.generate_grid(rng.choice((2, 3)), 2, rng.choice((50, 100)), rng.choice((1, 2, 3)), rng.choice((2, 4)), 1, rng.randrange(10 ** 6))
        g = nx.DiGraph()
        for e, (u, v, cap, _, _) in enumerate(mcf.net.arcs):
            c = min(cap if mcf.joint[e] else BIG, mcf.ub[0][e])
            if 0 < c < BIG:
                if g.has_edge(u, v):
                    g[u][v]["capacity"] += c
                else:
                    g.add_edge(u, v, capacity=c)
        F = nx.maximum_flow_value(g, mcf.net.s, mcf.net.t) if mcf.net.s in g and mcf.net.t in g else 0
        for method in ("gk", "fleischer"):
            r = alg.garg_koenemann(mcf, eps, method)
            if F == 0:
                assert r.primal == 0
                continue
            assert (1 - eps) ** 2 * F - TOL <= r.primal <= F + TOL and r.dual >= F - TOL
            checked += 1
    assert checked >= 20


def _single_path(arc_cap, demand, cost=1):
    """Ein Gut, ein Weg S -> Start -> Ziel -> T mit einer einzigen inneren Kante."""
    return md._terminals(3, [(0, 1, demand)], 1, [(2, 3, arc_cap, cost, sc.K_OTHER)], [(50, 96), (50, 4), (10, 50), (90, 50)], ["Quelle S", "Senke T", "A", "B"], ["S", "T", "A", "B"], "pairs")


@pytest.mark.parametrize("method", ["gk", "fleischer"])
@pytest.mark.parametrize("eps", [0.5, 0.2])
def test_edge_cases_no_path_one_path_and_demand_below_capacity(method, eps):
    # ein einziger Weg: jeder Schritt füllt den Engpass, die Auslastung ist die Zahl der Schritte, die Skalierung liefert genau die Engpasskapazität
    for cap, demand, expected in ((3, 5, 3), (3, 2, 2), (4, 4, 4), (1, 1, 1)):
        mcf = _single_path(cap, demand)
        r = alg.garg_koenemann(mcf, eps, method)
        assert r.primal == pytest.approx(expected, abs=1e-9) and r.dual >= expected - TOL
        assert r.gap <= 1 + 1e-9 and len(r.columns) == 1 and r.stopped == "D>=1"
        assert r.x[0, :].max() == pytest.approx(expected, abs=1e-9)
    # kein Weg: die Angebotskante des einzigen Guts hat die Obergrenze 0
    mcf = _single_path(3, 2)
    blocked = dataclasses.replace(mcf, ub=tuple(tuple(0 if i == 1 else x for i, x in enumerate(row)) for row in mcf.ub))
    r = alg.garg_koenemann(blocked, eps, method)
    assert r.primal == 0 and r.pushes == 0 and r.stopped == "kein Weg"

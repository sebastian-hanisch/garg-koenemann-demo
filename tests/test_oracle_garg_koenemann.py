"""Orakel: Garg-Könemann und Fleischer gegen unabhängige Rechenwege (schnell, < 10 s).
(1) Das Optimum der maximalen Lieferung (auch mit Kostenbudget) wird pfadbasiert mit scipy-linprog gelöst, über alle einfachen Wege eines eigenen Tiefensuche-Aufzählers, und mit dem
Kanten-LP der Demo verglichen; im Ein-Gut-Fall zusätzlich mit networkx-Max-Flow. (2) Der Fluss jedes Aufnahmepunkts wird mit einem eigenen Zulässigkeitstest geprüft, die untere Schranke
liegt nie über, die bewiesene obere nie unter dem Optimum, und die Theorie-Garantie (1 - eps)^2 gilt. (3) Eine unabhängige Neuimplementierung des Verfahrens (networkx-Dijkstra, Zeilen als
Wörterbuch) liefert dieselben Schritte, Aufrufe, Lieferung und Schranke."""

import math
import random

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linprog

import gk_algorithm as alg
import gk_edge_lp as el
import gk_model as md
from gk_lp import solve_max_delivery

BIG = 10 ** 6


def _instances():
    rng = random.Random(2024)
    for i in range(5):
        yield md.generate_mcf(*rng.choice([(2, 2, 3), (2, 3, 4), (3, 3, 6)]), rng.choice((40, 60, 100)), rng.choice((0, 50, 100)), rng.choice((60, 90, 130)), rng.randrange(10 ** 6), 1 + i % 3)
    for i in range(5):
        yield md.generate_grid(rng.choice((2, 3, 4)), rng.choice((2, 3)), rng.choice((30, 60, 100)), rng.choice((1, 2, 3)), rng.choice((1, 2, 4)), 1 + i % 3, rng.randrange(10 ** 6))
    for i in range(5):
        yield md.generate_pairs(rng.randint(3, 6), rng.choice((30, 50, 80)), 1 + i % 3, rng.randrange(10 ** 6), rng.choice((1, 2, 3)), rng.choice((1, 2, 3)))


def _paths(mcf, k):
    net = mcf.net
    out = {}
    for e, a in enumerate(net.arcs):
        if mcf.ub[k][e] > 0:
            out.setdefault(a[0], []).append(e)
    found = []

    def dfs(u, seen, edges):
        if u == net.t:
            found.append(tuple(edges))
            return
        for e in out.get(u, []):
            v = net.arcs[e][1]
            if v not in seen:
                seen.add(v)
                edges.append(e)
                dfs(v, seen, edges)
                edges.pop()
                seen.discard(v)
    dfs(net.s, {net.s}, [])
    return found


def _is_sink_arc(mcf, e):
    return mcf.net.arcs[e][1] == mcf.net.t


def _real_cost(mcf, k, edges):
    return sum(mcf.cost(k, e) for e in edges if not _is_sink_arc(mcf, e))


def _path_optimum(mcf, budget=None):
    """Maximale Lieferung als Pfad-LP mit scipy (alle einfachen Wege je Gut)."""
    cols = [(k, p) for k in range(mcf.K) for p in _paths(mcf, k)]
    if not cols:
        return 0.0
    rows, rhs = [], []
    for e, a in enumerate(mcf.net.arcs):
        if mcf.joint[e] and any(e in p for _, p in cols):
            rows.append([1.0 if e in p else 0.0 for _, p in cols])
            rhs.append(a[2])
    for k in range(mcf.K):
        for e in range(mcf.m):
            if mcf.ub[k][e] < BIG and any(kk == k and e in p for kk, p in cols):
                rows.append([1.0 if (kk == k and e in p) else 0.0 for kk, p in cols])
                rhs.append(mcf.ub[k][e])
    if budget is not None:
        rows.append([_real_cost(mcf, k, p) for k, p in cols])
        rhs.append(budget)
    res = linprog(-np.ones(len(cols)), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=(0, None), method="highs")
    assert res.status == 0
    return -res.fun


def _single_good_max_flow(mcf):
    g = nx.DiGraph()
    for e, a in enumerate(mcf.net.arcs):
        cap = min(a[2] if mcf.joint[e] else BIG, mcf.ub[0][e])
        if cap >= BIG:
            continue
        if g.has_edge(a[0], a[1]):
            g[a[0]][a[1]]["capacity"] += cap
        else:
            g.add_edge(a[0], a[1], capacity=cap)
    return nx.maximum_flow_value(g, mcf.net.s, mcf.net.t) if mcf.net.s in g and mcf.net.t in g else 0.0


def _assert_feasible(mcf, x, budget=None, tol=1e-6):
    net = mcf.net
    assert (x >= -tol).all()
    for k in range(mcf.K):
        for v in range(net.n):
            if v not in (net.s, net.t):
                inflow = sum(x[k, e] for e, a in enumerate(net.arcs) if a[1] == v)
                outflow = sum(x[k, e] for e, a in enumerate(net.arcs) if a[0] == v)
                assert abs(inflow - outflow) < tol
        assert all(x[k, e] <= mcf.ub[k][e] + tol for e in range(mcf.m) if mcf.ub[k][e] < BIG)
    for e, a in enumerate(net.arcs):
        if mcf.joint[e]:
            assert x[:, e].sum() <= a[2] + tol
    if budget is not None:
        assert sum(mcf.cost(k, e) * x[k, e] for k in range(mcf.K) for e in range(mcf.m) if not _is_sink_arc(mcf, e)) <= budget + tol


def _reimplementation(mcf, eps, method, budget=None):
    """Garg-Könemann/Fleischer noch einmal von Grund auf: Zeilen als Wörterbuch, networkx-Dijkstra statt eigener Heap-Suche."""
    net, K, m = mcf.net, mcf.K, mcf.m
    cap = {("j", e): float(a[2]) for e, a in enumerate(net.arcs) if mcf.joint[e] and a[2] > 0}
    cap.update({("u", k, e): float(mcf.ub[k][e]) for k in range(K) for e in range(m) if 0 < mcf.ub[k][e] < BIG})
    if budget is not None:
        cap[("b",)] = float(budget)
    rows = len(cap)
    delta = (1 + eps) / ((1 + eps) * rows) ** (1 / eps)
    y = {r: delta / b for r, b in cap.items()}

    def length(k, e):
        c = 0.0 if _is_sink_arc(mcf, e) else mcf.cost(k, e)
        return y.get(("j", e), 0.0) + y.get(("u", k, e), 0.0) + (y[("b",)] * c if budget is not None else 0.0)

    def shortest(k):
        g = nx.DiGraph()
        for e, a in enumerate(net.arcs):
            if mcf.ub[k][e] > 0:
                w = length(k, e)
                if not g.has_edge(a[0], a[1]) or g[a[0]][a[1]]["w"] > w:
                    g.add_edge(a[0], a[1], w=w, e=e)
        if net.s not in g or net.t not in g or not nx.has_path(g, net.s, net.t):
            return math.inf, None
        d, p = nx.single_source_dijkstra(g, net.s, net.t, weight="w")
        return d, [g[u][v]["e"] for u, v in zip(p, p[1:])]

    load = {r: 0.0 for r in cap}
    state = {"total": 0.0, "pushes": 0, "calls": 0, "best": math.inf}

    def potential():
        return sum(y[r] * cap[r] for r in cap)

    def push(k, edges):
        use = {}
        for e in edges:
            for r in (("j", e), ("u", k, e)):
                if r in cap:
                    use[r] = 1
        c = _real_cost(mcf, k, edges)
        if budget is not None and c > 0:
            use[("b",)] = c
        f = min(cap[r] / a for r, a in use.items())
        for r, a in use.items():
            load[r] += f * a
            y[r] *= 1 + eps * f * a / cap[r]
        state["total"] += f
        state["pushes"] += 1

    if method == "gk":
        while potential() < 1:
            best = []
            for k in range(K):
                d, edges = shortest(k)
                state["calls"] += 1
                best.append((d, k, edges))
            d, k, edges = min(best, key=lambda t: (t[0], t[1]))
            if edges is None:
                break
            state["best"] = min(state["best"], potential() / d)
            push(k, edges)
    else:
        alpha = min(shortest(k)[0] for k in range(K))
        state["calls"] += K
        if alpha != math.inf:
            state["best"] = min(state["best"], potential() / alpha)
            while potential() < 1:
                threshold, phase_min, checked = alpha * (1 + eps), math.inf, 0
                for k in range(K):
                    while potential() < 1:
                        d, edges = shortest(k)
                        state["calls"] += 1
                        if edges is None or d >= threshold:
                            phase_min, checked = min(phase_min, d), checked + 1
                            break
                        push(k, edges)
                if checked == K and phase_min != math.inf:
                    state["best"] = min(state["best"], potential() / phase_min)
                alpha = threshold
    congestion = max(load[r] / cap[r] for r in cap)
    return {"pushes": state["pushes"], "calls": state["calls"], "primal": state["total"] / congestion if state["total"] else 0.0, "dual": state["best"], "rows": rows, "delta": delta}


def test_the_demo_lp_optimum_agrees_with_the_path_lp_and_with_max_flow():
    for mcf in _instances():
        opt = _path_optimum(mcf)
        lp = el.solve_lp(mcf)
        assert lp.total_delivered == pytest.approx(opt, abs=1e-6)
        if mcf.K == 1:
            assert _single_good_max_flow(mcf) == pytest.approx(opt, abs=1e-6)
        if lp.cost > 1e-9:
            for frac in (1.0, 0.6):
                assert solve_max_delivery(mcf, frac * lp.cost)[0] == pytest.approx(_path_optimum(mcf, frac * lp.cost), abs=1e-6)


@pytest.mark.parametrize("method", ["gk", "fleischer"])
def test_flows_bounds_guarantee_and_counts_against_the_independent_optimum(method):
    rng = random.Random(7)
    for mcf in _instances():
        lp = el.solve_lp(mcf)
        for eps in (0.5, 0.3):
            frac = rng.choice([None, 1.0, 0.7])
            budget = frac * lp.cost if frac and lp.cost > 1e-9 else None
            opt = _path_optimum(mcf, budget)
            res = alg.garg_koenemann(mcf, eps, method, budget=budget)
            if opt < 1e-9:
                assert res.primal < 1e-9
                continue
            _assert_feasible(mcf, res.x, budget)
            for frame in res.frames:
                _assert_feasible(mcf, frame.x, budget)
                assert frame.primal <= opt + 1e-6
                assert frame.dual in (0.0, math.inf) or frame.dual >= opt - 1e-6
            assert res.primal <= opt + 1e-6 and res.dual >= opt - 1e-6
            assert res.primal_theory >= (1 - eps) ** 2 * opt - 1e-6
            assert res.primal == pytest.approx(sum(res.x[:, e].sum() for e in range(mcf.m) if _is_sink_arc(mcf, e)), abs=1e-6)
            if method == "gk":
                assert res.calls == mcf.K * res.pushes


@pytest.mark.parametrize("method", ["gk", "fleischer"])
def test_an_independent_reimplementation_takes_the_same_steps(method):
    rng = random.Random(11)
    for mcf in _instances():
        lp = el.solve_lp(mcf)
        for eps in (0.5, 0.3):
            frac = rng.choice([None, 0.7])
            budget = frac * lp.cost if frac and lp.cost > 1e-9 else None
            if _path_optimum(mcf, budget) < 1e-9:
                continue
            res = alg.garg_koenemann(mcf, eps, method, budget=budget)
            ref = _reimplementation(mcf, eps, method, budget)
            assert res.rows == ref["rows"] and res.delta == pytest.approx(ref["delta"], rel=1e-12)
            # Gleichstände zwischen gleich langen Wegen und der Rand von D = 1 (eine Phase mehr oder weniger) lassen die beiden Läufe leicht auseinanderlaufen; beide Schranken sind gültig
            assert abs(res.pushes - ref["pushes"]) <= 0.03 * ref["pushes"] + 1
            assert res.primal == pytest.approx(ref["primal"], rel=0.03)
            assert res.dual == pytest.approx(ref["dual"], rel=0.05)
            assert abs(res.calls - ref["calls"]) <= 0.03 * ref["calls"] + mcf.K

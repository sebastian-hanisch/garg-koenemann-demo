"""Kanten-LP für die maximale Lieferung, optional mit Kostenbudget: Gegenprobe und Vergleichsbasis des Garg-Könemann-Verfahrens (HiGHS über scipy).

max  Summe der Flüsse auf den Nachfragekanten   u.d.N.  Flusserhaltung je Gut, gemeinsame Kapazität, Gut-Obergrenzen, (optional) Summe cost(k, e) · x[k][e] <= Budget.
Ohne Budget ist das Ergebnis dieselbe maximale Lieferung wie im lexikographischen Kanten-LP der Vorgänger-Demos (dort zusätzlich die billigste unter den größten).
"""

import numpy as np
from scipy.optimize import linprog

import gk_edge_lp as el


def solve_max_delivery(mcf, budget=None):
    """Rückgabe (Lieferung, Kosten des Flusses, x (K, m)). Kostenbudget in Kosteneinheiten oder None."""
    K, m = mcf.K, mcf.m
    _, a_eq, b_eq, a_ub, b_ub, ub, joint = el._matrices(mcf)
    c = np.zeros(K * m)
    cost = np.zeros(K * m)
    for k in range(K):
        for e in range(m):
            if mcf.reward[e]:
                c[k * m + e] = -1.0
            else:
                cost[k * m + e] = mcf.cost(k, e)
    if budget is not None:
        from scipy.sparse import vstack, csr_matrix
        a_ub = vstack([a_ub, csr_matrix(cost.reshape(1, -1))]).tocsr()
        b_ub = np.append(b_ub, float(budget))
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=list(zip(np.zeros_like(ub), np.where(np.isinf(ub), None, ub))), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = np.asarray(res.x, dtype=float).reshape(K, m)
    x[np.abs(x) < 1e-9] = 0.0
    return float(-res.fun), float(cost @ res.x), x

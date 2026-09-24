"""Kennzahlen, Urteil und Experimente der Demo (eps gegen Güte und Schritte, Fleischer gegen Garg-Könemann, Größe gegen HiGHS und Column Generation, Kostenbudget, Negativkontrollen).
Rechnet mit ganzen Zahlen (Netze) und Gleitkommazahlen (Längen); das Vergleichs-LP löst HiGHS. Vergleiche mit Toleranz."""

import time
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from statistics import mean, median

import gk_algorithm as alg
import gk_cg as cg
import gk_constants as C
import gk_edge_lp as el
import gk_lp
import gk_model as md
import gk_paths as paths

TOL = 1e-6
NetParams = namedtuple("NetParams", "net k p d s density spread load seed gw gh gdensity gcap gdem gseed")
Opts = namedtuple("Opts", "eps method budget variant")
DEFAULT_OPTS = Opts(C.DEFAULT_EPS, C.DEFAULT_METHOD, C.DEFAULT_BUDGET, C.DEFAULT_VARIANT)
DEFAULT_PARAMS = NetParams(C.DEFAULT_NET, C.DEFAULT_K, C.DEFAULT_P, C.DEFAULT_D, C.DEFAULT_S, C.DEFAULT_DENSITY, C.DEFAULT_SPREAD, C.DEFAULT_LOAD, C.DEFAULT_SEED,
                           C.DEFAULT_GW, C.DEFAULT_GH, C.DEFAULT_GDENSITY, C.DEFAULT_GCAP, C.DEFAULT_GDEM, C.DEFAULT_GSEED)


def pct(numerator, denominator, digits=1):
    return round(100.0 * numerator / denominator, digits)


def normalise(params):
    """Feste Netze ignorieren alle Zufallsregler: gleiche Netze unter demselben Schlüssel (sonst würden sie mehrfach berechnet)."""
    if params.net in C.FIXED_NETS:
        return DEFAULT_PARAMS._replace(net=params.net, k=0)
    return params


def build(params):
    if params.net == "gap":
        return md.gap_net()
    if params.net == "preis":
        return md.preis_net()
    if params.net == "grid":
        return md.generate_grid(params.gw, params.gh, params.gdensity, params.gcap, params.gdem, params.k, params.gseed)
    return md.generate_mcf(params.p, params.d, params.s, params.density, params.spread, params.load, params.seed, params.k)


def with_seed(params, seed):
    return params._replace(gseed=seed) if params.net == "grid" else params._replace(seed=seed)


def run_gk(mcf, opts, budget_abs, **kw):
    return alg.garg_koenemann(mcf, opts.eps, opts.method, budget=budget_abs, update="add" if opts.variant == "add" else "mult", scale=opts.variant != "noscale",
                              max_pushes=kw.get("max_pushes", C.MAX_PUSHES))


@dataclass(frozen=True)
class Analysis:
    mcf: object
    lp: object             # lexikographisches Kanten-LP des Vorgängers (größte Lieferung, darunter billigste)
    opt: float             # Optimum der maximalen Lieferung (mit Kostenbudget, falls gesetzt)
    opt_cost: float        # Kosten des LP-Optimums mit Budget (bzw. ohne)
    budget_abs: float      # Kostenbudget in Kosteneinheiten oder None
    gk: object             # GKResult
    opts: Opts


def reference(mcf, budget_frac):
    """(lexikographisches LP, Optimum der maximalen Lieferung, dessen Kosten, absolutes Budget)."""
    lp = el.solve_lp(mcf)
    b = budget_frac * lp.cost if budget_frac else None
    opt, cost, _ = gk_lp.solve_max_delivery(mcf, b)
    return lp, opt, cost, b


@lru_cache(maxsize=64)
def analyse(params, opts=DEFAULT_OPTS):
    mcf = build(params)
    lp, opt, opt_cost, b = reference(mcf, opts.budget)
    return Analysis(mcf, lp, opt, opt_cost, b, run_gk(mcf, opts, b), opts)


def pushes_bound(result):
    """Beweisbare Obergrenze der Schiebeschritte: jeder Schritt vervielfacht die Länge der Engpasszeile mit 1 + eps, und jede Zeile kann höchstens log_{1+eps}((1 + eps) / delta) mal wachsen, bevor D >= 1."""
    return result.rows * result.theory_scale


def verdict(a):
    """(Stufe, Code, Daten): 'ok' = zulässig, Güte bewiesen; 'noscale' = ohne Skalierung ist der Fluss überlastet; 'bad' = Güte weit unter der Garantie (additive Längen); 'nothing' = nichts lieferbar."""
    mcf, g, opt = a.mcf, a.gk, a.opt
    o = opt if opt > TOL else float("nan")
    data = {
        "K": mcf.K, "m": mcf.m, "demand": mcf.total_demand(), "opt": opt, "opt_cost": a.opt_cost, "lp_delivered": a.lp.total_delivered, "lp_cost": a.lp.cost, "budget": a.budget_abs,
        "primal": g.primal, "dual": g.dual, "cost": g.cost, "ratio": g.primal / o if opt > TOL else 1.0, "certified": g.gap, "guarantee": (1 - g.eps) ** 2,
        "theory_ratio": g.primal_theory / o if opt > TOL else 1.0, "pushes": g.pushes, "calls": g.calls, "phases": g.phases, "rows": g.rows, "bound": pushes_bound(g),
        "congestion": g.congestion, "columns": len(g.columns), "eps": g.eps, "stopped": g.stopped, "dual_valid": g.dual >= opt - 1e-6 or g.dual == 0.0,
        "cost_ratio": g.cost / a.lp.cost if a.lp.cost > TOL else 1.0, "delivered": g.delivered,
    }
    if opt < TOL:
        return "warning", "nothing", data
    if a.opts.variant == "noscale":
        return "error", "noscale", data
    if a.opts.variant == "add":
        return "warning", "bad", data
    return "success", "ok", data


# --- Verteilungen über feste Netze ---------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _runs(params, opts, seeds):
    out = []
    for seed in seeds:
        mcf = build(with_seed(params, seed))
        lp, opt, opt_cost, b = reference(mcf, opts.budget)
        out.append((mcf, lp, opt, opt_cost, run_gk(mcf, opts, b)))
    return out


def _row(t, opts=DEFAULT_OPTS):
    mcf, lp, opt, opt_cost, g = t
    o = opt if opt > TOL else float("nan")
    return {"ratio": g.primal / o if opt > TOL else 1.0, "certified": g.gap, "theory_ratio": g.primal_theory / o if opt > TOL else 1.0, "pushes": g.pushes, "calls": g.calls,
            "bound": pushes_bound(g), "congestion": g.congestion, "opt": opt, "primal": g.primal, "dual_valid": g.dual >= opt - 1e-6 or g.dual == 0.0, "cost": g.cost, "lp_cost": lp.cost,
            "feasible": opts.variant != "noscale" or g.congestion <= 1.0 + 1e-9}


def distribution(params, opts=DEFAULT_OPTS, seeds=C.QUALITY_SEEDS):
    """Verteilung über feste Netze mit den Einstellungen des Nutzers (nur der Seed wechselt)."""
    params = normalise(params)
    rows = [_row(t, opts) for t in _runs(params, opts, seeds)]
    cols = {key: [r[key] for r in rows] for key in rows[0]}
    n = len(rows)
    return {"n_seeds": n, "cols": cols, "ratio_mean": mean(cols["ratio"]), "ratio_min": min(cols["ratio"]), "certified_mean": mean(cols["certified"]), "certified_min": min(cols["certified"]),
            "theory_mean": mean(cols["theory_ratio"]), "theory_min": min(cols["theory_ratio"]), "pushes_mean": mean(cols["pushes"]), "calls_mean": mean(cols["calls"]), "bound_mean": mean(cols["bound"]),
            "congestion_mean": mean(cols["congestion"]), "share_dual_valid": sum(cols["dual_valid"]) / n}


# --- Experimente -----------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def eps_table(params, budget=0.0, seeds=C.EPS_SEEDS):
    """eps gegen tatsächliche Güte, Garantie, Schritte und Orakelaufrufe - für beide Verfahren, Mittel über feste Netze mit den Netzeinstellungen des Nutzers."""
    params = normalise(params)
    rows = []
    for eps in C.EPSS:
        for method in ("gk", "fleischer"):
            o = Opts(eps, method, budget, "mult")
            rs = [_row(t, o) for t in _runs(params, o, seeds)]
            rows.append({"eps": eps, "method": method, "ratio": mean(r["ratio"] for r in rs), "ratio_min": min(r["ratio"] for r in rs), "theory": mean(r["theory_ratio"] for r in rs),
                         "guarantee": (1 - eps) ** 2, "certified": mean(r["certified"] for r in rs), "pushes": mean(r["pushes"] for r in rs), "calls": mean(r["calls"] for r in rs),
                         "bound": mean(r["bound"] for r in rs), "feasible": all(r["feasible"] for r in rs), "dual_valid": all(r["dual_valid"] for r in rs)})
    return rows


@lru_cache(maxsize=8)
def control_table(params, seeds=C.EPS_SEEDS):
    """Negativkontrollen (eps = 0,3, Garg-Könemann): wie beschrieben, additive Längen, ohne Skalierung am Ende."""
    params = normalise(params)
    rows = []
    for variant in C.VARIANTS:
        o = Opts(0.3, "gk", 0.0, variant)
        rs = [_row(t, o) for t in _runs(params, o, seeds)]
        rows.append({"variant": variant, "ratio": mean(r["ratio"] for r in rs), "ratio_min": min(r["ratio"] for r in rs), "congestion": mean(r["congestion"] for r in rs),
                     "pushes": mean(r["pushes"] for r in rs), "feasible": sum(r["feasible"] for r in rs) / len(rs)})
    return rows


@lru_cache(maxsize=4)
def budget_table(params, eps=0.2):
    """Kompromiss Lieferung/Kosten für das gezeigte Netz: Kostenbudget 100 / 90 / 75 / 50 % der Kosten des LP-Optimums, Garg-Könemann gegen das LP mit Budgetzeile."""
    params = normalise(params)
    mcf = build(params)
    lp = el.solve_lp(mcf)
    rows = []
    for frac in (0.0, 1.0, 0.9, 0.75, 0.5):
        b = frac * lp.cost if frac else None
        opt, opt_cost, _ = gk_lp.solve_max_delivery(mcf, b)
        g = alg.garg_koenemann(mcf, eps, "gk", budget=b)
        rows.append({"frac": frac, "budget": b, "opt": opt, "opt_cost": opt_cost, "primal": g.primal, "cost": g.cost, "dual": g.dual, "ratio": g.primal / opt if opt > TOL else 1.0, "pushes": g.pushes,
                     "feasible": g.cost <= (b if b else float("inf")) + 1e-6})
    return rows


@lru_cache(maxsize=2)
def size_table(sizes=C.SCALE_SIZES, K=5, seeds=C.SIZE_SEEDS, eps=0.3):
    """Volle Gitter wachsender Größe (Kapazität bis 2, Menge bis 5, fünf Güter): Fleischer (eps = 0,3) gegen Kanten-LP und Column Generation - Zeilen, Schritte, Orakelaufrufe, Sekunden (nur zur Information), Güte."""
    rows = []
    for w, h in sizes:
        per = []
        for seed in seeds:
            mcf = md.generate_grid(w, h, 100, 2, 5, K, seed)
            t = time.perf_counter()
            lp = el.solve_lp(mcf)
            t_lp = time.perf_counter() - t
            t = time.perf_counter()
            r = cg.column_generation(mcf)
            t_cg = time.perf_counter() - t
            t = time.perf_counter()
            g = alg.garg_koenemann(mcf, eps, "fleischer")
            t_gk = time.perf_counter() - t
            per.append((g.rows, g.pushes, g.calls, r.n_rounds * K, lp.total_delivered, g.primal, t_lp, t_cg, t_gk, g.gap))
        rows.append({"size": (w, h), "rows": per[0][0], "pushes": mean(p[1] for p in per), "calls": mean(p[2] for p in per), "cg_calls": mean(p[3] for p in per),
                     "ratio": mean(p[5] / p[4] for p in per), "t_lp": median(p[6] for p in per), "t_cg": median(p[7] for p in per), "t_gk": median(p[8] for p in per), "certified": mean(p[9] for p in per)})
    return rows


def frame_ratio(frame, opt):
    """(bewiesene Güte, tatsächliche Güte) eines Aufnahmepunkts; bewiesen ist erst, wenn eine obere Schranke bekannt ist."""
    cert = frame.primal / frame.dual if frame.dual not in (0.0, float("inf")) and frame.dual > TOL else None
    return cert, frame.primal / opt if opt > TOL else 1.0


# --- Anzeige-Helfer ---------------------------------------------------------------------------------------------------------------------------

def path_label(mcf, edges):
    """Knotenfolge eines Pfades ohne S und T, mit den Kurzbeschriftungen der Karte (im Streckennetz die Koordinaten)."""
    net = mcf.net
    out = []
    for v in paths.path_nodes(mcf, edges):
        if v in (net.s, net.t):
            continue
        out.append(net.labels[v] or net.names[v].replace("Knoten ", "").replace(" ", ""))
    return " → ".join(out)

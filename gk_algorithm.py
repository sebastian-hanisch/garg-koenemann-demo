"""Garg-Könemann und Fleischer für den Mehrgüterfluss: (1 - eps)-Näherung der maximalen Lieferung nur mit Kürzeste-Wege-Orakeln, ohne LP-Löser.

**Packungs-LP.** Variable x_p >= 0 je Pfad p (eines Guts), Ziel: maximiere die Summe aller x_p (jeder Pfad endet auf genau einer Nachfragekante, also ist das die Lieferung).
Zeilen (alle "<="): gemeinsame Kapazität je gemeinsamer Kante (Summe aller Pfade über die Kante <= u_e), Gut-Obergrenze je (Gut, Kante) mit endlicher Grenze, optional eine Kostenzeile
(Summe Kosten des Pfades mal x_p <= Budget). Das sind dieselben Zeilen wie im Master der Column Generation (Stück 8); dort löst ein LP-Löser den Master, hier gibt es keinen.

**Garg-Könemann.** Längen y_r = delta / b_r je Zeile. Wiederhole, solange D(y) = sum_r y_r b_r < 1: kürzester Weg (Kantenlänge = Summe der Zeilenlängen, die er berührt) über alle Güter, schiebe die
Engpassmenge f = min_r b_r / a_rp auf ihn, multipliziere die Längen der berührten Zeilen mit (1 + eps · f · a_rp / b_r). Am Ende ist der Fluss zu groß und wird skaliert.
**Fleischer.** Dasselbe, aber die Orakelaufrufe werden gespart: in Phasen mit Schwelle alpha·(1 + eps) wird je Gut so lange auf seinem kürzesten Weg geschoben, wie dieser unter der Schwelle liegt;
erst danach steigt alpha um den Faktor 1 + eps.

**Zertifikat.** Für beliebige Längen y > 0 ist y / alpha(y) dual zulässig (alpha = kürzester Weg über alle Güter), also gilt OPT <= D(y) / alpha(y). Das ist eine beweisbare obere Schranke für jeden
Zeitpunkt, unabhängig von der Theorie-Konstante; der zulässig skalierte Fluss ist die untere Schranke.

Optionen für die Negativkontrollen: `update` ("mult" wie im Verfahren oder "add": additive statt multiplikative Längenerhöhung), `scale` (Skalierung am Ende oder nicht).
"""

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

import gk_cg as cg
import gk_paths as paths

BIG = 10 ** 6
TOL = 1e-9


@dataclass
class Frame:
    """Zustand an einem Aufnahmepunkt: nach `push` Schiebeschritten."""
    push: int
    calls: int                        # Orakelaufrufe (Dijkstra) bis hierher
    primal: float                     # zulässig skalierte Lieferung
    dual: float                       # beste bisher bewiesene obere Schranke
    congestion: float                 # größte Auslastung Last/Grenze der ungeskalierten Summe
    x: np.ndarray                     # (K, m) zulässig skalierter Kantenfluss
    y: np.ndarray                     # (m,) Länge je gemeinsamer Kante (0 sonst)
    path: tuple = None                # zuletzt geschobener Pfad (Gut, Kanten)
    amount: float = 0.0               # dessen Menge


@dataclass
class GKResult:
    eps: float
    method: str
    budget: float                     # Kostenbudget (None: keine Kostenzeile)
    rows: int                         # Zahl der Zeilen R
    delta: float
    theory_scale: float               # log_{1+eps}((1 + eps) / delta): Skalierung nach der Theorie
    columns: list                     # (Gut, Kanten) der geschobenen Pfade
    amounts: np.ndarray               # Menge je Pfad, zulässig skaliert (durch die Auslastung)
    raw: np.ndarray                   # Menge je Pfad, ungeskaliert
    primal: float                     # zulässig skalierte Lieferung (Skalierung durch die Auslastung)
    primal_theory: float              # Lieferung mit der Skalierung der Theorie
    dual: float                       # beste obere Schranke
    congestion: float
    pushes: int
    calls: int
    phases: int
    frames: list = field(default_factory=list)
    cost: float = 0.0                 # Kosten des skalierten Flusses
    delivered: tuple = ()
    x: np.ndarray = None              # (K, m) skalierter Kantenfluss
    stopped: str = "D>=1"

    @property
    def gap(self):
        """Bewiesene Güte: primal / dual (1 = optimal)."""
        return self.primal / self.dual if self.dual > TOL else 1.0


def rows_of(mcf):
    """Zeilen der Packungsform: gemeinsame Kanten und (Gut, Kante) mit endlicher Grenze (wie im Master der Column Generation)."""
    m = cg.Master(mcf)
    return m.joint_rows, m.ub_rows, m.b


class _Rows:
    def __init__(self, mcf, budget):
        self.mcf = mcf
        self.joint_rows, self.ub_rows, self.b = rows_of(mcf)
        self.budget = budget
        self.R = len(self.b) + (1 if budget is not None else 0)
        self.nb = len(self.b)
        self.bcap = float(budget) if budget is not None else None


def _dijkstra(mcf, adj, k, yj, yu, yb, calls_counter=None):
    """Kürzester Weg für Gut k mit Kantenlänge yj[e] (gemeinsam) + yu[k, e] (Gut-Obergrenze) + yb · Kosten. Rückgabe (Länge, Kantentupel)."""
    net = mcf.net
    dist = {net.s: 0.0}
    prev = {}
    heap = [(0.0, net.s)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == net.t:
            break
        for e, v in adj[u]:
            nd = d + yj[e] + yu[k, e] + (yb * mcf.cost(k, e) if yb else 0.0)
            if nd < dist.get(v, float("inf")) - 1e-15:
                dist[v] = nd
                prev[v] = (u, e)
                heapq.heappush(heap, (nd, v))
    if net.t not in dist:
        return float("inf"), None
    edges, v = [], net.t
    while v != net.s:
        u, e = prev[v]
        edges.append(e)
        v = u
    edges.reverse()
    return dist[net.t], tuple(edges)


def garg_koenemann(mcf, eps=0.2, method="gk", budget=None, update="mult", scale=True, max_pushes=400000, n_frames=60):
    """Näherung der maximalen Lieferung. `method`: "gk" (Garg-Könemann) oder "fleischer". `budget`: Obergrenze der Kosten (Zeile) oder None."""
    K, m = mcf.K, mcf.m
    rows = _Rows(mcf, budget)
    R = rows.R
    delta = (1 + eps) * ((1 + eps) * R) ** (-1.0 / eps)
    theory_scale = math.log((1 + eps) / delta) / math.log(1 + eps)
    joint_rows, ub_rows, b = rows.joint_rows, rows.ub_rows, rows.b
    yj = np.zeros(m)
    yu = np.zeros((K, m))
    for e, i in joint_rows.items():
        yj[e] = delta / b[i]
    for (k, e), i in ub_rows.items():
        yu[k, e] = delta / b[i]
    yb = delta / budget if budget is not None else 0.0
    bj = np.zeros(m)
    for e, i in joint_rows.items():
        bj[e] = b[i]
    bu = np.zeros((K, m))
    for (k, e), i in ub_rows.items():
        bu[k, e] = b[i]
    adj = [paths.adjacency(mcf, k) for k in range(K)]

    def D():
        return float(np.sum(yj[bj > 0] * bj[bj > 0]) + np.sum(yu[bu > 0] * bu[bu > 0]) + (yb * budget if budget is not None else 0.0))

    cols, key_index, raw = [], {}, []
    load_j = np.zeros(m)
    load_u = np.zeros((K, m))
    load_b = 0.0
    calls = pushes = phases = 0
    best_dual = float("inf")
    frames = []
    frame_at = _checkpoints(n_frames)

    def congestion():
        c = 0.0
        pos = bj > 0
        if pos.any():
            c = max(c, float(np.max(load_j[pos] / bj[pos])))
        posu = bu > 0
        if posu.any():
            c = max(c, float(np.max(load_u[posu] / bu[posu])))
        if budget is not None:
            c = max(c, load_b / budget)
        return c

    def snapshot(path=None, amount=0.0):
        cong = congestion()
        total = float(sum(raw))
        s = max(cong, 1e-12)
        x = np.zeros((K, m))
        for (k, edges), a in zip(cols, raw):
            for e in edges:
                x[k, e] += a
        prim = total / s if total > 0 else 0.0
        frames.append(Frame(pushes, calls, prim, best_dual, cong, x / s, yj.copy(), path, amount))

    def push(k, edges):
        nonlocal pushes, load_b
        cost = paths.path_cost(mcf, k, edges)
        f = float("inf")
        for e in edges:
            if mcf.joint[e]:
                f = min(f, bj[e])
            if bu[k, e] > 0:
                f = min(f, bu[k, e])
        if budget is not None and cost > 0:
            f = min(f, budget / cost)
        key = (k, edges)
        if key not in key_index:
            key_index[key] = len(cols)
            cols.append(key)
            raw.append(0.0)
        raw[key_index[key]] += f
        for e in edges:
            if mcf.joint[e]:
                load_j[e] += f
                yj[e] = yj[e] * (1 + eps * f / bj[e]) if update == "mult" else yj[e] + eps * f / bj[e] / R
            if bu[k, e] > 0:
                load_u[k, e] += f
                yu[k, e] = yu[k, e] * (1 + eps * f / bu[k, e]) if update == "mult" else yu[k, e] + eps * f / bu[k, e] / R
        if budget is not None and cost > 0:
            load_b += f * cost
            yb_new = yb * (1 + eps * f * cost / budget) if update == "mult" else yb + eps * f * cost / budget / R
            return f, yb_new
        return f, yb

    stopped = "D>=1"

    def bound(alpha):
        nonlocal best_dual
        if alpha > TOL and alpha != float("inf"):
            best_dual = min(best_dual, D() / alpha)

    if method == "gk":
        while D() < 1.0:
            if pushes >= max_pushes:
                stopped = "max_pushes"
                break
            best = (float("inf"), None, -1)
            for k in range(K):
                d, edges = _dijkstra(mcf, adj[k], k, yj, yu, yb)
                calls += 1
                if edges is not None and d < best[0]:
                    best = (d, edges, k)
            if best[1] is None:
                stopped = "kein Weg"
                break
            bound(best[0])
            f, yb = push(best[2], best[1])
            pushes += 1
            if pushes in frame_at:
                snapshot((best[2], best[1]), f)
    else:
        alpha = float("inf")
        for k in range(K):
            d, edges = _dijkstra(mcf, adj[k], k, yj, yu, yb)
            calls += 1
            alpha = min(alpha, d)
        if alpha == float("inf"):
            stopped = "kein Weg"
        else:
            bound(alpha)
            while D() < 1.0 and stopped == "D>=1":
                phases += 1
                threshold = alpha * (1 + eps)
                phase_min = float("inf")
                checked = 0
                for k in range(K):
                    while D() < 1.0:
                        d, edges = _dijkstra(mcf, adj[k], k, yj, yu, yb)
                        calls += 1
                        if edges is None or d >= threshold:
                            phase_min = min(phase_min, d)
                            checked += 1
                            break
                        f, yb = push(k, edges)
                        pushes += 1
                        if pushes in frame_at:
                            snapshot((k, edges), f)
                        if pushes >= max_pushes:
                            stopped = "max_pushes"
                            break
                    if stopped != "D>=1":
                        break
                # Nur eine vollständige Phase beweist etwas: jeder Weg jedes Guts war bei seiner letzten Prüfung mindestens so lang wie phase_min, und Längen wachsen nur
                if checked == K and phase_min != float("inf"):
                    bound(phase_min)
                alpha = threshold
    cong = congestion()
    total = float(sum(raw))
    s = max(cong, 1e-12)
    amounts = np.array(raw) / s if raw else np.zeros(0)
    if not scale:
        amounts = np.array(raw)
    x = np.zeros((K, m))
    for (k, edges), a in zip(cols, amounts):
        for e in edges:
            x[k, e] += a
    primal = float(amounts.sum())
    delivered = tuple(float(sum(a for (k, edges), a in zip(cols, amounts) if k == kk)) for kk in range(K))
    cost = float(sum(a * paths.path_cost(mcf, k, edges) for (k, edges), a in zip(cols, amounts)))
    frames.append(Frame(pushes, calls, primal, best_dual if best_dual != float("inf") else 0.0, cong, x.copy(), yj.copy(), None, 0.0))
    return GKResult(eps, method, budget, R, delta, theory_scale, cols, amounts, np.array(raw), primal, total / theory_scale, best_dual if best_dual != float("inf") else 0.0, cong, pushes, calls, phases,
                    frames, cost, delivered, x, stopped)


def _checkpoints(n):
    """Aufnahmepunkte (Zahl der Schiebeschritte): die ersten acht einzeln, dann geometrisch wachsend."""
    pts = set(range(1, 9))
    t = 8.0
    factor = 1.28
    while len(pts) < n and t < 10 ** 7:
        t *= factor
        pts.add(int(t))
    return pts

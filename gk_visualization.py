"""Plotly-Abbildungen: Netz mit den Längen der multiplikativen Gewichte, Fortschritt und Güte je Aufnahmepunkt, Verteilung, eps-Tabelle, Aufrufe, Größe und Kostenbudget.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Kanten haben über unsichtbare Marker einen Hover-Text
(Plotly-Linien reagieren nur an ihren Stützpunkten)."""

from math import atan2, degrees, hypot

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gk_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _layout(fig, net, height, skip=()):
    xs = [p[0] for v, p in enumerate(net.pos) if v not in skip]
    ys = [p[1] for v, p in enumerate(net.pos) if v not in skip]
    pad = 9
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    return _base(fig, height)


def _curve(p0, p1, bulge, steps=8):
    """Punkte von p0 nach p1; mit `bulge` > 0 als flacher Bogen nach rechts (so trennen sich Vorwärts- und Rückkante). Dazu der Pfeilwinkel bei 65 %."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = hypot(dx, dy) or 1.0
    cx, cy = (x0 + x1) / 2 + bulge * length * dy / length, (y0 + y1) / 2 - bulge * length * dx / length
    ts = [k / steps for k in range(steps + 1)]
    xs = [(1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1 for t in ts]
    ys = [(1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1 for t in ts]
    t = 0.65
    tx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
    ty = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)
    ax = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1
    ay = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1
    return xs, ys, (ax, ay, degrees(atan2(tx, ty))), (xs[steps // 2], ys[steps // 2])


def _segments(curves):
    x, y = [], []
    for xs, ys, _, _ in curves:
        x += xs + [None]
        y += ys + [None]
    return x, y


def _lines(fig, curves, color, width, name, dash=None, showlegend=True):
    if not curves:
        return
    x, y = _segments(curves)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", name=name, showlegend=showlegend))


def _arrows(fig, curves, color, size=9):
    if not curves:
        return
    fig.add_trace(go.Scatter(x=[c[2][0] for c in curves], y=[c[2][1] for c in curves], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="arrow", size=size, color=color, angle=[c[2][2] for c in curves])))


def _hover_points(fig, net, entries):
    """Unsichtbare Marker entlang jeder Kante, damit der Hover-Text überall auf der Kante erscheint. entries: [(Kurve, Text)]"""
    x, y, text = [], [], []
    for curve, label in entries:
        xs, ys = curve[0], curve[1]
        for k in range(1, len(xs) - 1):
            x.append(xs[k]); y.append(ys[k]); text.append(label)
    if x:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=9, opacity=0), hovertext=text, hoverinfo="text", showlegend=False))


def _labels(fig, points):
    """points: [(x, y, Text)] - als Annotationen mit heller Hinterlegung, damit sie Kanten, Pfeile und Knotenbeschriftungen nicht unlesbar machen."""
    for x, y, text in points:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", font=dict(size=11, color="#111"), bgcolor="rgba(255,255,255,0.88)", borderpad=1)


def _arc_name(net, i):
    u, v = net.arcs[i][0], net.arcs[i][1]
    return f"{net.names[u]} → {net.names[v]}"


def _nodes(fig, net, reach=None):
    """Knoten: S und T als Quadrate, alle anderen als Kreise; mit `reach` grün (von S erreichbar) oder grau eingefärbt."""
    text_pos = {0: "top center", 1: "bottom center"}
    for kind, idx in (("Quelle/Senke", [net.s, net.t]), ("Knoten", [v for v in range(net.n) if v not in (net.s, net.t)])):
        colors = [C.COLORS["node"] if reach is None else (C.COLORS["reach"] if reach[v] else C.COLORS["unreach"]) for v in idx]
        pos = [text_pos.get(v, "top center" if net.pos[v][1] > 70 else ("bottom center" if net.pos[v][1] < 30 else "middle left")) for v in idx]
        if net.logistic and kind == "Knoten":
            pos = ["top center" if net.names[v].startswith("Werk") else "bottom center" if net.names[v].startswith("Filiale") else "middle left" for v in idx]
        fig.add_trace(go.Scatter(
            x=[net.pos[v][0] for v in idx], y=[net.pos[v][1] for v in idx], mode="markers+text", showlegend=False,
            text=[net.labels[v] for v in idx], textposition=pos, hovertext=[net.names[v] for v in idx], hoverinfo="text",
            marker=dict(symbol="square" if kind == "Quelle/Senke" else "circle", size=13 if kind == "Quelle/Senke" else 10, color=colors, line=dict(width=1.5, color="#333"))))


def _wscale(net):
    return max(c for _, _, c, _, _ in net.arcs)


def _width(amount, top, lo=1.0, hi=6.0):
    return lo + (hi - lo) * amount / top if top else lo


def _node_text_positions(net, idx):
    if net.logistic:
        return ["top center" if (v == 0 or net.names[v].startswith("Werk")) else "bottom center" if (v == 1 or net.names[v].startswith("Filiale")) else "middle left" for v in idx]
    return ["top center" if v == net.s else "bottom center" if v == net.t else "middle left" for v in idx]


TOL = 1e-6


def _shift(curve, dx, dy):
    xs, ys, (ax, ay, ang), (mx, my) = curve
    return [x + dx for x in xs], [y + dy for y in ys], (ax + dx, ay + dy, ang), (mx + dx, my + dy)


def _fmt(x):
    return f"{x:.1f}".replace(".", ",") if abs(x - round(x)) > TOL else f"{round(x)}"


def mcg_color(k):
    import gk_model as md
    return md.GOOD_COLORS[k % len(md.GOOD_COLORS)]


def _is_terminal(net, e):
    return net.arcs[e][0] == net.s or net.arcs[e][1] == net.t


def _factor_text(f):
    return f"×{f:.0f}" if f < 1000 else f"×{f / 1000:.0f}k" if f < 1e6 else f"×{f:.0e}".replace("e+0", "e").replace("e+", "e")


def build_gk(mcf, frame, y0, height=460):
    """Netz an einem Aufnahmepunkt: je Gut eine Farbe (Breite ~ zulässig skalierter Fluss), orange Unterlage mit wachsender Stärke nach dem Längenfaktor y / y0 der gemeinsamen Kanten (ab ×10; drei Stufen nach dem Logarithmus des Faktors, bezogen auf den größten)
    (die multiplikativen Gewichte: heiße Kanten sind lang), die fünf längsten Kanten mit ihrem Faktor beschriftet, schwarz gestrichelt der zuletzt geschobene Pfad. Im Streckennetz sind die Kanten von S
    und zu T ausgeblendet; Start (Raute) und Ziel (Stern) jedes Guts sind markiert."""
    net = mcf.net
    grid = mcf.layout == "grid"
    fig = go.Figure()
    x = frame.x
    K = mcf.K
    caps = [a[2] for a in net.arcs]
    top = max((caps[e] for e in range(mcf.m) if mcf.joint[e]), default=1)
    bulge = 0.0 if net.logistic else 0.12
    factor = np.ones(mcf.m)
    for e in range(mcf.m):
        if mcf.joint[e] and y0[e] > 0:
            factor[e] = frame.y[e] / y0[e]
    lf = np.log10(np.maximum(factor, 1.0))
    lmax = max(float(lf.max()), 1.0)
    groups, faint, hover, labels = {}, [], [], []
    bins = {1: [], 2: [], 3: []}
    small = mcf.m <= 30
    hot = sorted((e for e in range(mcf.m) if mcf.joint[e] and factor[e] >= 10 and not (grid and _is_terminal(net, e))), key=lambda e: -factor[e])[:5]
    for e, (u, v, cap, cost, kind) in enumerate(net.arcs):
        if grid and _is_terminal(net, e):
            continue
        base = _curve(net.pos[u], net.pos[v], bulge)
        load = float(x[:, e].sum())
        parts = ", ".join(f"{mcf.names[k]} {_fmt(x[k, e])}" for k in range(K) if x[k, e] > TOL)
        ftxt = f", Länge ×{factor[e]:.0f}" if mcf.joint[e] and factor[e] >= 1.5 else ""
        hover.append((base, f"{_arc_name(net, e)}: Summe {_fmt(load)}" + (f" von {cap}" if mcf.joint[e] else "") + (f" ({parts})" if parts else "") + ftxt))
        faint.append(base)
        if mcf.joint[e] and factor[e] >= 10:
            bins[1 if lf[e] < 0.4 * lmax else 2 if lf[e] < 0.75 * lmax else 3].append(base)
        dx0, dy0 = net.pos[v][0] - net.pos[u][0], net.pos[v][1] - net.pos[u][1]
        length = (dx0 ** 2 + dy0 ** 2) ** 0.5 or 1.0
        nx, ny = -dy0 / length, dx0 / length
        active = [k for k in range(K) if x[k, e] > TOL]
        for j, k in enumerate(active):
            off = (j - (len(active) - 1) / 2) * 1.1
            f = x[k, e]
            frac = abs(f - round(f)) > TOL
            groups.setdefault((k, max(1, round(1.5 + 5 * f / top)), frac), []).append(_shift(base, nx * off, ny * off))
        if e in hot:
            labels.append((base[0][3] + 1.5, base[1][3], _factor_text(factor[e])))
        elif small and load > TOL:
            labels.append((base[0][3] + 1.5, base[1][3], f"{_fmt(load)}" + (f"/{cap}" if mcf.joint[e] else "")))
    for level, name, alpha in ((1, "Länge gewachsen (unteres Drittel, logarithmisch)", 0.2), (2, "Länge gewachsen (mittleres Drittel)", 0.4), (3, "Länge gewachsen (oberes Drittel)", 0.65)):
        _lines(fig, bins[level], f"rgba(255,127,14,{alpha})", 12, name)
    _lines(fig, faint, C.COLORS["faint"], 1.0, "Kante", showlegend=False)
    for (k, w, frac), curves in sorted(groups.items()):
        _lines(fig, curves, mcg_color(k), w, mcf.names[k], dash="dot" if frac else None, showlegend=False)
    for k in range(K):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=mcg_color(k), width=4), name=mcf.names[k]))
    if frame.path is not None:
        k, edges = frame.path
        curves = []
        for e in edges:
            if grid and _is_terminal(net, e):
                continue
            u, v = net.arcs[e][0], net.arcs[e][1]
            curves.append(_curve(net.pos[u], net.pos[v], bulge))
        _lines(fig, curves, C.COLORS["new"], 2.5, "zuletzt geschobener Pfad", dash="dash")
    if net.m <= 80:
        _arrows(fig, faint, "rgba(60,60,60,0.55)", 7)
    _hover_points(fig, net, hover)
    _labels(fig, labels)
    idx = [v for v in range(net.n) if not (grid and v in (net.s, net.t))]
    fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in idx], y=[net.pos[v][1] for v in idx], mode="markers+text", showlegend=False, text=[net.labels[v] for v in idx], textposition=_node_text_positions(net, idx),
                             hovertext=[net.names[v] for v in idx], hoverinfo="text", marker=dict(symbol=["square" if v in (net.s, net.t) else "circle" for v in idx], size=[13 if v in (net.s, net.t) else (7 if grid else 10) for v in idx],
                                                                                              color=C.COLORS["node"], line=dict(width=1.5, color="#333"))))
    if grid:
        for k in range(K):
            o = [net.arcs[e][1] for e in range(mcf.m) if net.arcs[e][0] == net.s and mcf.ub[k][e] > 0]
            t = [net.arcs[e][0] for e in range(mcf.m) if net.arcs[e][1] == net.t and mcf.ub[k][e] > 0]
            dem = mcf.demand(k)
            fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in o], y=[net.pos[v][1] for v in o], mode="markers+text", showlegend=False, text=[f"{k + 1}" for _ in o], textposition="top center",
                                     hovertext=[f"Start {mcf.names[k]} (Menge {dem})" for _ in o], hoverinfo="text", marker=dict(symbol="diamond", size=14, color=mcg_color(k), line=dict(width=1.5, color="#333"))))
            fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in t], y=[net.pos[v][1] for v in t], mode="markers+text", showlegend=False, text=[f"{k + 1}" for _ in t], textposition="bottom center",
                                     hovertext=[f"Ziel {mcf.names[k]} (Menge {dem})" for _ in t], hoverinfo="text", marker=dict(symbol="star", size=14, color=mcg_color(k), line=dict(width=1.5, color="#333"))))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="diamond", size=10, color="#777"), name="Start des Guts (Nummer)"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="star", size=10, color="#777"), name="Ziel des Guts"))
    fig = _layout(fig, net, height, skip=(net.s, net.t) if grid else ())
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=90 if grid else 70), legend=dict(orientation="h", y=-0.12))
    return fig


def _finite(frames, key):
    return [(f.push, getattr(f, key)) for f in frames if getattr(f, key) not in (0.0, float("inf")) and getattr(f, key) == getattr(f, key)]


def build_progress(frames, current, opt, height=300):
    """Lieferung gegen die Zahl der Schiebeschritte (logarithmisch): zulässig skalierter Fluss (untere Schranke), bewiesene obere Schranke (Längen), gestrichelt das LP-Optimum."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[max(f.push, 1) for f in frames], y=[f.primal for f in frames], mode="lines+markers", name="zulässiger Fluss", line=dict(color=C.COLORS["flow"], width=2)))
    dual = _finite(frames, "dual")
    if dual:
        fig.add_trace(go.Scatter(x=[max(p, 1) for p, _ in dual], y=[v for _, v in dual], mode="lines+markers", name="bewiesene obere Schranke", line=dict(color=C.COLORS["dual"], width=2)))
    fig.add_hline(y=opt, line=dict(color="#555", dash="dash"), annotation_text="LP-Optimum", annotation_position="bottom right")
    fig.add_vline(x=max(frames[current].push, 1), line=dict(color=C.COLORS["optimal"], dash="dot"))
    fig.update_xaxes(title="Schiebeschritte", type="log")
    fig.update_yaxes(title="Lieferung", rangemode="tozero")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.35), height=height + 40, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_quality(frames, current, opt, eps, height=260):
    """Güte gegen die Schiebeschritte: tatsächlich (Fluss / LP-Optimum) und bewiesen (Fluss / obere Schranke) gegen die Garantie (1 - eps)^2."""
    fig = go.Figure()
    xs = [max(f.push, 1) for f in frames]
    fig.add_trace(go.Scatter(x=xs, y=[100 * f.primal / opt if opt > TOL else 100 for f in frames], mode="lines+markers", name="tatsächlich", line=dict(color=C.COLORS["flow"], width=2)))
    proven = [(max(f.push, 1), 100 * f.primal / f.dual) for f in frames if f.dual not in (0.0, float("inf")) and f.dual > TOL]
    if proven:
        fig.add_trace(go.Scatter(x=[p for p, _ in proven], y=[v for _, v in proven], mode="lines+markers", name="bewiesen", line=dict(color=C.COLORS["dual"], width=2)))
    fig.add_hline(y=100 * (1 - eps) ** 2, line=dict(color=C.COLORS["optimal"], dash="dash"), annotation_text=f"Garantie (1 − ε)² = {100 * (1 - eps) ** 2:.0f} %".replace(".", ","), annotation_position="bottom right")
    fig.add_vline(x=max(frames[current].push, 1), line=dict(color=C.COLORS["optimal"], dash="dot"))
    fig.update_xaxes(title="Schiebeschritte", type="log")
    fig.update_yaxes(title="Güte [% des Optimums]", range=[0, 105])
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.4), height=height + 50, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_ratio_hist(ratios, current=None, height=260):
    """Tatsächliche Güte (Fluss / LP-Optimum) über die 40 festen Netze; rot gestrichelt das gezeigte Netz."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=[100 * r for r in ratios], xbins=dict(size=1), marker_color=C.COLORS["flow"], showlegend=False))
    if current is not None:
        fig.add_vline(x=100 * current, line=dict(color=C.COLORS["optimal"], dash="dash"), annotation_text="Ihre Ziehung", annotation_position="top")
    fig.update_xaxes(title="Güte [% des Optimums]")
    fig.update_yaxes(title="Netze")
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=30 if current is not None else 10, b=10))
    return fig


def build_eps(rows, height=320):
    """Güte gegen eps: tatsächlich (Fluss / Optimum), nach der Theorie skaliert, bewiesen (Fluss / obere Schranke) und die Garantie (1 - eps)^2 (Garg-Könemann)."""
    fig = go.Figure()
    sub = [r for r in rows if r["method"] == "gk"]
    xs = [r["eps"] for r in sub]
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["ratio"] for r in sub], mode="lines+markers", name="tatsächlich", line=dict(color=C.COLORS["flow"])))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["certified"] for r in sub], mode="lines+markers", name="bewiesen", line=dict(color=C.COLORS["dual"])))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["theory"] for r in sub], mode="lines+markers", name="mit der Skalierung der Theorie", line=dict(color="#9467bd")))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["guarantee"] for r in sub], mode="lines+markers", name="Garantie (1 − ε)²", line=dict(color=C.COLORS["optimal"], dash="dash")))
    fig.update_xaxes(title="ε", autorange="reversed", tickvals=xs)
    fig.update_yaxes(title="Güte [% des Optimums]", range=[0, 105])
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.35), height=height + 50, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_calls(rows, height=300):
    """Schiebeschritte und Orakelaufrufe gegen eps (logarithmisch): Garg-Könemann gegen Fleischer, dazu die beweisbare Schranke der Schritte."""
    fig = go.Figure()
    gk = [r for r in rows if r["method"] == "gk"]
    fl = [r for r in rows if r["method"] == "fleischer"]
    xs = [r["eps"] for r in gk]
    fig.add_trace(go.Scatter(x=xs, y=[r["calls"] for r in gk], mode="lines+markers", name="Aufrufe Garg–Könemann", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=xs, y=[r["calls"] for r in fl], mode="lines+markers", name="Aufrufe Fleischer", line=dict(color=C.COLORS["flow"])))
    fig.add_trace(go.Scatter(x=xs, y=[r["pushes"] for r in gk], mode="lines+markers", name="Schiebeschritte", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=xs, y=[r["bound"] for r in gk], mode="lines+markers", name="Schranke der Schritte", line=dict(color="#7f7f7f", dash="dot")))
    fig.update_xaxes(title="ε", autorange="reversed", tickvals=xs)
    fig.update_yaxes(title="Anzahl", type="log")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.4), height=height + 50, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_size(rows, height=300):
    """Orakelaufrufe je Gittergröße (logarithmisch): Garg-Könemann/Fleischer (eps = 0,3) gegen Column Generation (Stück 8, Aufrufe = Runden mal Güter)."""
    fig = go.Figure()
    labels = [f"{r['size'][0]}×{r['size'][1]}" for r in rows]
    fig.add_trace(go.Bar(x=labels, y=[r["calls"] for r in rows], name="Fleischer", marker_color=C.COLORS["flow"]))
    fig.add_trace(go.Bar(x=labels, y=[r["cg_calls"] for r in rows], name="Column Generation", marker_color="#9467bd"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Gitter")
    fig.update_yaxes(title="Kürzeste-Wege-Aufrufe", type="log")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 40, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_budget(rows, height=300):
    """Lieferung gegen Kosten: Garg-Könemann gegen das LP mit Budgetzeile, je Kostenbudget."""
    fig = go.Figure()
    sub = [r for r in rows if r["frac"]]
    fig.add_trace(go.Scatter(x=[r["opt_cost"] for r in sub], y=[r["opt"] for r in sub], mode="lines+markers", name="LP mit Budgetzeile", line=dict(color=C.COLORS["optimal"], dash="dash")))
    fig.add_trace(go.Scatter(x=[r["cost"] for r in sub], y=[r["primal"] for r in sub], mode="lines+markers", name="Garg–Könemann", line=dict(color=C.COLORS["flow"])))
    free = [r for r in rows if not r["frac"]]
    if free:
        fig.add_trace(go.Scatter(x=[free[0]["cost"]], y=[free[0]["primal"]], mode="markers", name="ohne Budget (kostenblind)", marker=dict(symbol="diamond", size=12, color="#ff7f0e")))
    fig.update_xaxes(title="Kosten", rangemode="tozero")
    fig.update_yaxes(title="Lieferung", rangemode="tozero")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.35), height=height + 40, margin=dict(l=10, r=10, t=10, b=10))
    return fig

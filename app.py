"""Garg–Könemann – Näherung ohne LP-Löser – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - das Garg-Könemann-Verfahren (mit der Variante von Fleischer) für den Mehrgüterfluss - und lässt stattdessen das Beispiel wachsen.
Neuntes Stück der Netzwerkfluss-Linie der "Konzepte"-Reihe, drittes im Mehrgüter-Ast: dieselben Kürzeste-Wege-Orakel wie die Column Generation des Vorgängers, aber ohne LP-Löser und mit einer Garantie. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import gk_algorithm as alg
import gk_constants as C
import gk_evaluation as ev
from gk_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from gk_visualization import (
    build_budget,
    build_calls,
    build_eps,
    build_gk,
    build_progress,
    build_quality,
    build_ratio_hist,
    build_size,
)

st.set_page_config(page_title="Garg–Könemann – Sebastian Hanisch", layout="wide")


def _pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f} %".replace(".", ",")


def _share(x):
    return f"{100 * x:.0f} %"


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _u(x):
    """Menge: ganze Zahl ohne Nachkommastellen, sonst eine."""
    return f"{round(x)}" if abs(x - round(x)) < 1e-6 else _f(x, 1)


def _n(x):
    """Ganze Zahl mit Tausendertrennung."""
    return f"{int(round(x)):,}".replace(",", " ")


def _eps(e):
    return f"{e:g}".replace(".", ",")


@st.cache_resource(show_spinner=False, max_entries=48)
def _analysis(params, opts):
    return ev.analyse(params, opts)


st.title("🎯 Garg–Könemann – Näherung ohne LP-Löser")
st.markdown(
    """
Die Column Generation des Vorgängers löst in jeder Runde ein LP. **Garg–Könemann** kommt ganz ohne aus: jede Kapazitätszeile bekommt eine **Länge**, zu Beginn winzig. Ein Kürzeste-Wege-Lauf (Dijkstra, dasselbe Orakel wie im Pricing) findet den kürzesten Weg der Güter,
auf ihn wird so viel geschoben, wie die engste Zeile hergibt, und die Längen der berührten Zeilen wachsen **multiplikativ** um den Faktor $1+\\varepsilon\\cdot\\text{Anteil}$. Volle Kanten werden dadurch lang und meiden sich selbst - der Fluss weicht aus.
Am Ende ist die Summe der Pfade viel zu groß und wird durch die größte Auslastung geteilt: zulässig, und beweisbar höchstens einen Faktor $(1-\\varepsilon)^2$ vom Optimum entfernt. **Fleischer** spart die meisten Kürzeste-Wege-Aufrufe, indem es je Gut so lange auf demselben Weg schiebt, wie er unter einer Schwelle liegt.
Diese Demo lässt die Schritte einzeln durchlaufen, zeigt an jedem Punkt die **bewiesene** Güte (die Längen liefern eine obere Schranke) und misst, wie viel besser als die Garantie das Verfahren wirklich ist - und was es kostet.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - neuntes Stück der Netzwerkfluss-Linie der \"Konzepte\"-Reihe, drittes im Mehrgüter-Ast - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Folgestück setzt an der Kosten-Zeile an: das **Netzwerkdesign mit Fixkosten** (Benders-Zerlegung, Slope Scaling)."
)

with st.expander("So funktioniert Garg–Könemann", expanded=True):
    st.markdown(
        r"""
1. **Packungs-LP:** $\max\ \sum_p x_p$ über die Pfade $p$ aller Güter, unter $\sum_{p\ni r}x_p\le b_r$ für jede Zeile $r$: gemeinsame Kapazität je Kante, Gut-Obergrenze je Werks- und Nachfragekante, optional eine **Kostenzeile** (Kosten des Pfades mal Menge $\le$ Budget). Dieselben Zeilen wie im Master der Column Generation.
2. **Längen:** $y_r=\delta/b_r$ zu Beginn mit $\delta=(1+\varepsilon)\big((1+\varepsilon)R\big)^{-1/\varepsilon}$ ($R$ = Zahl der Zeilen). Solange $D(y)=\sum_r b_r y_r<1$:
3. **Orakel:** kürzester Weg über alle Güter, Länge = Summe der $y_r$ der berührten Zeilen. **Schieben:** die Engpassmenge $f=\min_r b_r/a_{rp}$ auf ihn. **Wachsen:** $y_r\leftarrow y_r\,(1+\varepsilon f a_{rp}/b_r)$ für jede berührte Zeile.
4. **Skalieren:** die Pfadsumme überlastet die Kanten (bis zum Vielfachen der Grenze); geteilt durch die größte Auslastung ist sie zulässig. Die Theorie teilt durch $\log_{1+\varepsilon}\!\big((1+\varepsilon)/\delta\big)$ und garantiert $(1-\varepsilon)^2$ des Optimums; die größte Auslastung ist nie größer und deshalb besser.
5. **Zertifikat:** für beliebige Längen $y>0$ ist $y/\alpha(y)$ dual zulässig ($\alpha$ = kürzester Weg), also $\text{OPT}\le D(y)/\alpha(y)$. Damit hat jeder Aufnahmepunkt eine **bewiesene** untere und obere Schranke - ohne das Optimum zu kennen.
6. **Fleischer:** Phasen mit Schwelle $\alpha(1+\varepsilon)$; je Gut wird geschoben, solange sein kürzester Weg darunter liegt. Gleiche Güte, deutlich weniger Aufrufe.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()


def _kept(key, default):
    return int(st.session_state.get(KEPT[key], default))


def _slider(label, key, help, step=None):
    kw = {"step": step} if step else {}
    seed_widget(key)
    value = st.slider(label, *bounds(key), key=key, help=help, **kw)
    st.session_state[KEPT[key]] = value
    return value


with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Das Distributionsnetz des Vorgängers (dreistufig), ein Streckennetz (Gitter mit Start-Ziel-Aufträgen) oder eines von zwei festen Lehrnetzen: die Preis-Wende und das Frachtnetz, dessen LP gebrochen ist.",
    )
    show_dist, show_grid = net_key == "random", net_key == "grid"
    if show_dist or show_grid:
        K = _slider("Zahl der Güter", "k_slider", "Frische, Trocken, Kühl, Getränke, Tiefkühl (in dieser Reihenfolge). Im Distributionsnetz stellt jedes Werk ein Gut mit 70 % Wahrscheinlichkeit her; im Streckennetz hat jedes Gut einen eigenen Start und ein eigenes Ziel.")
    else:
        K = _kept("k_slider", C.DEFAULT_K)
    if show_dist:
        p = _slider("Werke", "p_slider", "Anzahl der Werke (oben im Netz).")
        d = _slider("Verteilzentren", "d_slider", "Anzahl der Verteilzentren; Durchsatz gemeinsam für alle Güter.")
        s = _slider("Filialen", "s_slider", "Anzahl der Filialen (unten im Netz).")
        density = _slider("Netzdichte [%]", "density_slider", "Anteil der möglichen Lanes, die es gibt.", step=10)
        spread = _slider("Streuung der Lane-Breiten [%]", "spread_slider", "0 = alle Lanes einer Stufe gleich breit, 100 = Kapazitäten gleichverteilt von 1 bis zum Doppelten der Grundbreite.", step=25)
        load = _slider("Auslastung [% der Werkskapazität]", "load_slider", "Gesamtnachfrage der Filialen (alle Güter zusammen) in Prozent der Werkskapazität.", step=10)
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, args=("seed_input",), help="Würfelt einen neuen Zufalls-Seed. Die Verteilungen über 40 feste Netze weiter unten ändern sich dabei nicht - nur die Marke „Ihre Ziehung“.")
    else:
        p, d, s = _kept("p_slider", C.DEFAULT_P), _kept("d_slider", C.DEFAULT_D), _kept("s_slider", C.DEFAULT_S)
        density, spread, load, seed = _kept("density_slider", C.DEFAULT_DENSITY), _kept("spread_slider", C.DEFAULT_SPREAD), _kept("load_slider", C.DEFAULT_LOAD), _kept("seed_input", C.DEFAULT_SEED)
    if show_grid:
        gw = _slider("Breite des Gitters", "gw_slider", "Knoten je Zeile.")
        gh = _slider("Höhe des Gitters", "gh_slider", "Knoten je Spalte.")
        gdensity = _slider("Anteil der Gitterkanten [%]", "gdensity_slider", "Ein zufälliger Spannbaum hält das Netz zusammenhängend; jede weitere Gitterkante gibt es mit diesem Anteil. 100 % = volles Gitter.", step=10)
        gcap = _slider("Größte Kapazität je Kante", "gcap_slider", "Jede Kante trägt in beide Richtungen 1 bis zu diesem Wert, gemeinsam für alle Güter.")
        gdem = _slider("Größte Menge je Gut", "gdem_slider", "Jedes Gut fährt 1 bis zu diesem Wert von seinem Start zu seinem Ziel.")
        seed_widget("gseed_input")
        gseed = st.number_input("Zufalls-Seed (Streckennetz)", *bounds("gseed_input"), key="gseed_input", step=1)
        st.session_state[KEPT["gseed_input"]] = gseed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, args=("gseed_input",), key="rand_grid", help="Würfelt einen neuen Zufalls-Seed für das Streckennetz.")
    else:
        gw, gh, gdensity = _kept("gw_slider", C.DEFAULT_GW), _kept("gh_slider", C.DEFAULT_GH), _kept("gdensity_slider", C.DEFAULT_GDENSITY)
        gcap, gdem, gseed = _kept("gcap_slider", C.DEFAULT_GCAP), _kept("gdem_slider", C.DEFAULT_GDEM), _kept("gseed_input", C.DEFAULT_GSEED)
    if not (show_dist or show_grid):
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Die Regler für Güter, Größe und Seed gehören zu den zufälligen Netzen.")
    st.subheader("Garg–Könemann")
    eps = st.radio("Genauigkeit ε", list(C.EPSS), key="eps_radio", format_func=_eps,
                   help="Kleineres ε = bessere Güte, aber etwa 1/ε² mal so viele Schritte. Die Garantie ist (1 − ε)² des Optimums; tatsächlich liegt die Güte weit darüber.")
    method = st.radio("Verfahren", list(C.METHODS), key="method_radio", format_func=lambda k: C.METHODS[k],
                      help="Garg–Könemann fragt in jedem Schritt jedes Gut nach seinem kürzesten Weg. Fleischer schiebt je Gut so lange auf demselben Weg, wie er unter einer Schwelle liegt.")
    budget = st.radio("Kostenbudget", list(C.BUDGETS), key="budget_radio", format_func=lambda b: "kein Budget (kostenblind)" if b == 0 else f"{b * 100:.0f} % der Kosten des LP-Optimums",
                      help="Eine zusätzliche Packungszeile: die Kosten des Flusses dürfen das Budget nicht überschreiten. Das Verfahren maximiert nur die Menge - ohne Budgetzeile sind die Kosten beliebig.")
    variant = st.radio("Variante", list(C.VARIANTS), key="variant_radio", format_func=lambda k: C.VARIANTS[k],
                       help="Die Negativkontrollen zeigen, was jede Zutat leistet: additive Längen statt multiplikativer, oder keine Skalierung am Ende.")

sync_query_params({"net_select": net_key, "k_slider": int(K), "p_slider": int(p), "d_slider": int(d), "s_slider": int(s), "density_slider": int(density), "spread_slider": int(spread),
                   "load_slider": int(load), "seed_input": int(seed), "gw_slider": int(gw), "gh_slider": int(gh), "gdensity_slider": int(gdensity), "gcap_slider": int(gcap),
                   "gdem_slider": int(gdem), "gseed_input": int(gseed), "eps_radio": eps, "method_radio": method, "budget_radio": budget, "variant_radio": variant})

params = ev.normalise(ev.NetParams(net_key, int(K), int(p), int(d), int(s), int(density), int(spread), int(load), int(seed), int(gw), int(gh), int(gdensity), int(gcap), int(gdem), int(gseed)))
opts = ev.Opts(float(eps), method, float(budget), variant)
with st.spinner("Rechne..."):
    a = _analysis(params, opts)
mcf, lp, res = a.mcf, a.lp, a.gk
level, code, dat = ev.verdict(a)
K = mcf.K
is_fixed = net_key in C.FIXED_NETS
frames = res.frames
opt = a.opt
rows_j, _, b_rows = alg.rows_of(mcf)
y0 = np.zeros(mcf.m)
for e_, i_ in rows_j.items():
    y0[e_] = res.delta / b_rows[i_]


def _pt(k, edges):
    return f"{mcf.names[k]}: {ev.path_label(mcf, edges)}"


# --- Schritte -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Schritt für Schritt zur Näherung")
owner = (params, opts)
if st.session_state.get("gk_owner") != owner:
    st.session_state["gk_frame"] = len(frames) - 1
    st.session_state["gk_owner"] = owner
step_col, play_col = st.columns([5, 2])
with step_col:
    if len(frames) > 1:
        fr_no = st.slider("Aufnahmepunkt", 0, len(frames) - 1, key="gk_frame", help="Die Schritte sind eng am Anfang (jeder der ersten acht) und dann in wachsenden Abständen aufgenommen; der letzte Punkt ist das Ende des Verfahrens mit der Skalierung.")
    else:
        fr_no = 0
        st.caption("In diesem Netz gibt es nur einen Aufnahmepunkt.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _caption(i):
    f = frames[i]
    last = i == len(frames) - 1
    cert = None if f.dual in (0.0, float("inf")) or f.dual <= 1e-9 else f.primal / f.dual
    text = f"**{'Ende' if last else 'Nach ' + _n(f.push) + ' Schiebeschritten'}:** {_n(f.push)} Schritte, {_n(f.calls)} Kürzeste-Wege-Aufrufe. "
    text += f"Die Summe der Pfade überlastet die Kanten bis zum {_f(f.congestion, 1)}-Fachen ihrer Grenze; geteilt durch diese größte Auslastung ist der Fluss zulässig und liefert {_u(f.primal)} Einheiten"
    text += f" ({_pct(f.primal / opt if opt > 1e-9 else 1)} des Optimums {_u(opt)}). " if opt > 1e-9 else ". "
    if f.dual not in (0.0, float("inf")):
        text += f"Die Längen beweisen: das Optimum liegt höchstens bei {_u(f.dual)} - der Fluss ist mindestens {_pct(cert)} des Optimums. "
    else:
        text += "Eine obere Schranke ist noch nicht bekannt. "
    if f.path is not None:
        text += f"Zuletzt geschoben: {_pt(*f.path)}, Menge {_f(f.amount, 1)}."
    return text


def _render(i):
    with view_slot.container():
        f = frames[i]
        c1, c2 = st.columns(2)
        c1.markdown(f"**Nach {_n(f.push)} Schiebeschritten:** die Längen der Kanten (orange) und der skalierte Fluss")
        c2.markdown(f"**Lieferung und Güte** - zulässiger Fluss {_u(f.primal)} von {_u(opt)} Einheiten")
        c1.plotly_chart(build_gk(mcf, f, y0), width="stretch", key=f"gk_map_{i}")
        c2.plotly_chart(build_progress(frames, i, opt), width="stretch", key=f"progress_chart_{i}")
        c2.plotly_chart(build_quality(frames, i, opt, res.eps), width="stretch", key=f"quality_chart_{i}")
        st.caption(_caption(i))


if auto_play:
    for i in range(len(frames)):
        _render(i)
        time.sleep(min(0.9, 8.0 / max(len(frames), 1)))
    fr_no = len(frames) - 1
else:
    _render(fr_no)

st.caption("Links: je Gut eine Farbe, Breite ~ zulässig skalierter Fluss; orange Unterlage = Kante, deren Länge durch das multiplikative Wachstum gegenüber dem Start um mindestens den Faktor 10 gewachsen ist, in drei Stufen nach dem Logarithmus des Faktors (stark = oberes Drittel), mit dem Faktor an den fünf längsten; "
           "schwarz gestrichelt der zuletzt geschobene Pfad. Rechts oben die Lieferung gegen die Schiebeschritte (logarithmisch): der zulässige Fluss (untere Schranke) steigt, die aus den Längen bewiesene obere Schranke sinkt, gestrichelt das LP-Optimum; "
           "unten die Güte in Prozent des Optimums gegen die Garantie (1 − ε)².")

st.markdown("---")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wie gut ist die Näherung - und was kostet sie?")
st.caption("**Güte tatsächlich** = zulässiger Fluss / LP-Optimum (das Vergleichs-LP löst HiGHS, das Verfahren selbst kennt es nicht); **bewiesen** = Fluss / obere Schranke aus den Längen; **Garantie** = (1 − ε)²; **Aufrufe** = Kürzeste-Wege-Läufe (Dijkstra).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lieferung", f"{_u(dat['primal'])} von {_u(dat['opt'])}", delta=f"{_pct(dat['ratio'])} des Optimums", delta_color="off", help=f"Zulässiger Fluss des Verfahrens gegen das Optimum der maximalen Lieferung des LP{' mit Kostenbudget' if a.budget_abs else ''}.")
m2.metric("Güte bewiesen", _pct(dat["certified"]), delta=f"Garantie {_pct(dat['guarantee'], 0)}", delta_color="off", help="Fluss / obere Schranke aus den Längen - ohne das Optimum zu kennen. Die Garantie (1 − ε)² gilt mit der Skalierung der Theorie; hier ist die Skalierung durch die größte Auslastung besser.")
m3.metric("Schiebeschritte", _n(dat["pushes"]), delta=f"Schranke {_n(dat['bound'])}", delta_color="off", help="Jeder Schritt vervielfacht die Länge der Engpasszeile mit 1 + ε; jede Zeile kann nur log_{1+ε}((1 + ε)/δ) mal wachsen. Also höchstens Zeilen mal so viele Schritte.")
m4.metric("Aufrufe", _n(dat["calls"]), delta=f"{dat['rows']} Zeilen, {dat['columns']} Pfade", delta_color="off", help="Kürzeste-Wege-Läufe insgesamt. Es wird kein LP gelöst; die Pfade sind die, auf die geschoben wurde.")

pth = f"Kosten {_u(dat['cost'])}" + (f" (Budget {_u(a.budget_abs)})" if a.budget_abs else f" - {_f(dat['cost_ratio'], 2)}-fach der Kosten des LP-Optimums ({_u(dat['lp_cost'])}): das Verfahren maximiert nur die Menge")
if code == "ok":
    st.success(f"✅ Zulässiger Fluss mit {_u(dat['primal'])} von {_u(dat['opt'])} Einheiten: **{_pct(dat['ratio'])} des Optimums**, bewiesen mindestens {_pct(dat['certified'])} (Garantie {_pct(dat['guarantee'], 0)}). "
               f"{_n(dat['pushes'])} Schiebeschritte (Schranke {_n(dat['bound'])}), {_n(dat['calls'])} Kürzeste-Wege-Aufrufe, kein LP gelöst. {pth}.")
elif code == "noscale":
    st.error(f"❌ Ohne die Skalierung am Ende ist der Fluss unzulässig: die Pfadsumme ({_u(dat['primal'])} Einheiten) überlastet die Kanten bis zum {_f(dat['congestion'], 1)}-Fachen ihrer Grenze - das Optimum ist {_u(dat['opt'])}. "
             "Erst das Teilen durch die größte Auslastung macht den Fluss zulässig.")
elif code == "bad":
    st.warning(f"⚠️ Mit additiven statt multiplikativen Längen bricht das Verfahren nach {_n(dat['pushes'])} Schritten ab: {_u(dat['primal'])} von {_u(dat['opt'])} Einheiten, nur {_pct(dat['ratio'])} des Optimums (Garantie mit der richtigen Regel: {_pct(dat['guarantee'], 0)}, tatsächlich meist über 90 %). "
               "Erst das multiplikative Wachstum lässt volle Kanten so schnell lang werden, dass der Fluss ausweicht, bevor die Abbruchbedingung greift.")
else:
    st.warning("⚠️ Es kommt gar nichts an: kein Gut kann von seinem Start zu seinem Ziel gelangen.")

if is_fixed:
    st.info("Festes Netz: es gibt nur diese eine Ziehung. Für die Verteilungen über viele Netze ein zufälliges Netz wählen.")
    dist = None
else:
    with st.spinner("Rechne die 40 festen Netze..."):
        dist = ev.distribution(params, opts)
    kind = "Streckennetze" if net_key == "grid" else "Distributionsnetze"
    st.markdown(f"**Nicht nur dieses eine Netz:** {dist['n_seeds']} feste {kind} mit denselben Einstellungen (Güter {K}, ε = {_eps(opts.eps)}, {'Fleischer' if method == 'fleischer' else 'Garg–Könemann'}), getrennt vom Seed oben.")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Güte tatsächlich", _pct(dist["ratio_mean"]), delta=f"schlechtestes Netz {_pct(dist['ratio_min'])}", delta_color="off", help="Mittel über die Netze: zulässiger Fluss / LP-Optimum.")
    p2.metric("Güte bewiesen", _pct(dist["certified_mean"]), delta=f"schlechtestes {_pct(dist['certified_min'])}", delta_color="off", help="Mittel über die Netze: Fluss / obere Schranke aus den Längen.")
    p3.metric("Schiebeschritte", _n(dist["pushes_mean"]), delta=f"Schranke {_n(dist['bound_mean'])}", delta_color="off", help="Mittel über die Netze.")
    p4.metric("Aufrufe", _n(dist["calls_mean"]), delta=f"Garantie {_pct((1 - opts.eps) ** 2, 0)}", delta_color="off", help="Mittel über die Netze: Kürzeste-Wege-Läufe.")
    st.plotly_chart(build_ratio_hist(dist["cols"]["ratio"], current=dat["ratio"]), width="stretch", key="ratio_hist")
    st.caption(f"Über {dist['n_seeds']} feste Netze liegt die tatsächliche Güte im Mittel bei {_pct(dist['ratio_mean'])} (schlechtestes Netz {_pct(dist['ratio_min'])}), die Garantie bei {_pct((1 - opts.eps) ** 2, 0)}; "
               f"mit der Skalierung der Theorie wären es im Mittel {_pct(dist['theory_mean'])}. Die obere Schranke ist in allen Netzen gültig ({_share(dist['share_dual_valid'])}).")

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – die Zutaten"):
    st.markdown("**Was das Verfahren für dieses Netz getan hat**")
    st.table({"Größe": ["Zeilen R (Kapazitäten, Obergrenzen" + (", Budget)" if a.budget_abs else ")"), "Anfangslänge δ", "Skalierung der Theorie", "größte Auslastung der Pfadsumme", "Pfade mit Fluss", "Phasen (Fleischer)"],
              "Wert": [str(dat["rows"]), f"{res.delta:.2e}".replace(".", ","), _f(res.theory_scale, 1), _f(dat["congestion"], 1), str(dat["columns"]), str(dat["phases"]) if method == "fleischer" else "–"]})
    st.caption("Die Skalierung der Theorie ist ein fester Wert je (ε, R); die größte Auslastung ist der Wert, den der Lauf wirklich braucht - nie größer, deshalb ist der Fluss mit ihr mindestens so gut wie mit der Theorie.")
    st.markdown("**Die längsten Kanten am Ende**")
    ends = np.argsort(-(frames[-1].y / np.where(y0 > 0, y0, np.inf)))[:8]
    st.table({"Kante": [f"{mcf.net.names[mcf.net.arcs[e][0]]} → {mcf.net.names[mcf.net.arcs[e][1]]}" for e in ends if mcf.joint[e]],
              "Längenfaktor": [_n(frames[-1].y[e] / y0[e]) for e in ends if mcf.joint[e]]})

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel besser als die Garantie ist das Verfahren?")
st.caption("Garantie (1 − ε)² gegen tatsächliche Güte, die Güte mit der Skalierung der Theorie und die aus den Längen bewiesene Güte - beide Verfahren, 20 feste Netze mit den Einstellungen oben.")
if dist is None:
    st.info("Für dieses Experiment ein zufälliges Netz wählen.")
else:
    if st.button("ε von 0,5 bis 0,1 durchrechnen (20 Netze, beide Verfahren)", key="eps_start"):
        st.session_state["eps_on"] = True
    if st.session_state.get("eps_on"):
        with st.spinner("Rechne..."):
            et = ev.eps_table(params)
        st.plotly_chart(build_eps(et), width="stretch", key="eps_chart")
        gk_rows = [r for r in et if r["method"] == "gk"]
        st.table({"ε": [_eps(r["eps"]) for r in gk_rows], "Garantie": [_pct(r["guarantee"], 0) for r in gk_rows], "Theorie-Skalierung": [_pct(r["theory"], 0) for r in gk_rows],
                  "tatsächlich": [_pct(r["ratio"]) for r in gk_rows], "schlechtestes Netz": [_pct(r["ratio_min"]) for r in gk_rows], "bewiesen": [_pct(r["certified"]) for r in gk_rows]})
        st.caption(f"Mittel über 20 feste Netze, Garg–Könemann. Bei ε = 0,5 garantiert die Theorie 25 %, tatsächlich sind es {_pct(gk_rows[0]['ratio'], 0)}; bei ε = 0,1 garantiert sie 81 %, tatsächlich {_pct(gk_rows[-1]['ratio'])}. "
                   "Die Garantie ist ein Worst-Case-Wert; die zulässige Skalierung durch die größte Auslastung holt fast alles heraus.")
        st.plotly_chart(build_calls(et), width="stretch", key="calls_chart")
        fl_rows = [r for r in et if r["method"] == "fleischer"]
        st.table({"ε": [_eps(r["eps"]) for r in gk_rows], "Schiebeschritte": [_n(r["pushes"]) for r in gk_rows], "Schranke": [_n(r["bound"]) for r in gk_rows], "Aufrufe Garg–Könemann": [_n(r["calls"]) for r in gk_rows],
                  "Aufrufe Fleischer": [_n(r["calls"]) for r in fl_rows], "Fleischer spart": [f"{g['calls'] / f['calls']:.1f}-fach".replace(".", ",") for g, f in zip(gk_rows, fl_rows)]})
        s_ratio = et[6]["pushes"] / et[0]["pushes"] if et[0]["pushes"] else 0
        st.caption(f"Die Schritte wachsen etwa mit 1/ε²: von ε = 0,5 auf 0,1 (fünffach kleiner) auf das {_f(s_ratio, 0)}-Fache statt 25. Jeder Schritt kostet bei Garg–Könemann einen Aufruf je Gut, bei Fleischer im Mittel wenig mehr als einen; die Schranke ist eine beweisbare Obergrenze, tatsächlich wird nur ein Bruchteil gebraucht.")

st.subheader("🔬 Was kostet ein Kostenbudget?")
st.caption("Das Verfahren maximiert nur die Menge - die Kosten sind beliebig. Eine zusätzliche Packungszeile begrenzt sie. Kompromiss Lieferung/Kosten für das gezeigte Netz (ε = 0,2), gegen das LP mit Budgetzeile.")
if st.button("Budgets von 100 bis 50 % durchrechnen", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne..."):
        bt = ev.budget_table(params)
    st.plotly_chart(build_budget(bt), width="stretch", key="budget_chart")
    st.table({"Budget": ["kein Budget" if not r["frac"] else f"{r['frac'] * 100:.0f} %" for r in bt], "LP-Optimum": [_u(r["opt"]) for r in bt], "Garg–Könemann": [_f(r["primal"], 1) for r in bt],
              "Güte": [_pct(r["ratio"]) for r in bt], "Kosten": [_u(r["cost"]) for r in bt], "Schritte": [_n(r["pushes"]) for r in bt]})
    st.caption("Budget in Prozent der Kosten des LP-Optimums (größte Lieferung, darunter billigste). Ohne Budget liefert das Verfahren fast das Optimum - zu deutlich höheren Kosten; mit Budget weichen die Pfade auf billigere aus, und die Lieferung fällt entlang der LP-Kurve.")

st.subheader("🔬 Wann gewinnt das Verfahren gegen HiGHS und die Column Generation?")
st.caption("Gemessen in Kürzeste-Wege-Aufrufen (Sekunden nur zur Information): volle Gitter von 5 × 4 bis 16 × 10 mit fünf Gütern, Fleischer mit ε = 0,3, 4 feste Netze je Größe.")
if st.button("Gitter von 5 × 4 bis 16 × 10 durchrechnen", key="size_start"):
    st.session_state["size_on"] = True
if st.session_state.get("size_on"):
    with st.spinner("Rechne..."):
        sz = ev.size_table()
    st.plotly_chart(build_size(sz), width="stretch", key="size_chart")
    st.table({"Gitter": [f"{r['size'][0]} × {r['size'][1]}" for r in sz], "Zeilen": [_n(r["rows"]) for r in sz], "Schritte": [_n(r["pushes"]) for r in sz], "Aufrufe Fleischer": [_n(r["calls"]) for r in sz],
              "Aufrufe Column Generation": [_n(r["cg_calls"]) for r in sz], "Güte": [_pct(r["ratio"]) for r in sz]})
    st.table({"Gitter": [f"{r['size'][0]} × {r['size'][1]}" for r in sz], "Zeit Kanten-LP [ms]": [_f(1000 * r["t_lp"], 0) for r in sz], "Zeit Column Generation [ms]": [_f(1000 * r["t_cg"], 0) for r in sz],
              "Zeit Fleischer [ms]": [_f(1000 * r["t_gk"], 0) for r in sz]})
    st.caption("Median der Sekunden über die Netze, nur zur Information (Python-Dijkstra gegen einen HiGHS-Lauf). Die Aussage steht in den Aufrufen: Fleischer braucht ein Vielfaches der Aufrufe der Column Generation, weil es ohne Optimierung jeden Schritt einzeln geht - "
               "und einen Fluss mit einigen Prozent Abstand statt dem Optimum. Die Schritte wachsen mit den Zeilen nur langsam, aber jeder Aufruf wird teurer: im gemessenen Bereich gewinnt HiGHS immer.")

st.subheader("🔬 Was leistet jede Zutat?")
st.caption("Negativkontrollen bei ε = 0,3 (Garg–Könemann, 20 feste Netze mit den Einstellungen oben): das Verfahren wie beschrieben, mit additiven statt multiplikativen Längen, und ohne die Skalierung am Ende.")
if dist is None:
    st.info("Für dieses Experiment ein zufälliges Netz wählen.")
else:
    if st.button("Zutaten durchrechnen (20 Netze)", key="control_start"):
        st.session_state["control_on"] = True
    if st.session_state.get("control_on"):
        with st.spinner("Rechne..."):
            ct = ev.control_table(params)
        st.table({"Variante": [C.VARIANTS[r["variant"]] for r in ct], "Fluss / Optimum": [_pct(r["ratio"]) for r in ct], "größte Auslastung": [_f(r["congestion"], 1) for r in ct],
                  "zulässig": [_share(r["feasible"]) for r in ct], "Schritte": [_n(r["pushes"]) for r in ct]})
        st.caption("Ohne Skalierung ist der Fluss ein Vielfaches des Optimums - und unzulässig, weil die Kanten bis zum Vielfachen ihrer Grenze überlastet sind. Mit additiven Längen bricht das Verfahren nach wenigen Schritten ab; der Fluss ist zulässig, aber weit vom Optimum: "
                   "erst das multiplikative Wachstum lässt volle Kanten so schnell lang werden, dass der Fluss rechtzeitig ausweicht.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist - und wer setzt an |
|---|---|
| **Teilbare Ströme** | Das Verfahren löst das LP näherungsweise; die Pfadmengen dürfen Bruchteile sein (Frachtnetz mit Bruch). Ganzzahlig: Rundung oder **Branch-and-Price**, hier nicht gebaut. |
| **Nur die Menge zählt** | Das Ziel ist die maximale Lieferung; die Kosten sind beliebig. Mit einer Kostenzeile kommt man an einen Kompromiss - das Kostenminimum bei gegebener Menge braucht eine Suche über das Budget. |
| **Eine Näherung genügt** | Die Güte ist bewiesen, aber nicht das Optimum; wer die exakte Lösung oder die Duale (Schattenpreise) braucht, nimmt das LP oder die **Column Generation** des Vorgängers. |
| **Die Schritte sind billig** | Jeder Schritt braucht einen Dijkstra-Lauf je Gut; für die Größen hier gewinnt der LP-Löser trotzdem (Experiment). Das Verfahren lohnt, wo kein LP-Löser passt oder das LP nicht in den Speicher geht. |
| **Die Kanten stehen fest** | Hier gibt es die Kanten; wer sie erst bauen muss, zahlt Fixkosten. **Ansatzpunkt:** Netzwerkdesign mit Fixkosten (gebaut: fixkosten-netzdesign-demo; danach Benders-Zerlegung, Slope Scaling). |
| **Keine Zeit** | Ein Fluss ist eine Momentaufnahme. **Ansatzpunkt:** Zeit-Raum-Netz in der Demo „leercontainer-demo“. |
"""
)
st.caption("Die Netzwerkfluss-Linie ist als Ganzes geplant: Edmonds-Karp, Dinic, Push-Relabel, Successive Shortest Paths, Cycle-Canceling, Cost Scaling, Mehrgüterfluss, Column Generation, Garg-Könemann (dieses Stück), Fixkosten-Netzwerkdesign (gebaut), Benders-Zerlegung und Slope Scaling - bisher sind die ersten zehn gebaut.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Packungs-LP.** Pfade $P$ aller Güter, Zeilen $r=1,\dots,R$ mit Grenzen $b_r>0$ und Koeffizienten $a_{rp}\in\{0,1\}$ (Kostenzeile: $a_{rp}=$ Kosten des Pfades):
$$\max\ \sum_{p\in P}x_p\quad\text{u.d.N.}\quad \sum_{p}a_{rp}x_p\le b_r\ \ (r=1,\dots,R),\qquad x\ge 0.$$
Dual: $\min\ \sum_r b_r y_r$ u.d.N. $\sum_r a_{rp}y_r\ge 1$ für alle Pfade $p$, $y\ge 0$. Für Längen $y$ und $\alpha(y)=\min_p\sum_r a_{rp}y_r$ ist $y/\alpha(y)$ dual zulässig, also $\text{OPT}\le D(y)/\alpha(y)$ mit $D(y)=\sum_r b_r y_r$.

**Garg–Könemann.** $\delta=(1+\varepsilon)\big((1+\varepsilon)R\big)^{-1/\varepsilon}$, $y_r=\delta/b_r$. Solange $D(y)<1$: kürzesten Pfad $p$ bezüglich $y$ bestimmen, $f=\min_{r:\,a_{rp}>0}b_r/a_{rp}$ auf ihn schieben ($x_p\mathrel{+}=f$), $y_r\leftarrow y_r\big(1+\varepsilon f a_{rp}/b_r\big)$. Am Ende $x/\log_{1+\varepsilon}\!\big((1+\varepsilon)/\delta\big)$ ist zulässig und mindestens $(1-\varepsilon)^2\cdot\text{OPT}$ (Garg und Könemann 1998, Fleischer 2000). Jeder Schritt vervielfacht die Länge der Engpasszeile mit $1+\varepsilon$; keine Länge wächst über $(1+\varepsilon)/b_r$, bevor $D\ge1$ - also höchstens $R\log_{1+\varepsilon}\!\big((1+\varepsilon)/\delta\big)$ Schritte.

**Skalierung durch die größte Auslastung.** Ist $\lambda=\max_r\sum_p a_{rp}x_p/b_r$, so ist $x/\lambda$ zulässig. Weil die Skalierung der Theorie $\ge\lambda$ ist, ist $x/\lambda$ mindestens so gut - und jederzeit zulässig, auch mitten im Lauf.

**Fleischer.** Phasen mit Schwelle $\alpha(1+\varepsilon)$: für jedes Gut $k$ wird so lange auf seinem kürzesten Weg geschoben, wie dieser unter der Schwelle liegt; danach $\alpha\leftarrow\alpha(1+\varepsilon)$. Die Zahl der Schritte ist dieselbe wie bei Garg–Könemann, die der Aufrufe etwa ein Gut-Faktor kleiner. Nach einer vollständigen Phase ist jeder Weg mindestens so lang wie die kleinste geprüfte Länge, daraus folgt eine obere Schranke.

Implementiert in `gk_algorithm.py` (Verfahren, Zertifikat, Aufnahmepunkte), `gk_lp.py` (Kanten-LP mit Budgetzeile, HiGHS über `scipy`), `gk_cg.py` (Column Generation des Vorgängers als Vergleich), `gk_paths.py`, `gk_model.py` und `gk_scenario.py` (Netze), `gk_evaluation.py` (Kennzahlen, Verteilungen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)

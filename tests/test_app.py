"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Optionen, Randgrößen, Aufnahmepunkt-Regler, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import itertools
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gk_constants as C
from gk_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# Anfang der Meldung zum gezeigten Netz (Streamlit legt das führende Emoji in `icon`, nicht in `value`)
EXPECTED = {
    "🚚 Zufallsnetz": "Zulässiger Fluss mit 66,5 von 67 Einheiten:",
    "🗺️ Streckennetz": "Zulässiger Fluss mit 6,5 von 7 Einheiten:",
    "🔀 Preis-Wende": "Zulässiger Fluss mit 2,0 von 2 Einheiten:",
    "🧩 Frachtnetz mit Bruch": "Zulässiger Fluss mit 1,4 von 1,5 Einheiten:",
    "⚡ Fleischer": "Zulässiger Fluss mit 66,9 von 67 Einheiten:",
    "🎯 Feine Näherung": "Zulässiger Fluss mit 66,8 von 67 Einheiten:",
    "💰 Kostenbudget 75 %": "Zulässiger Fluss mit 52,4 von 56,2 Einheiten:",
    "🚫 Ohne Skalierung": "Ohne die Skalierung am Ende ist der Fluss unzulässig:",
}
OPTION_LABELS = {"Genauigkeit ε", "Verfahren", "Kostenbudget", "Variante"}


def _run(setup=None, timeout=900):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.radio)}


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info) + list(at.error)]


def _has(at, prefix):
    return any(t.startswith(prefix) for t in _texts(at))


def _frame(at):
    found = [s for s in at.slider if s.key == "gk_frame"]
    return found[0] if found else None


def _metric(at, label):
    return [m.value for m in at.metric if m.label == label]


def _captions(at):
    return [c.value for c in at.caption]


def test_default_renders_without_exception():
    at = _run()
    assert any("Schritt für Schritt zur Näherung" in m.value for m in at.markdown)
    assert _has(at, EXPECTED["🚚 Zufallsnetz"]) and not at.error
    assert _metric(at, "Lieferung")[0].startswith("66,5 von 67") and _metric(at, "Güte bewiesen")[0].endswith("%") and _metric(at, "Güte tatsächlich")[0].endswith("%")
    assert _frame(at).value == _frame(at).max >= 20


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdicts(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert _has(at, EXPECTED[name]), _texts(at)
    if C.PRESETS[name]["net"] in C.FIXED_NETS:
        assert any(t.startswith("Festes Netz") for t in _texts(at))
    else:
        assert any(m.value.startswith("**Nicht nur dieses eine Netz:**") for m in at.markdown)
    assert _frame(at).value == _frame(at).max


@pytest.mark.parametrize("K", range(C.K_MIN, C.K_MAX + 1))
@pytest.mark.parametrize("net", ["random", "grid"])
def test_every_number_of_goods_renders(net, K):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["k_slider"] = K
    at = _run(setup)
    assert not at.error and any(t.startswith("Zulässiger Fluss") for t in _texts(at)) and _frame(at).max >= 10


@pytest.mark.parametrize("eps,method", list(itertools.product(C.EPSS, C.METHODS)))
def test_every_eps_and_method_renders(eps, method):
    def setup(at):
        at.session_state["eps_radio"] = eps
        at.session_state["method_radio"] = method
    at = _run(setup)
    assert not at.error and any(t.startswith("Zulässiger Fluss") for t in _texts(at))


@pytest.mark.parametrize("budget", C.BUDGETS)
@pytest.mark.parametrize("variant", list(C.VARIANTS))
def test_every_budget_and_variant_renders(budget, variant):
    def setup(at):
        at.session_state["budget_radio"] = budget
        at.session_state["variant_radio"] = variant
    at = _run(setup)
    assert not at.exception and len(_texts(at)) >= 1


def test_extreme_sizes_render():
    for net, vals in (("random", (("p_slider", C.P_MIN), ("d_slider", C.D_MIN), ("s_slider", C.S_MIN), ("density_slider", C.DENSITY_MIN), ("spread_slider", C.SPREAD_MIN), ("load_slider", C.LOAD_MIN), ("k_slider", C.K_MIN))),
                      ("random", (("p_slider", C.P_MAX), ("d_slider", C.D_MAX), ("s_slider", C.S_MAX), ("density_slider", C.DENSITY_MAX), ("spread_slider", C.SPREAD_MAX), ("load_slider", C.LOAD_MAX), ("k_slider", C.K_MAX))),
                      ("grid", (("gw_slider", C.GW_MIN), ("gh_slider", C.GH_MIN), ("gdensity_slider", C.GDENSITY_MIN), ("gcap_slider", C.GCAP_MIN), ("gdem_slider", C.GDEM_MIN), ("k_slider", C.K_MIN))),
                      ("grid", (("gw_slider", C.GW_MAX), ("gh_slider", C.GH_MAX), ("gdensity_slider", C.GDENSITY_MAX), ("gcap_slider", C.GCAP_MAX), ("gdem_slider", C.GDEM_MAX), ("k_slider", C.K_MAX)))):
        def setup(at, net=net, vals=vals):
            at.session_state["net_select"] = net
            for key, value in vals:
                at.session_state[key] = value
        at = _run(setup)
        assert not at.error


def test_a_net_where_nothing_arrives_renders_and_says_so():
    """Zwei Werke, sechs Verteilzentren, drei Filialen, dünnes Netz: kein Weg von S nach T, es gibt nur einen Aufnahmepunkt (kein Regler mit min = max)."""
    def setup(at):
        for key, value in (("p_slider", 2), ("d_slider", 6), ("s_slider", 3), ("density_slider", 20), ("spread_slider", 50), ("load_slider", 90), ("seed_input", 8)):
            at.session_state[key] = value
    at = _run(setup)
    assert any("kommt gar nichts an" in t for t in _texts(at)) and _frame(at) is None and any("nur einen Aufnahmepunkt" in c for c in _captions(at))


def test_hidden_controls_follow_the_net():
    def labels_for(net):
        return _labels(_run(lambda a: a.session_state.__setitem__("net_select", net)))
    random_labels, grid, fixed = labels_for("random"), labels_for("grid"), labels_for("preis")
    assert {"Netz", "Zahl der Güter", "Werke", "Verteilzentren", "Filialen", "Netzdichte [%]", "Streuung der Lane-Breiten [%]", "Auslastung [% der Werkskapazität]", "Zufalls-Seed"} | OPTION_LABELS == random_labels
    assert {"Netz", "Zahl der Güter", "Breite des Gitters", "Höhe des Gitters", "Anteil der Gitterkanten [%]", "Größte Kapazität je Kante", "Größte Menge je Gut", "Zufalls-Seed (Streckennetz)"} | OPTION_LABELS == grid
    assert fixed == {"Netz"} | OPTION_LABELS


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    at = _run(lambda a: (a.session_state.__setitem__("density_slider", 80), a.session_state.__setitem__("k_slider", 4)))
    at.session_state["net_select"] = "gap"
    at.run()
    at.session_state["net_select"] = "random"
    at.run()
    assert not at.exception and at.slider(key="density_slider").value == 80 and at.slider(key="k_slider").value == 4
    at = _run(lambda a: (a.session_state.__setitem__("net_select", "grid"), a.session_state.__setitem__("gcap_slider", 3)))
    at.session_state["net_select"] = "preis"
    at.run()
    at.session_state["net_select"] = "grid"
    at.run()
    assert not at.exception and at.slider(key="gcap_slider").value == 3


def test_frame_slider_shows_every_frame_with_its_caption():
    at = _run(lambda a: _apply(a, C.PRESETS["🔀 Preis-Wende"]))
    top = _frame(at).max
    _frame(at).set_value(0)
    at.run()
    assert not at.exception and any(c.startswith("**Nach 1 Schiebeschritten:**") and "Kürzeste-Wege-Aufrufe" in c for c in _captions(at))
    _frame(at).set_value(top // 2)
    at.run()
    assert not at.exception and any("geteilt durch diese größte Auslastung ist der Fluss zulässig" in c and "Zuletzt geschoben:" in c for c in _captions(at))
    _frame(at).set_value(top)
    at.run()
    assert any(c.startswith("**Ende:**") and "Die Längen beweisen: das Optimum liegt höchstens bei" in c for c in _captions(at))


def test_changing_the_net_resets_the_frame_to_the_end():
    at = _run()
    _frame(at).set_value(2)
    at.run()
    assert _frame(at).value == 2
    at.session_state["k_slider"] = 4
    at.run()
    assert not at.exception and _frame(at).value == _frame(at).max


def test_play_runs_through_all_frames_without_duplicate_chart_keys():
    at = _run(lambda a: _apply(a, C.PRESETS["🔀 Preis-Wende"]))
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.query_params["density"] = "9999"
    at.query_params["spread"] = "abc"
    at.query_params["p"] = "-5"
    at.query_params["k"] = "9"
    at.run()
    assert not at.exception
    assert at.slider(key="density_slider").value == C.DENSITY_MAX and at.slider(key="spread_slider").value == C.DEFAULT_SPREAD and at.slider(key="p_slider").value == C.P_MIN and at.slider(key="k_slider").value == C.K_MAX
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.query_params["net"] = "grid"
    at.query_params["gdensity"] = "63"
    at.query_params["eps"] = "0.5"
    at.query_params["method"] = "fleischer"
    at.query_params["budget"] = "0.9"
    at.query_params["variant"] = "add"
    at.run()
    assert at.slider(key="gdensity_slider").value == 60 and at.radio(key="eps_radio").value == 0.5 and at.radio(key="method_radio").value == "fleischer"
    assert at.radio(key="budget_radio").value == 0.9 and at.radio(key="variant_radio").value == "add"


def test_unknown_values_in_the_permalink_fall_back_to_the_defaults():
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.query_params["net"] = "ring"
    at.query_params["method"] = "lifo"
    at.query_params["eps"] = "0.3333"
    at.query_params["variant"] = "vielleicht"
    at.query_params["k"] = "abc"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET and at.radio(key="method_radio").value == C.DEFAULT_METHOD
    assert at.radio(key="eps_radio").value == C.DEFAULT_EPS and at.radio(key="variant_radio").value == C.DEFAULT_VARIANT and at.slider(key="k_slider").value == C.DEFAULT_K


def test_randomize_moves_the_seed_but_not_the_distribution():
    at = _run()
    labels = ("Güte tatsächlich",)
    pick = lambda a: [_metric(a, label) for label in labels] + [_metric(a, "Güte bewiesen")[1:], _metric(a, "Schiebeschritte")[1:], _metric(a, "Aufrufe")[1:]]
    before = pick(at)
    seed_before = at.number_input(key="seed_input").value
    [b for b in at.sidebar.button if "Neues Netz" in b.label][0].click()
    at.run()
    assert not at.exception and at.number_input(key="seed_input").value != seed_before and pick(at) == before
    at.session_state["net_select"] = "grid"
    at.run()
    g_before = at.number_input(key="gseed_input").value
    at.button(key="rand_grid").click()
    at.run()
    assert not at.exception and at.number_input(key="gseed_input").value != g_before


def test_experiments_run_on_demand():
    at = _run()
    for text in ("Mittel über 20 feste Netze, Garg–Könemann.", "Budget in Prozent der Kosten", "Median der Sekunden", "Ohne Skalierung ist der Fluss ein Vielfaches"):
        assert not any(text in c for c in _captions(at))
    for key in ("eps_start", "budget_start", "size_start", "control_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    text = " ".join(_captions(at))
    assert "Mittel über 20 feste Netze, Garg–Könemann." in text and "Budget in Prozent der Kosten" in text and "Median der Sekunden" in text and "Ohne Skalierung ist der Fluss ein Vielfaches" in text
    assert "Die Schritte wachsen etwa mit 1/ε²" in text and "im gemessenen Bereich gewinnt HiGHS immer" in text


def test_experiments_on_a_fixed_net_show_hints_instead_of_dead_controls():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "gap"))
    assert not [b for b in at.button if b.key in ("eps_start", "control_start")] and {b.key for b in at.button if b.key in ("budget_start", "size_start")} == {"budget_start", "size_start"}
    assert sum("zufälliges Netz wählen" in t for t in _texts(at)) >= 3


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert all(re.search(r'key=f?"[a-z_]+(_\{\w+\})?"', c) for c in calls), calls
    assert len(calls) == 8 and len(set(keys)) == 8 and keys[:3] == ["gk_map", "progress_chart", "quality_chart"]
    viz = (ROOT / "gk_visualization.py").read_text(encoding="utf-8")
    bodies = [b for b in viz.split(chr(10) + "def ") if b.startswith("build_")]
    assert "fixedrange=True" in viz and len(bodies) == 8 and all("_base(" in b or "_layout(" in b for b in bodies)


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_footer_is_verbatim():
    src = APP.read_text(encoding="utf-8")
    assert "https://sebastianhanisch.net/kontakt.html" in src and "Interesse an einer maßgeschneiderten Lösung für" in src and "Operations Research und Machine Learning" in src


def test_runtime_needs_scipy_but_never_networkx():
    """Der Vergleichs-LP-Löser ist HiGHS über scipy (Laufzeit); networkx bleibt reines Testorakel."""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "scipy" in req and "networkx" not in req
    for path in ROOT.glob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+networkx\b", path.read_text(encoding="utf-8"), re.M), path.name

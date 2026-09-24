"""Presets: vollständig, in den Grenzen, und jedes Beispielnetz zeigt, was sein Hilfetext behauptet."""

import pytest

import gk_constants as C
import gk_evaluation as ev
import gk_presets as P

KEYS = set(P.PRESET_KEYS)


def _analyse(p):
    params = ev.normalise(ev.NetParams(p["net"], p["k"], p["p"], p["d"], p["s"], p["density"], p["spread"], p["load"], p["seed"], p["gw"], p["gh"], p["gdensity"], p["gcap"], p["gdem"], p["gseed"]))
    return ev.analyse(params, ev.Opts(p["eps"], p["method"], p["budget"], p["variant"]))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["eps"] in C.EPSS and p["method"] in C.METHODS and p["budget"] in C.BUDGETS and p["variant"] in C.VARIANTS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["density"] - C.DENSITY_MIN) % 10 == 0 and p["spread"] % 25 == 0 and (p["load"] - C.LOAD_MIN) % 10 == 0 and (p["gdensity"] - C.GDENSITY_MIN) % 10 == 0


def test_setting_specs_have_room_to_move():
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS and p["gseed"] not in C.DIST_SEEDS, name


def test_defaults_equal_the_random_net_preset():
    p = C.PRESETS["🚚 Zufallsnetz"]
    assert p == {**C._BASE} and ev.DEFAULT_PARAMS.net == p["net"] and ev.DEFAULT_OPTS == ev.Opts(p["eps"], p["method"], p["budget"], p["variant"])


def test_the_variant_presets_change_only_what_their_name_says():
    base = C.PRESETS["🚚 Zufallsnetz"]
    for name, changed in (("⚡ Fleischer", ("method",)), ("🎯 Feine Näherung", ("eps",)), ("💰 Kostenbudget 75 %", ("budget",)), ("🚫 Ohne Skalierung", ("variant",))):
        assert all(C.PRESETS[name][k] == base[k] for k in base if k not in changed), name
        assert all(C.PRESETS[name][k] != base[k] for k in changed)


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"🔀 Preis-Wende", "🧩 Frachtnetz mit Bruch"}


def test_each_preset_shows_what_its_help_text_says():
    v = {name: ev.verdict(_analyse(p)) for name, p in C.PRESETS.items()}
    _, code, d = v["🚚 Zufallsnetz"]
    assert code == "ok" and d["opt"] == 67.0 and d["ratio"] > 0.98 and 0.88 <= d["certified"] <= 0.93 and 700 <= d["pushes"] <= 860 and d["calls"] == 3 * d["pushes"] and d["rows"] == 56
    _, code, d = v["🗺️ Streckennetz"]
    assert code == "ok" and d["opt"] == 7.0 and 0.91 <= d["ratio"] <= 0.95 and 0.85 <= d["certified"] <= 0.92 and 280 <= d["pushes"] <= 350
    _, code, d = v["🔀 Preis-Wende"]
    assert code == "ok" and d["opt"] == 2.0 and 1.9 <= d["primal"] <= 2.0 and 35 <= d["pushes"] <= 55
    _, code, d = v["🧩 Frachtnetz mit Bruch"]
    assert code == "ok" and d["opt"] == 1.5 and 1.38 <= d["primal"] <= 1.46 and d["columns"] == 3
    base, fl = v["🚚 Zufallsnetz"][2], v["⚡ Fleischer"][2]
    assert fl["calls"] < 0.45 * base["calls"] and abs(fl["pushes"] - base["pushes"]) < 0.05 * base["pushes"] and fl["ratio"] > 0.99 and fl["phases"] > 0
    _, code, d = v["🎯 Feine Näherung"]
    assert code == "ok" and 7000 <= d["pushes"] <= 8000 and d["ratio"] > 0.99 and d["certified"] > 0.99 and d["pushes"] > 8 * base["pushes"]
    _, code, d = v["💰 Kostenbudget 75 %"]
    assert code == "ok" and d["opt"] == 56.25 and d["budget"] == 1161.0 and d["cost"] <= 1161.0 + 1e-6 and 0.90 <= d["ratio"] <= 0.96
    assert base["cost_ratio"] > 1.15
    _, code, d = v["🚫 Ohne Skalierung"]
    assert code == "noscale" and d["primal"] > 2000 and 40 <= d["congestion"] <= 48


def test_the_presets_show_both_good_and_bad_news():
    codes = [ev.verdict(_analyse(p))[1] for p in C.PRESETS.values()]
    assert codes.count("ok") == 7 and codes.count("noscale") == 1

"""Tests for the device-level graph (glayout.schematic.graph, schema v1). No PDK needed.

Run with pytest, or directly: ``python tests/test_graph_schema.py``.
"""
import json
import random

from glayout.schematic import (
    SCHEMA_VERSION, HeteroGraph, annotate, curated, parse_analoggenie, schematic_graph, size_schematic,
    textbook_schematic, validate_graph,
)
from glayout.schematic.graph import EDGE_TYPES, FEATURES


def _row(d, ntype, name):
    t = d["node_types"][ntype]
    return dict(zip(t["feature_names"], t["x"][t["names"].index(name)]))


def _edges(d, etype):
    t = d["edge_types"].get(etype)
    if not t:
        return set()
    src = d["node_types"][t["src"]]["names"]
    dst = d["node_types"][t["dst"]]["names"]
    return {(src[a], dst[b]) for a, b in zip(*t["index"])}


def _c08(sized=True):
    sch = textbook_schematic("C08")
    ann = annotate(sch)
    return sch, ann, schematic_graph(sch, ann, size_schematic(sch, ann, 0.28, 0.5) if sized else None).to_dict()


def test_five_transistor_ota_graph():
    sch, ann, d = _c08()
    assert validate_graph(d) == [] and d["schema_version"] == SCHEMA_VERSION
    assert {t: len(v["names"]) for t, v in d["node_types"].items()} == {"device": 5, "pin": 20, "net": 8}
    assert d["node_types"]["net"]["names"][:6] == list(sch.ports)          # ports first
    assert len(_edges(d, "device__has__pin")) == 20 and len(_edges(d, "pin__in__net")) == 20
    assert ("M1.S", "net23") in _edges(d, "pin__in__net")
    assert _edges(d, "device__matched__device") == {("M1", "M2"), ("M2", "M1"), ("M4", "M5"), ("M5", "M4")}
    assert _edges(d, "device__symmetric__device") == _edges(d, "device__matched__device")


def test_device_pin_and_net_features():
    sch, ann, d = _c08()
    m4 = _row(d, "device", "M4")
    assert m4["kind=pfet"] == 1 and m4["diode"] == 1 and m4["block=CM"] == 1 and m4["struct=mirror"] == 1
    assert m4["sized"] == 1 and m4["w_um"] > 0 and m4["l_um"] in (0.56, 1.12)
    m3 = _row(d, "device", "M3")
    assert m3["block=BIAS"] == 1 and m3["matched"] == 0 and m3["diode"] == 0
    assert _row(d, "pin", "M3.B") == {**{k: 0.0 for k in FEATURES["pin"]}, "role=body": 1.0, "net=ground": 1.0}
    tail = _row(d, "net", "net23")
    assert (tail["pins"], tail["devices"], tail["sd_pins"], tail["gate_pins"], tail["port"]) == (3, 3, 3, 0, 0)
    assert _row(d, "net", "VB1")["class=bias"] == 1 and _row(d, "net", "VB1")["port"] == 1
    assert _row(d, "net", "VDD")["body_pins"] == 2


def test_unsized_and_unannotated_graphs():
    sch, ann, d = _c08(sized=False)
    assert all(_row(d, "device", n)["sized"] == 0 and _row(d, "device", n)["w_um"] == 0 for n in ("M1", "M4"))
    bare = schematic_graph(sch, annotate_missing=False).to_dict()
    assert validate_graph(bare) == []
    assert "device__matched__device" not in bare["edge_types"]
    assert sum(v for k, v in _row(bare, "device", "M1").items() if k.startswith("block=")) == 0


def test_bjt_base_is_not_a_body_and_composites_are_other():
    sch = parse_analoggenie("Q1 (net5 net5 VSS 0) npn\nX0 (IN OUT VDD VSS) INVERTER\nV0 (a 0) vsource\n", "mix",
                            ["VSS", "IN", "OUT", "VDD"], strict=False)
    d = schematic_graph(sch).to_dict()
    assert validate_graph(d) == []
    assert _row(d, "pin", "Q1.B")["role=base"] == 1 and _row(d, "pin", "Q1.B")["role=body"] == 0
    assert _row(d, "pin", "Q1.S")["role=substrate"] == 1
    assert _row(d, "pin", "X0.Q")["role=other"] == 1
    assert _row(d, "device", "X0")["kind=composite"] == 1 and _row(d, "device", "V0")["kind=unknown"] == 1
    assert _row(d, "net", "0")["class=ground"] == 1


def test_json_round_trip_is_exact_and_deterministic():
    _, _, a = _c08()
    _, _, b = _c08()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    again = HeteroGraph.from_dict(json.loads(json.dumps(a))).to_dict()
    assert again == a


def test_every_curated_and_random_schematic_gives_a_valid_graph():
    for name, sch in curated().items():
        ann = annotate(sch)
        d = schematic_graph(sch, ann, size_schematic(sch, ann, 0.15, 0.5)).to_dict()
        assert validate_graph(d) == [], name
        assert set(d["edge_types"]) <= set(EDGE_TYPES)
    rng = random.Random(1)
    nets = ["VDD", "VSS", "VB1", "VIN1", "VOUT1"] + [f"n{i}" for i in range(6)]
    kinds = ["nmos4", "pmos4", "resistor", "capacitor", "npn", "diode"]
    for case in range(200):
        lines = []
        for i in range(rng.randint(1, 10)):
            k = rng.choice(kinds)
            n = {"nmos4": 4, "pmos4": 4, "npn": 4}.get(k, 2)
            lines.append(f"X{i} ({' '.join(rng.choice(nets) for _ in range(n))}) {k}")
        sch = parse_analoggenie("\n".join(lines), f"fuzz{case}", ["VDD", "VSS", "VIN1"])
        assert validate_graph(schematic_graph(sch).to_dict()) == [], lines


def test_validate_catches_broken_graphs():
    _, _, d = _c08()
    bad = json.loads(json.dumps(d))
    bad["node_types"]["device"]["x"][0].pop()
    bad["edge_types"]["pin__in__net"]["index"][1][0] = 999
    probs = validate_graph(bad)
    assert any("device" in p for p in probs) and any("out of range" in p for p in probs)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

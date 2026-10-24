"""Tests for the device + block heterograph of a floorplan Circuit. Needs the gf180 PDK (about a minute).

Run with pytest, or directly: ``python tests/test_graph_circuit.py``.
"""
import json
import os

os.environ.setdefault("GLAYOUT_TABLES", "0")

from glayout import gf180 as pdk  # noqa: E402
from glayout.floorplan.anneal import anneal_floorplan  # noqa: E402
from glayout.floorplan.circuit import KINDS  # noqa: E402
from glayout.floorplan.drr import apply_drr  # noqa: E402
from glayout.floorplan.route_flow import route_result  # noqa: E402
from glayout.floorplan.router import RouterTech  # noqa: E402
from glayout.schematic import schematic_to_circuit, textbook_schematic, validate_graph  # noqa: E402
from glayout.schematic.vocab import BLOCK_KINDS  # noqa: E402


def _edges(d, etype):
    t = d["edge_types"].get(etype)
    if not t:
        return set()
    src, dst = d["node_types"][t["src"]]["names"], d["node_types"][t["dst"]]["names"]
    return {(src[a], dst[b]) for a, b in zip(*t["index"])}


def _row(d, ntype, name):
    t = d["node_types"][ntype]
    return dict(zip(t["feature_names"], t["x"][t["names"].index(name)]))


def test_block_vocabulary_matches_the_floorplanner():
    assert tuple(KINDS) == BLOCK_KINDS


def test_both_levels_of_the_five_transistor_ota():
    c = schematic_to_circuit(pdk, textbook_schematic("C08"))
    d = c.hetero_graph().to_dict()
    assert validate_graph(d) == []
    assert d["meta"]["level"] == "device+block"
    assert d["node_types"]["block"]["names"] == ["DP1", "CM1", "BIAS1"]
    assert _edges(d, "device__in__block") == {("M1", "DP1"), ("M2", "DP1"), ("M4", "CM1"), ("M5", "CM1"),
                                              ("M3", "BIAS1")}
    pin_nets = _edges(d, "block_pin__in__net")
    assert ("DP1.B", "VSS") in pin_nets                            # the pair's sources are the tail: own bulk pin
    assert not any(p == "CM1.B" for p, _ in pin_nets)              # the load's VDD rail is tied to its ring ...
    assert any(p.startswith("CM1.") and n == "VDD" for p, n in pin_nets)   # ... and is its one VDD pin
    # nets are shared: the block pins land on the device level's net nodes
    assert len(d["node_types"]["net"]["names"]) == len(textbook_schematic("C08").nets())
    dp = _row(d, "block", "DP1")
    assert dp["kind=DP"] == 1 and dp["nfet"] == 1 and dp["devices"] == 2 and dp["variants"] == 3
    assert dp["self_symmetric"] == 1 and dp["area_min_um2"] > 0
    assert dp["match=precise"] == 1 and _row(d, "block", "BIAS1")["match=precise"] == 0
    assert all(0 <= _row(d, "block_pin", n)["x_rel"] <= 1 for n in d["node_types"]["block_pin"]["names"])


def test_internal_nets_have_device_pins_but_no_block_pin():
    d = schematic_to_circuit(pdk, textbook_schematic("C06")).hetero_graph().to_dict()
    on_blocks = {n for _, n in _edges(d, "block_pin__in__net")}
    assert {"net9", "net10"} & on_blocks == set()
    assert ("M3.D", "net9") in _edges(d, "pin__in__net")


def test_no_floorplan_or_routing_result_leaks_into_the_features():
    c = schematic_to_circuit(pdk, textbook_schematic("C08"))
    before = json.dumps(c.hetero_graph().to_dict(), sort_keys=True)
    tech = RouterTech.from_pdk(pdk)
    apply_drr(pdk, [i.block for i in c.instances], scale=1.0, layers=tech.layers)
    assert any(v.pad.left > 0 for i in c.instances for v in i.block.variants)   # padding really changed
    fp = anneal_floorplan(c, seed=0)
    route_result(c, fp, tech)
    assert json.dumps(c.hetero_graph().to_dict(), sort_keys=True) == before


def test_hand_written_circuit_gets_the_block_level_only():
    from glayout.floorplan.block import Block, Pin, Variant
    from glayout.floorplan.circuit import Circuit, Instance

    def blk(name, kind, nets):
        pins = {n: Pin(n, 1.0, 1.0 + k, "EW", "met2") for k, n in enumerate(nets)}
        return Block(name, kind, [Variant({}, 4.0, 2.0 + len(nets), pins)])
    c = Circuit("abstract", [Instance("A", blk("A", "DP", ["X", "Y", "B"]), {"X": "N1", "Y": "N2", "B": "VSS"}),
                             Instance("L", blk("L", "CM", ["X", "Y", "B"]), {"X": "N1", "Y": "N2", "B": "VDD"})],
                self_symmetric=["A", "L"], ports=["N1", "VDD", "VSS"])
    d = c.hetero_graph().to_dict()
    assert validate_graph(d) == [] and d["meta"]["level"] == "block"
    assert set(d["node_types"]) == {"block", "variant", "block_pin", "net"}
    assert sorted(d["node_types"]["net"]["names"]) == ["N1", "N2", "VDD", "VSS"]
    assert _row(d, "net", "N1")["port"] == 1 and _row(d, "block", "L")["kind=CM"] == 1


def test_cell_parasitics_become_variant_loads_and_coupling_nodes():
    from glayout.schematic.model import net_class

    c = schematic_to_circuit(pdk, textbook_schematic("C08"))
    d0 = c.hetero_graph().to_dict()
    assert d0["meta"]["cell_parasitics"] is False and "cell_coupling" not in d0["node_types"]
    inst = c["DP1"]
    var = inst.block.variants[0]
    a, b = [p for p in var.pins if net_class(inst.nets[p]) not in ("power", "ground")][:2]
    var.parasitics = [(a, "VSUBS", 1.0e-15), (a, "B", 0.5e-15),     # to ground: substrate and the VSS body pin
                      (a, b, 0.25e-15),                              # between two signal nets
                      ("B", "VSUBS", 3.0e-15),                       # ground to ground: not a load
                      ("n99", a, 9.0e-15)]                           # a net the block does not export: skipped
    d = c.hetero_graph().to_dict()
    assert validate_graph(d) == [] and d["meta"]["cell_parasitics"] is True
    loads = d["edge_types"]["variant__loads__net"]
    assert loads["stage"] == "schematic" and loads["attr_names"] == ["c_fF"]
    got = {(d["node_types"]["variant"]["names"][v], d["node_types"]["net"]["names"][n]): at[0]
           for v, n, at in zip(*loads["index"], loads["attr"])}
    assert got == {("DP1.v0", inst.nets[a]): 1.5}
    cc = d["node_types"]["cell_coupling"]
    assert cc["stage"] == "schematic" and [r[0] for r in cc["x"]] == [0.25]
    assert _edges(d, "variant__has__cell_coupling") == {("DP1.v0", cc["names"][0])}
    assert _edges(d, "cell_coupling__on__net") == {(cc["names"][0], inst.nets[a]), (cc["names"][0], inst.nets[b])}


def test_legacy_block_graph_still_works_on_derived_circuits():
    c = schematic_to_circuit(pdk, textbook_schematic("C08"))
    legacy = c.graph().to_dict()
    assert {n["type"] for n in legacy["nodes"]} == {"device", "pin", "net"}
    assert sum(n["type"] == "device" for n in legacy["nodes"]) == 3          # the paper's "devices" are blocks


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

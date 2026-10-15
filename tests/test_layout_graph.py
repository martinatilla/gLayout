"""Tests for the layout stage of the learning graph (glayout.schematic.layout_graph, schema v2).
Needs the gf180 PDK (a few minutes: two circuits through the closed loop).

Run with pytest, or directly: ``python tests/test_layout_graph.py``.
"""
import json
import os
from collections import defaultdict
from functools import lru_cache

os.environ.setdefault("GLAYOUT_TABLES", "0")

from glayout import gf180 as pdk  # noqa: E402
from glayout.floorplan.route_flow import closed_loop_floorplan  # noqa: E402
from glayout.floorplan.router import RouterTech, _path_wire  # noqa: E402
from glayout.schematic import (  # noqa: E402
    HeteroGraph, curated, layout_graph, schematic_to_circuit, validate_graph,
)
from glayout.schematic.graph import LAYOUT_TYPES  # noqa: E402


@lru_cache(maxsize=None)
def _flow(key):
    c = schematic_to_circuit(pdk, curated()[key])
    before = c.hetero_graph().to_dict()
    loop = closed_loop_floorplan(pdk, c, [i.block for i in c.instances], RouterTech.from_pdk(pdk), seed=0)
    g = layout_graph(c, loop.floorplan, loop.routing, loop)
    return c, loop, before, g, g.to_dict()


def _edges(d, etype):
    t = d["edge_types"].get(etype)
    if not t:
        return []
    src, dst = d["node_types"][t["src"]]["names"], d["node_types"][t["dst"]]["names"]
    return [(src[a], dst[b]) for a, b in zip(*t["index"])]


def _y(d, ntype, name):
    t = d["node_types"][ntype]
    return dict(zip(t["label_names"], t["y"][t["names"].index(name)]))


def test_variants_are_schematic_nodes_one_per_shape():
    c, _, before, _, _ = _flow("C08")
    assert validate_graph(before) == []
    v = before["node_types"]["variant"]
    assert v["stage"] == "schematic"
    assert len(v["names"]) == sum(len(i.block.variants) for i in c.instances)
    has = _edges(before, "block__has__variant")
    assert sorted(b for b, _ in has) == sorted(i.name for i in c.instances for _ in i.block.variants)
    rows = [dict(zip(v["feature_names"], x)) for x in v["x"]]
    assert all(r["area_um2"] > 0 and r["fingers"] > 0 and r["area_rel"] >= 1 for r in rows)
    assert min(r["area_rel"] for r in rows) == 1                    # each block's smallest variant


def test_the_schematic_part_is_unchanged_by_layout():
    _, _, before, g, d = _flow("C08")
    assert validate_graph(d) == []
    assert g.schematic_part().to_dict() == before
    for t, v in d["node_types"].items():
        assert v["stage"] == ("layout" if t in LAYOUT_TYPES else "schematic")
        if t in before["node_types"]:
            assert v["names"] == before["node_types"][t]["names"] and v["x"] == before["node_types"][t]["x"]
    for t, v in before["edge_types"].items():
        assert d["edge_types"][t]["index"] == v["index"]


def test_exactly_one_chosen_variant_per_block_and_it_is_the_placed_one():
    c, loop, _, _, d = _flow("C08")
    for inst in c.instances:
        chosen = [n for n in d["node_types"]["variant"]["names"]
                  if n.startswith(inst.name + ".v") and _y(d, "variant", n)["chosen"] == 1]
        assert chosen == [f"{inst.name}.v{loop.floorplan.placed[inst.name].variant}"]
        b = _y(d, "block", inst.name)
        assert 0 < b["cx_rel"] < 1 and 0 < b["cy_rel"] < 1 and b["drr_scale"] == loop.scales[inst.name]


def test_segments_add_up_to_the_router():
    for key in ("C08", "two_stage_ota"):
        _, loop, _, _, d = _flow(key)
        rr = loop.routing
        seg = d["node_types"]["segment"]
        rows = [dict(zip(seg["feature_names"], x)) for x in seg["x"]]
        of = dict(_edges(d, "segment__of__net"))
        assert set(of) == set(seg["names"])
        wire, stub = defaultdict(float), defaultdict(float)
        for name, r in zip(seg["names"], rows):
            (stub if r["stub"] else wire)[of[name]] += r["length_um"]
        for net, plist in rr.paths.items():
            assert abs(wire[net] - _path_wire(rr.grid, plist)) < 1e-6, net
            assert abs(_y(d, "net", net)["wire_um"] - _path_wire(rr.grid, plist)) < 1e-4, net
        assert abs(sum(wire.values()) + sum(stub.values()) - rr.wirelength) < 1e-4
        net = d["node_types"]["net"]
        assert sum(r[net["label_names"].index("vias")] for r in net["y"]) == rr.vias
        assert all(sum(r[f"layer={l}"] for l in rr.tech.layers) == 1 for r in rows if not r["stub"])
        assert d["graph_labels"]["wirelength_um"] == round(rr.wirelength, 6)


def test_every_stub_serves_a_block_pin_of_its_own_net():
    _, _, _, _, d = _flow("two_stage_ota")
    of = dict(_edges(d, "segment__of__net"))
    pin_net = dict(_edges(d, "block_pin__in__net"))
    at = _edges(d, "segment__at__block_pin")
    assert at and all(pin_net[p] == of[s] for s, p in at)
    stubs = {n for n, x in zip(d["node_types"]["segment"]["names"], d["node_types"]["segment"]["x"])
             if x[d["node_types"]["segment"]["feature_names"].index("stub")] == 1}
    assert {s for s, _ in at} == stubs


def test_each_routed_net_is_one_connected_wire_tree():
    _, _, _, _, d = _flow("two_stage_ota")
    of = dict(_edges(d, "segment__of__net"))
    links = defaultdict(set)
    for a, b in _edges(d, "segment__connects__segment"):
        assert of[a] == of[b]
        links[a].add(b)
    by_net = defaultdict(set)
    for s, n in of.items():
        by_net[n].add(s)
    for net, segs in by_net.items():
        if _y(d, "net", net)["routed"] != 1:
            continue
        seen, todo = set(), [next(iter(segs))]
        while todo:
            s = todo.pop()
            if s not in seen:
                seen.add(s)
                todo += links[s]
        assert seen == segs, net


def test_coupling_is_symmetric_and_rolls_up_to_nets():
    _, _, _, _, d = _flow("two_stage_ota")
    for etype in ("segment__beside__segment", "net__coupled__net"):
        e = d["edge_types"].get(etype)
        if not e:
            continue
        attr = {pair: a for pair, a in zip(_edges(d, etype), e["attr"])}
        assert all(attr[(b, a)] == v for (a, b), v in attr.items())
        assert all(a != b for a, b in attr)
    of = dict(_edges(d, "segment__of__net"))
    seg_e = d["edge_types"].get("segment__beside__segment")
    net_par = defaultdict(float)
    if seg_e:
        for (a, b), (par, _) in zip(_edges(d, "segment__beside__segment"), seg_e["attr"]):
            assert of[a] != of[b]
            net_par[of[a]] += par
    for net in d["node_types"]["net"]["names"]:
        assert abs(_y(d, "net", net)["parallel_um"] - net_par[net]) < 1e-4


def test_json_round_trip_keeps_labels_and_attributes():
    _, _, _, _, d = _flow("C08")
    assert HeteroGraph.from_dict(json.loads(json.dumps(d))).to_dict() == d


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

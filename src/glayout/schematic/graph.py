"""Heterogeneous circuit graph for learning (schema v1).

Two levels share one set of net nodes:

- **device level** (from a ``Schematic``; no PDK needed): ``device`` nodes
  (one per transistor or passive), ``pin`` nodes (one per device terminal) and
  ``net`` nodes; edges ``device→pin``, ``pin→net``, and ``device↔device`` for
  matched and symmetric devices (from the annotation).
- **block level** (from a floorplan ``Circuit``): ``block`` nodes (one per
  instance) and ``block_pin`` nodes (one per exported pin of its smallest
  variant); edges ``block→block_pin``, ``block_pin→net``, ``block↔block``
  for symmetric block pairs, and ``device→block`` linking the levels.

Edges are stored in one direction only (both directions for the symmetric
relations); a loader adds reverse edges if it wants them. Feature names are
stored with the graph, so a reader needs neither gLayout nor this module.

**Leakage rule.** Nothing a floorplan or a router produces is a feature: no
positions, padding, DRR scales or routing results. Block shapes are features,
because they are known before floorplanning. ``Circuit.graph()`` (the
block-only graph of the floorplanning paper) is unchanged; this is
``Circuit.hetero_graph()``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from glayout.schematic.annotate import Annotation, annotate
from glayout.schematic.model import MOS, Schematic, net_class
from glayout.schematic.sizing import Sizing
from glayout.schematic.vocab import (
    BLOCK_KINDS, DEVICE_KINDS, LAYERS, NET_CLASSES, SIDES, STRUCTURES, TERMINALS, device_kind_class, one_hot,
    terminal_role,
)

SCHEMA_VERSION = "1.0"

FEATURES: dict[str, list[str]] = {
    "device": [*(f"kind={k}" for k in DEVICE_KINDS), "sized", "w_um", "l_um", "fingers_base", "diode",
               *(f"block={k}" for k in BLOCK_KINDS), *(f"struct={s}" for s in STRUCTURES),
               "matched", "symmetric", "matched_set_size"],
    "pin": [*(f"role={t}" for t in TERMINALS), *(f"net={c}" for c in NET_CLASSES)],
    "net": ["pins", "devices", "gate_pins", "sd_pins", "body_pins", "port", *(f"class={c}" for c in NET_CLASSES)],
    "block": [*(f"kind={k}" for k in BLOCK_KINDS), "nfet", "pfet", "devices", "variants", "area_min_um2",
              "width_um", "height_um", "aspect", "self_symmetric", "symmetric_pair"],
    "block_pin": [*(f"side={s}" for s in SIDES), *(f"net={c}" for c in NET_CLASSES),
                  *(f"layer={l}" for l in LAYERS), "x_rel", "y_rel"],
}
EDGE_TYPES: dict[str, tuple[str, str]] = {
    "device__has__pin": ("device", "pin"),
    "pin__in__net": ("pin", "net"),
    "device__matched__device": ("device", "device"),
    "device__symmetric__device": ("device", "device"),
    "device__in__block": ("device", "block"),
    "block__has__block_pin": ("block", "block_pin"),
    "block_pin__in__net": ("block_pin", "net"),
    "block__symmetric__block": ("block", "block"),
}
_GATE = {("nfet", "G"), ("pfet", "G"), ("npn", "B"), ("pnp", "B")}
_SD = {("nfet", "D"), ("nfet", "S"), ("pfet", "D"), ("pfet", "S"), ("npn", "C"), ("npn", "E"), ("pnp", "C"), ("pnp", "E")}
_BODY = {("nfet", "B"), ("pfet", "B"), ("npn", "S"), ("pnp", "S")}


@dataclass
class HeteroGraph:
    name: str
    nodes: dict[str, dict] = field(default_factory=dict)     # type -> {"names": [...], "x": [[...]]}
    edges: dict[str, list[list[int]]] = field(default_factory=dict)   # type -> [src indices, dst indices]
    meta: dict = field(default_factory=dict)

    def _add_node(self, ntype: str, name: str, x: list[float]) -> int:
        t = self.nodes.setdefault(ntype, {"names": [], "x": []})
        t["names"].append(name)
        t["x"].append([round(float(v), 6) for v in x])
        return len(t["names"]) - 1

    def _add_edge(self, etype: str, src: int, dst: int) -> None:
        e = self.edges.setdefault(etype, [[], []])
        e[0].append(src)
        e[1].append(dst)

    def counts(self) -> dict:
        return {**{t: len(v["names"]) for t, v in self.nodes.items()},
                **{t: len(v[0]) for t, v in self.edges.items()}}

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "meta": self.meta,
            "node_types": {t: {"feature_names": FEATURES[t], "names": v["names"], "x": v["x"]}
                           for t, v in self.nodes.items()},
            "edge_types": {t: {"src": EDGE_TYPES[t][0], "dst": EDGE_TYPES[t][1], "index": v}
                           for t, v in self.edges.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "HeteroGraph":
        g = HeteroGraph(d["name"], meta=dict(d.get("meta", {})))
        g.nodes = {t: {"names": list(v["names"]), "x": [list(r) for r in v["x"]]} for t, v in d["node_types"].items()}
        g.edges = {t: [list(v["index"][0]), list(v["index"][1])] for t, v in d["edge_types"].items()}
        return g


def validate_graph(d: dict) -> list[str]:
    """Shape problems of a graph dict (empty when it is well formed)."""
    out = []
    if d.get("schema_version") != SCHEMA_VERSION:
        out.append(f"schema_version {d.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    sizes = {}
    for t, v in d.get("node_types", {}).items():
        sizes[t] = len(v["names"])
        if len(v["x"]) != len(v["names"]):
            out.append(f"{t}: {len(v['x'])} feature rows for {len(v['names'])} nodes")
        bad = [i for i, r in enumerate(v["x"]) if len(r) != len(v["feature_names"])]
        if bad:
            out.append(f"{t}: rows {bad[:5]} do not have {len(v['feature_names'])} features")
    for t, v in d.get("edge_types", {}).items():
        s, dst = v["src"], v["dst"]
        if s not in sizes or dst not in sizes:
            out.append(f"{t}: endpoint type missing ({s}, {dst})")
            continue
        src_i, dst_i = v["index"]
        if len(src_i) != len(dst_i):
            out.append(f"{t}: index rows differ in length")
        if any(not 0 <= i < sizes[s] for i in src_i) or any(not 0 <= i < sizes[dst] for i in dst_i):
            out.append(f"{t}: index out of range")
    return out


# ---- device level ------------------------------------------------------------
def _device_level(g: HeteroGraph, sch: Schematic, ann: Optional[Annotation], sizing: Optional[Sizing]) -> dict[str, int]:
    """Adds device, pin and net nodes and their edges. Returns net name -> net node index."""
    block_of, structs, matched_of, sym, set_size = {}, {}, set(), set(), {}
    if ann is not None:
        for b in ann.blocks:
            for d in b.devices:
                block_of[d] = b.kind
        for s in ann.structures:
            for d in s.devices:
                structs.setdefault(d, set()).add(s.kind)
        for m in ann.matched:
            for d in m.devices:
                matched_of.add(d)
                set_size[d] = max(set_size.get(d, 0), len(m.devices))
                if m.symmetric:
                    sym.add(d)
    sizes = {}
    if sizing is not None:
        for b in sizing.blocks.values():
            for d, n in b.base.items():
                sizes[d] = (b.total_width(d), b.length, n)

    net_pins = sch.net_pins()
    net_idx = {}
    for net in sch.nets():
        pins = net_pins.get(net, [])
        kinds = [(sch[d].kind, p) for d, p in pins]
        net_idx[net] = g._add_node("net", net, [
            len(pins), len({d for d, _ in pins}),
            sum(k in _GATE for k in kinds), sum(k in _SD for k in kinds), sum(k in _BODY for k in kinds),
            float(net in sch.ports), *one_hot(net_class(net), NET_CLASSES)])

    dev_idx = {}
    for dev in sch.devices:
        w, l, n = sizes.get(dev.name, (0.0, 0.0, 0))
        diode = dev.kind in MOS and dev.net("D") == dev.net("G")
        dev_idx[dev.name] = di = g._add_node("device", dev.name, [
            *one_hot(device_kind_class(dev.kind), DEVICE_KINDS), float(dev.name in sizes), w, l, n, float(diode),
            *one_hot(block_of.get(dev.name), BLOCK_KINDS),
            *[float(s in structs.get(dev.name, ())) for s in STRUCTURES],
            float(dev.name in matched_of), float(dev.name in sym), set_size.get(dev.name, 0)])
        for pin, net in dev.pins.items():
            pi = g._add_node("pin", f"{dev.name}.{pin}", [*one_hot(terminal_role(dev.kind, pin), TERMINALS),
                                                          *one_hot(net_class(net), NET_CLASSES)])
            g._add_edge("device__has__pin", di, pi)
            g._add_edge("pin__in__net", pi, net_idx[net])

    if ann is not None:
        for m in ann.matched:
            for a in m.devices:
                for b in m.devices:
                    if a != b:
                        g._add_edge("device__matched__device", dev_idx[a], dev_idx[b])
                        if m.symmetric:
                            g._add_edge("device__symmetric__device", dev_idx[a], dev_idx[b])
    g.meta["devices"] = dev_idx
    return net_idx



def schematic_graph(sch: Schematic, ann: Optional[Annotation] = None, sizing: Optional[Sizing] = None,
                    annotate_missing: bool = True) -> HeteroGraph:
    """Device-level graph. Annotates ``sch`` when no annotation is given (unless ``annotate_missing=False``)."""
    if ann is None and annotate_missing:
        ann = annotate(sch)
    g = HeteroGraph(sch.name, meta={"level": "device", "provenance": sch.provenance,
                                    "sizing": None if sizing is None else {"seed": sizing.seed, "note": sizing.note},
                                    "annotated": ann is not None})
    _device_level(g, sch, ann, sizing)
    del g.meta["devices"]
    return g


def circuit_graph(circuit) -> HeteroGraph:
    """Both levels for a floorplan ``Circuit`` built from a schematic; the block level alone otherwise."""
    sch: Optional[Schematic] = circuit.schematic
    ann: Optional[Annotation] = circuit.annotation
    g = HeteroGraph(circuit.name, meta={"level": "device+block" if sch is not None else "block",
                                        "provenance": getattr(sch, "provenance", {}),
                                        "sizing": None if circuit.sizing is None else
                                        {"seed": circuit.sizing.seed, "note": circuit.sizing.note}})
    if sch is not None:
        net_idx = _device_level(g, sch, ann, circuit.sizing)
        dev_idx = g.meta.pop("devices")
    else:
        net_idx, dev_idx = {}, {}
        for net, pins in circuit.net_pins().items():
            insts = {i for i, _ in pins}
            net_idx[net] = g._add_node("net", net, [len(pins), len(insts), 0, 0, 0, float(net in circuit.ports),
                                                    *one_hot(net_class(net), NET_CLASSES)])
    ann_blocks = {b.name: b for b in ann.blocks} if ann is not None else {}
    paired = {n for p in circuit.symmetric_pairs for n in p}
    blk_idx = {}
    for inst in circuit.instances:
        v = min(inst.block.variants, key=lambda v: v.area)
        ab = ann_blocks.get(inst.name)
        kind = inst.block.kind if inst.block.kind in BLOCK_KINDS else "OTHER"
        blk_idx[inst.name] = bi = g._add_node("block", inst.name, [
            *one_hot(kind, BLOCK_KINDS), float(ab is not None and ab.device_kind == "nfet"),
            float(ab is not None and ab.device_kind == "pfet"), len(ab.devices) if ab else 0,
            len(inst.block.variants), v.area, v.width, v.height, v.width / v.height if v.height else 0.0,
            float(inst.name in circuit.self_symmetric), float(inst.name in paired)])
        for bnet, pin in v.pins.items():
            cnet = inst.nets[bnet]
            if cnet not in net_idx:
                net_idx[cnet] = g._add_node("net", cnet, [0, 0, 0, 0, 0, float(cnet in circuit.ports),
                                                          *one_hot(net_class(cnet), NET_CLASSES)])
            pi = g._add_node("block_pin", f"{inst.name}.{bnet}", [
                *[float(s in pin.sides) for s in SIDES], *one_hot(net_class(cnet), NET_CLASSES),
                *one_hot(pin.layer, LAYERS), pin.x / v.width if v.width else 0.0,
                pin.y / v.height if v.height else 0.0])
            g._add_edge("block__has__block_pin", bi, pi)
            g._add_edge("block_pin__in__net", pi, net_idx[cnet])
        if ab is not None:
            for d in ab.devices:
                g._add_edge("device__in__block", dev_idx[d], bi)
    for a, b in circuit.symmetric_pairs:
        g._add_edge("block__symmetric__block", blk_idx[a], blk_idx[b])
        g._add_edge("block__symmetric__block", blk_idx[b], blk_idx[a])
    return g

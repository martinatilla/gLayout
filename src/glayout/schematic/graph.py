"""Heterogeneous circuit graph for learning (schema v2).

Two levels share one set of net nodes:

- **device level** (from a ``Schematic``; no PDK needed): ``device`` nodes
  (one per transistor or passive), ``pin`` nodes (one per device terminal) and
  ``net`` nodes; edges ``device→pin``, ``pin→net``, and ``device↔device`` for
  matched and symmetric devices (from the annotation).
- **block level** (from a floorplan ``Circuit``): ``block`` nodes (one per
  instance), ``variant`` nodes (one per candidate shape of a block, so choosing
  a shape is choosing a node) and ``block_pin`` nodes (one per exported pin of
  its smallest variant); edges ``block→variant``, ``block→block_pin``,
  ``block_pin→net``, ``block↔block`` for symmetric block pairs, and
  ``device→block`` linking the levels.

Both levels are stage ``schematic``. ``glayout.schematic.layout_graph`` adds
the stage ``layout`` to the same graph without re-indexing anything: labels
(``y``) on existing nodes, ``segment`` nodes for the routed wires, and
coupling edges with attributes.

Edges are stored in one direction only (both directions for the symmetric
relations); a loader adds reverse edges if it wants them. Feature, label and
edge-attribute names are stored with the graph, so a reader needs neither
gLayout nor this module.

**Leakage rule.** Nothing a floorplan or a router produces is a feature of a
``schematic`` node: no positions, padding, DRR scales or routing results.
Block and variant shapes are features, because they are known before
floorplanning. ``Circuit.graph()`` (the block-only graph of the floorplanning
paper) is unchanged; this is ``Circuit.hetero_graph()``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from glayout.schematic.annotate import Annotation, annotate
from glayout.schematic.model import MOS, Schematic, net_class
from glayout.schematic.sizing import Sizing
from glayout.schematic.vocab import (
    BLOCK_KINDS, DEVICE_KINDS, LAYERS, MATCH_CLASSES, NET_CLASSES, SENSITIVITY, SIDES, STRUCTURES, TERMINALS,
    device_kind_class, one_hot, terminal_role,
)

# 1.1: matching class (device, block) and net sensitivity features
# 2.0: variant nodes; stages; layout labels, segment nodes and coupling edges (glayout.schematic.layout_graph)
SCHEMA_VERSION = "2.0"

FEATURES: dict[str, list[str]] = {
    "device": [*(f"kind={k}" for k in DEVICE_KINDS), "sized", "w_um", "l_um", "fingers_base", "diode",
               *(f"block={k}" for k in BLOCK_KINDS), *(f"struct={s}" for s in STRUCTURES),
               "matched", "symmetric", "matched_set_size", *(f"match={c}" for c in MATCH_CLASSES)],
    "pin": [*(f"role={t}" for t in TERMINALS), *(f"net={c}" for c in NET_CLASSES)],
    "net": ["pins", "devices", "gate_pins", "sd_pins", "body_pins", "port", *(f"class={c}" for c in NET_CLASSES),
            *(f"sens={c}" for c in SENSITIVITY)],
    "block": [*(f"kind={k}" for k in BLOCK_KINDS), "nfet", "pfet", "devices", "variants", "area_min_um2",
              "width_um", "height_um", "aspect", "self_symmetric", "symmetric_pair",
              *(f"match={c}" for c in MATCH_CLASSES)],
    "block_pin": [*(f"side={s}" for s in SIDES), *(f"net={c}" for c in NET_CLASSES),
                  *(f"layer={l}" for l in LAYERS), "x_rel", "y_rel"],
    "variant": ["fingers", "finger_w_um", "length_um", "width_um", "height_um", "area_um2", "aspect", "area_rel",
                "pins", "centroid_error_um"],
    # layout stage: one straight wire on one layer (between vias), or a landing stub
    "segment": [*(f"layer={l}" for l in LAYERS), "stub", "horizontal", "length_um",
                "x0_rel", "y0_rel", "x1_rel", "y1_rel"],
}
# layout labels on schematic-stage nodes (glayout.schematic.layout_graph)
LABELS: dict[str, list[str]] = {
    "variant": ["chosen"],
    "block": ["x_um", "y_um", "cx_rel", "cy_rel", "mirrored", "drr_scale"],
    "net": ["routed", "failed", "unreachable_pins", "landings", "wire_um", "stub_um", "wirelength_um", "vias",
            "segments", "parallel_um", "crossings"],
}
GRAPH_LABELS = ["routed", "area_um2", "width_um", "height_um", "dead_space", "hpwl_um", "wirelength_um", "vias",
                "failed_nets", "unreachable_pins", "rounds"]
EDGE_TYPES: dict[str, tuple[str, str]] = {
    "device__has__pin": ("device", "pin"),
    "pin__in__net": ("pin", "net"),
    "device__matched__device": ("device", "device"),
    "device__symmetric__device": ("device", "device"),
    "device__in__block": ("device", "block"),
    "block__has__block_pin": ("block", "block_pin"),
    "block_pin__in__net": ("block_pin", "net"),
    "block__symmetric__block": ("block", "block"),
    "block__has__variant": ("block", "variant"),
    "segment__of__net": ("segment", "net"),
    "segment__at__block_pin": ("segment", "block_pin"),
    "segment__connects__segment": ("segment", "segment"),
    "segment__beside__segment": ("segment", "segment"),
    "net__coupled__net": ("net", "net"),
}
EDGE_ATTRS: dict[str, list[str]] = {
    "segment__beside__segment": ["parallel_um", "crossings"],
    "net__coupled__net": ["parallel_um", "crossings"],
}
LAYOUT_TYPES = frozenset({"segment", "segment__of__net", "segment__at__block_pin", "segment__connects__segment",
                          "segment__beside__segment", "net__coupled__net"})


def stage(type_name: str) -> str:
    """``layout`` for node and edge types that exist only after layout, ``schematic`` otherwise."""
    return "layout" if type_name in LAYOUT_TYPES else "schematic"
_GATE = {("nfet", "G"), ("pfet", "G"), ("npn", "B"), ("pnp", "B")}
_SD = {("nfet", "D"), ("nfet", "S"), ("pfet", "D"), ("pfet", "S"), ("npn", "C"), ("npn", "E"), ("pnp", "C"), ("pnp", "E")}
_BODY = {("nfet", "B"), ("pfet", "B"), ("npn", "S"), ("pnp", "S")}


def _row(x) -> list[float]:
    return [round(float(v), 6) for v in x]


@dataclass
class HeteroGraph:
    name: str
    nodes: dict[str, dict] = field(default_factory=dict)     # type -> {"names": [...], "x": [[...]]}
    edges: dict[str, list[list[int]]] = field(default_factory=dict)   # type -> [src indices, dst indices]
    meta: dict = field(default_factory=dict)
    labels: dict[str, list[list[float]]] = field(default_factory=dict)     # node type -> rows (LABELS)
    edge_attr: dict[str, list[list[float]]] = field(default_factory=dict)  # edge type -> rows (EDGE_ATTRS)
    graph_labels: dict[str, float] = field(default_factory=dict)           # GRAPH_LABELS

    def _add_node(self, ntype: str, name: str, x: list[float]) -> int:
        t = self.nodes.setdefault(ntype, {"names": [], "x": []})
        t["names"].append(name)
        t["x"].append(_row(x))
        return len(t["names"]) - 1

    def _add_edge(self, etype: str, src: int, dst: int, attr: Optional[list[float]] = None) -> None:
        e = self.edges.setdefault(etype, [[], []])
        e[0].append(src)
        e[1].append(dst)
        if etype in EDGE_ATTRS:
            self.edge_attr.setdefault(etype, []).append(_row(attr))

    def _set_labels(self, ntype: str, rows: list[list[float]]) -> None:
        self.labels[ntype] = [_row(r) for r in rows]

    def index(self, ntype: str) -> dict[str, int]:
        """node name -> index, for one node type"""
        return {n: i for i, n in enumerate(self.nodes.get(ntype, {"names": []})["names"])}

    def counts(self) -> dict:
        return {**{t: len(v["names"]) for t, v in self.nodes.items()},
                **{t: len(v[0]) for t, v in self.edges.items()}}

    def to_dict(self) -> dict:
        nodes = {}
        for t, v in self.nodes.items():
            nodes[t] = {"stage": stage(t), "feature_names": FEATURES[t], "names": v["names"], "x": v["x"]}
            if t in self.labels:
                nodes[t].update({"label_names": LABELS[t], "y": self.labels[t]})
        edges = {}
        for t, v in self.edges.items():
            edges[t] = {"stage": stage(t), "src": EDGE_TYPES[t][0], "dst": EDGE_TYPES[t][1], "index": v}
            if t in EDGE_ATTRS:
                edges[t].update({"attr_names": EDGE_ATTRS[t], "attr": self.edge_attr.get(t, [])})
        d = {"schema_version": SCHEMA_VERSION, "name": self.name, "meta": self.meta,
             "node_types": nodes, "edge_types": edges}
        if self.graph_labels:
            d["graph_labels"] = dict(self.graph_labels)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "HeteroGraph":
        g = HeteroGraph(d["name"], meta=dict(d.get("meta", {})))
        g.nodes = {t: {"names": list(v["names"]), "x": [list(r) for r in v["x"]]} for t, v in d["node_types"].items()}
        g.edges = {t: [list(v["index"][0]), list(v["index"][1])] for t, v in d["edge_types"].items()}
        g.labels = {t: [list(r) for r in v["y"]] for t, v in d["node_types"].items() if "y" in v}
        g.edge_attr = {t: [list(r) for r in v["attr"]] for t, v in d["edge_types"].items() if "attr" in v}
        g.graph_labels = dict(d.get("graph_labels", {}))
        return g

    def schematic_part(self) -> "HeteroGraph":
        """The same graph without the layout stage: no layout types, labels or graph labels."""
        g = HeteroGraph(self.name, meta={k: v for k, v in self.meta.items() if k != "layout"})
        g.nodes = {t: {"names": list(v["names"]), "x": [list(r) for r in v["x"]]}
                   for t, v in self.nodes.items() if stage(t) == "schematic"}
        g.edges = {t: [list(v[0]), list(v[1])] for t, v in self.edges.items() if stage(t) == "schematic"}
        g.edge_attr = {t: [list(r) for r in v] for t, v in self.edge_attr.items() if stage(t) == "schematic"}
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
        if "y" in v:
            if len(v["y"]) != len(v["names"]):
                out.append(f"{t}: {len(v['y'])} label rows for {len(v['names'])} nodes")
            if any(len(r) != len(v["label_names"]) for r in v["y"]):
                out.append(f"{t}: label rows do not have {len(v['label_names'])} labels")
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
        if "attr" in v and (len(v["attr"]) != len(src_i) or any(len(r) != len(v["attr_names"]) for r in v["attr"])):
            out.append(f"{t}: attribute rows do not match the edges")
    return out


# ---- device level ------------------------------------------------------------
def _device_level(g: HeteroGraph, sch: Schematic, ann: Optional[Annotation], sizing: Optional[Sizing]) -> dict[str, int]:
    """Adds device, pin and net nodes and their edges. Returns net name -> net node index."""
    block_of, structs, matched_of, sym, set_size, precision = {}, {}, set(), set(), {}, {}
    rank = {c: i for i, c in enumerate(MATCH_CLASSES)}
    sens = ann.net_sensitivity if ann is not None else {}
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
                if d not in precision or rank[m.precision] < rank[precision[d]]:
                    precision[d] = m.precision
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
            float(net in sch.ports), *one_hot(net_class(net), NET_CLASSES), *one_hot(sens.get(net), SENSITIVITY)])

    dev_idx = {}
    for dev in sch.devices:
        w, l, n = sizes.get(dev.name, (0.0, 0.0, 0))
        diode = dev.kind in MOS and dev.net("D") == dev.net("G")
        dev_idx[dev.name] = di = g._add_node("device", dev.name, [
            *one_hot(device_kind_class(dev.kind), DEVICE_KINDS), float(dev.name in sizes), w, l, n, float(diode),
            *one_hot(block_of.get(dev.name), BLOCK_KINDS),
            *[float(s in structs.get(dev.name, ())) for s in STRUCTURES],
            float(dev.name in matched_of), float(dev.name in sym), set_size.get(dev.name, 0),
            *one_hot(precision.get(dev.name), MATCH_CLASSES)])
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


def _variant_features(v, min_area: float) -> list[float]:
    """Shape of one candidate variant. Finger counts and widths come from finger-sequence params (0 otherwise)."""
    devices = v.params.get("devices", [])
    fingers = sum(d.get("fingers", 0) for d in devices) if isinstance(devices, list) else 0
    cerr = v.component.info.get("centroid_error_um", 0.0) if v.component is not None else 0.0
    return [fingers, v.params.get("width", 0.0), v.params.get("length", 0.0), v.width, v.height, v.area,
            v.width / v.height if v.height else 0.0, v.area / min_area if min_area else 0.0, len(v.pins), cerr]


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
                                                    *one_hot(net_class(net), NET_CLASSES),
                                                    *one_hot(None, SENSITIVITY)])
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
            float(inst.name in circuit.self_symmetric), float(inst.name in paired),
            *one_hot(ab.matching if ab else None, MATCH_CLASSES)])
        for k, var in enumerate(inst.block.variants):
            vi = g._add_node("variant", f"{inst.name}.v{k}", _variant_features(var, inst.block.area))
            g._add_edge("block__has__variant", bi, vi)
        for bnet, pin in v.pins.items():
            cnet = inst.nets[bnet]
            if cnet not in net_idx:
                net_idx[cnet] = g._add_node("net", cnet, [0, 0, 0, 0, 0, float(cnet in circuit.ports),
                                                          *one_hot(net_class(cnet), NET_CLASSES),
                                                          *one_hot(None, SENSITIVITY)])
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

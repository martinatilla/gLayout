"""Layout stage of the learning graph (schema v2): the graph of a floorplan ``Circuit`` with what
floorplanning and routing produced added to it.

Nothing is re-indexed: every node, edge and feature row of ``circuit_graph(circuit)`` is kept as is
(``layout_graph(...).schematic_part()`` gives it back), so one training sample is the schematic
graph and the same graph with layout labels. The layout adds:

- **labels** (``y``, names in ``graph.LABELS``) on existing nodes: which variant each block uses
  (``variant.chosen``); where each block sits; per net whether the router connected it
  (``routed``: two or more landings, all joined) or gave up (``failed``: rip-up and reroute failed,
  or a pin got no landing), its wirelength on the router layers and in landing stubs, its vias,
  and how far it runs beside or crosses other nets.
  Graph-level labels (``graph.GRAPH_LABELS``): area, dead space, wirelength, routed, rounds.
- **segment** nodes: each straight router wire (one layer, between two vias or a via and a
  junction) and each landing stub, with edges ``segment__of__net``, ``segment__at__block_pin``
  (a stub and the pin it serves) and ``segment__connects__segment`` (a via, a stub's landing or a
  junction where a net's later path meets its earlier ones; both directions).
- **coupling**: ``segment__beside__segment`` and ``net__coupled__net`` (both directions), with
  ``parallel_um`` (length on adjacent tracks of the same router layer) and ``crossings`` (grid
  points where one runs over the other).

Coupling here is geometric, from the router grid: it says which wires sit next to which, not how
many fF that costs. Extracted capacitance is a later label on the same nodes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from glayout.schematic.graph import HeteroGraph, circuit_graph
from glayout.schematic.vocab import LAYERS, one_hot

H = 0       # router layer 0 carries horizontal wires (glayout.floorplan.router.H)


def _runs(grid, path) -> list[list[tuple]]:
    """A path cut at its vias: runs of consecutive nodes on one layer."""
    runs, cur = [], [path[0]]
    for a, b in zip(path, path[1:]):
        if grid.layer(a) == grid.layer(b):
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return runs


def _length(grid, run) -> float:
    return sum(abs(ax - bx) + abs(ay - by)
               for (ax, ay), (bx, by) in ((grid.xy(a), grid.xy(b)) for a, b in zip(run, run[1:])))


def _steps(run):
    """Grid steps of a run as (layer, (i, j), (i', j')) with the ends ordered; pin nodes have none."""
    for a, b in zip(run, run[1:]):
        if a[0] != "P" and b[0] != "P":
            lo, hi = sorted(((a[1], a[2]), (b[1], b[2])))
            yield (a[0], lo, hi)


def _beside(step, d: int):
    z, (i1, j1), (i2, j2) = step
    return (z, (i1, j1 + d), (i2, j2 + d)) if z == H else (z, (i1 + d, j1), (i2 + d, j2))


class _Segments:
    """Segment nodes as they are added: net, geometry, and the grid steps/points they occupy."""

    def __init__(self, g: HeteroGraph, width: float, height: float):
        self.g, self.w, self.h = g, width or 1.0, height or 1.0
        self.net: list[str] = []
        self.count: dict[str, int] = defaultdict(int)
        self.steps: dict[tuple, set[int]] = defaultdict(set)
        self.points: dict[tuple, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        self.links: set[tuple[int, int]] = set()

    def add(self, net: str, layer: str, stub: bool, horizontal: bool, length: float, xy: list) -> int:
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        s = self.g._add_node("segment", f"{net}.s{self.count[net]}", [
            *one_hot(layer, LAYERS), float(stub), float(horizontal), length,
            min(xs) / self.w, min(ys) / self.h, max(xs) / self.w, max(ys) / self.h])
        self.net.append(net)
        self.count[net] += 1
        return s

    def occupy(self, s: int, run) -> None:
        for e in _steps(run):
            self.steps[e].add(s)
        for n in run:
            if n[0] != "P":
                self.points[(n[1], n[2])][n[0]].add(s)

    def link(self, a: Optional[int], b: Optional[int]) -> None:
        if a is not None and b is not None and a != b:
            self.links.add((a, b))
            self.links.add((b, a))


def layout_graph(circuit, fp, rr, loop=None) -> HeteroGraph:
    """The schematic graph of ``circuit`` with the layout of ``fp`` (``FloorplanResult``) and ``rr``
    (``RoutingResult``) added. ``loop`` (``ClosedLoopResult``) adds DRR scales and the round count."""
    g = circuit_graph(circuit)
    grid, tech = rr.grid, rr.tech
    m = fp.metrics
    width, height = m["width"], m["height"]
    scales = loop.scales if loop is not None else {}
    g.meta["layout"] = {"router_layers": list(tech.layers), "pitch_um": tech.pitch,
                        "closed_loop": loop is not None}

    # ---- placement: chosen variant, position ---------------------------------------
    chosen = {f"{i.name}.v{fp.placed[i.name].variant}" for i in circuit.instances}
    g._set_labels("variant", [[float(n in chosen)] for n in g.nodes["variant"]["names"]])
    rows = []
    for name in g.nodes["block"]["names"]:
        p = fp.placed[name]
        v = circuit[name].block.variants[p.variant]
        x, y = tech.snap(p.x + v.pad.left), tech.snap(p.y + v.pad.bottom)
        rows.append([x, y, (x + v.width / 2) / width, (y + v.height / 2) / height, float(p.mirrored),
                     scales.get(name, 0.0)])
    g._set_labels("block", rows)

    # ---- segments ------------------------------------------------------------------
    net_idx, pin_idx = g.index("net"), g.index("block_pin")
    segs = _Segments(g, width, height)
    per_net: dict[str, dict] = defaultdict(lambda: {"wire": 0.0, "stub": 0.0, "vias": 0, "segments": 0})
    for net in sorted(set(rr.landings) | set(rr.paths)):
        stats = per_net[net]
        at: dict[tuple, int] = {}           # path node -> a segment of this net through it
        for land in rr.landings.get(net, []):
            o = land.option
            length = abs(land.x - o.x) + abs(land.y - o.y)
            s = segs.add(net, o.layer, True, o.side in "EW", length, [(o.x, o.y), (land.x, land.y)])
            g._add_edge("segment__of__net", s, net_idx[net])
            if land.pin.owner in pin_idx:
                g._add_edge("segment__at__block_pin", s, pin_idx[land.pin.owner])
            at[land.node] = s
            stats["stub"] += length
            stats["vias"] += 1
        for path in rr.paths.get(net, []):
            before = dict(at)
            kept, stacks = [], []           # stacks: (runs without wire, the segment kept before them)
            for run in _runs(grid, path):
                length = _length(grid, run)
                if length <= 1e-9:          # a via stack: no wire on this layer
                    stacks.append((run, kept[-1] if kept else None))
                    continue
                z = grid.layer(run[0])
                s = segs.add(net, tech.layers[z], False, z == H, length, [grid.xy(n) for n in run])
                g._add_edge("segment__of__net", s, net_idx[net])
                segs.occupy(s, run)
                if kept:
                    segs.link(kept[-1], s)
                for n in run:
                    at.setdefault(n, s)
                kept.append(s)
                stats["wire"] += length
            for run, s in stacks:           # a via stack is the same node as the wire next to it
                s = s if s is not None else (kept[0] if kept else None)
                for n in run:
                    if s is not None:
                        at.setdefault(n, s)
            stats["vias"] += sum(1 for a, b in zip(path, path[1:]) if grid.layer(a) != grid.layer(b))
            stats["segments"] += len(kept)
            # the path's ends meet a landing stub or what the net had already routed
            if kept:
                segs.link(before.get(path[0]), kept[0])
                segs.link(before.get(path[-1]), kept[-1])
            else:
                segs.link(before.get(path[0]), before.get(path[-1]))
        stats["segments"] += len(rr.landings.get(net, []))
    for a, b in sorted(segs.links):
        g._add_edge("segment__connects__segment", a, b)

    # ---- coupling ------------------------------------------------------------------
    parallel: dict[tuple[int, int], float] = defaultdict(float)
    cross: dict[tuple[int, int], int] = defaultdict(int)
    for e, owners in segs.steps.items():
        for d in (-1, 1):
            for s in owners:
                for t in segs.steps.get(_beside(e, d), ()):
                    if segs.net[s] != segs.net[t]:
                        parallel[(s, t)] += tech.pitch
    for layers in segs.points.values():
        for s in layers.get(0, ()):
            for t in layers.get(1, ()):
                if segs.net[s] != segs.net[t]:
                    cross[(s, t)] += 1
                    cross[(t, s)] += 1
    net_par: dict[tuple[str, str], float] = defaultdict(float)
    net_cross: dict[tuple[str, str], int] = defaultdict(int)
    for s, t in sorted(set(parallel) | set(cross)):
        g._add_edge("segment__beside__segment", s, t, [parallel.get((s, t), 0.0), cross.get((s, t), 0)])
        net_par[(segs.net[s], segs.net[t])] += parallel.get((s, t), 0.0)
        net_cross[(segs.net[s], segs.net[t])] += cross.get((s, t), 0)
    for a, b in sorted(set(net_par) | set(net_cross)):
        g._add_edge("net__coupled__net", net_idx[a], net_idx[b], [net_par.get((a, b), 0.0), net_cross.get((a, b), 0)])

    # ---- net and graph labels ------------------------------------------------------
    failed = set(rr.failed_nets)
    unreachable: dict[str, int] = defaultdict(int)
    for owner in rr.unreachable:
        inst, bnet = owner.split(".", 1)
        unreachable[circuit[inst].nets[bnet]] += 1
    rows = []
    for net in g.nodes["net"]["names"]:
        st = per_net.get(net, {"wire": 0.0, "stub": 0.0, "vias": 0, "segments": 0})
        lands = len(rr.landings.get(net, []))
        bad = net in failed or unreachable[net] > 0
        rows.append([float(lands >= 2 and net in rr.paths and not bad), float(bad),
                     unreachable[net], lands, st["wire"], st["stub"], st["wire"] + st["stub"],
                     st["vias"], st["segments"],
                     sum(v for (a, _), v in net_par.items() if a == net),
                     sum(v for (a, _), v in net_cross.items() if a == net)])
    g._set_labels("net", rows)
    g.graph_labels = {
        "routed": float(rr.success), "area_um2": round(m["area"], 6), "width_um": round(width, 6),
        "height_um": round(height, 6), "dead_space": round(m["dead_space"], 6), "hpwl_um": round(m["hpwl"], 6),
        "wirelength_um": round(rr.wirelength, 6), "vias": float(rr.vias), "failed_nets": float(len(rr.failed_nets)),
        "unreachable_pins": float(len(rr.unreachable)), "rounds": float(len(loop.rounds)) if loop is not None else 1.0,
    }
    return g

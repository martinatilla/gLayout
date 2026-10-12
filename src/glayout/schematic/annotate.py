"""Find functional structures, matched devices and layout blocks in a schematic.

Rule-based (survey item P6): every structure has two to four devices, so the
matcher works directly on pins and nets.

1. **Seeds** (matched sets):
   - ``dp``: two devices of one type on one non-supply source net, distinct
     gates and drains, neither diode-connected. Also the degenerated form,
     whose sources meet through two passives of one kind on a common net.
   - ``ccp``: cross-coupled pair (each gate on the other's drain, one source).
   - ``mirror``: devices of one type sharing gate, source and body, with
     evidence that the gate is a mirror's (see ``_mirror_groups``).
   - ``gate_diode``: a diode-connected device and another of its type on its
     gate net with a different source (translinear and class-AB biasing).
2. **Propagation** from every matched *pair* until nothing changes:
   ``cascode`` (one device on each drain, or under each source, sharing a
   gate), ``load`` (one device on each drain, by its drain, sharing a gate),
   ``passive`` (two passives of one kind from each side to one common net).
   A pair is symmetric when it descends from a ``dp`` or ``ccp``.
3. **Blocks** (one polarity and one body each, so ``finger_sequence_cell`` can
   draw them in one row): differential pairs, then mirror groups, then the
   remaining propagated pairs, then two-device stacks of leftovers (``CASC``),
   then single devices, classified as ``BIAS``, ``CS``, ``SF``, ``CASC``,
   ``INV``, ``TG``, ``RES``, ``CAP`` or ``OTHER``. Kinds are the floorplanner's
   ``KINDS``.
4. **Symmetry for the floorplanner**: a block holding both halves of a
   symmetric pair is self-symmetric; two single-device blocks holding the two
   halves form a symmetric block pair.

The matched sets are what a designer would declare; the PR #127 benchmark's
30 declared pairs are all recovered (see ``tests/test_annotate.py``).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from glayout.schematic.model import Device, Schematic, net_class

PASSIVES = ("res", "cap", "ind")
_SUPPLY = ("power", "ground")


@dataclass(frozen=True)
class Structure:
    kind: str                     # dp, degenerated_dp, ccp, mirror, inverter, tg, stack
    devices: tuple[str, ...]


@dataclass
class Matched:
    devices: tuple[str, ...]
    reasons: list[str]
    symmetric: bool = False

    def pairs(self) -> list[frozenset]:
        d = self.devices
        return [frozenset((d[i], d[j])) for i in range(len(d)) for j in range(i + 1, len(d))]


@dataclass
class AnnotatedBlock:
    name: str
    kind: str                     # one of floorplan.circuit.KINDS
    devices: tuple[str, ...]
    device_kind: str              # nfet / pfet / res / ... (one per block)
    body: Optional[str] = None


@dataclass
class Annotation:
    schematic: str
    blocks: list[AnnotatedBlock]
    structures: list[Structure]
    matched: list[Matched]
    symmetric_pairs: list[tuple[str, str]] = field(default_factory=list)   # block names
    self_symmetric: list[str] = field(default_factory=list)                # block names

    def block_of(self, device: str) -> AnnotatedBlock:
        for b in self.blocks:
            if device in b.devices:
                return b
        raise KeyError(device)

    def matched_pairs(self, symmetric_only: bool = False) -> set[frozenset]:
        out = set()
        for m in self.matched:
            if m.symmetric or not symmetric_only:
                out.update(m.pairs())
        return out

    def to_json(self) -> dict:
        return {
            "schematic": self.schematic,
            "blocks": [{"name": b.name, "kind": b.kind, "devices": list(b.devices), "device_kind": b.device_kind,
                        "body": b.body} for b in self.blocks],
            "structures": [{"kind": s.kind, "devices": list(s.devices)} for s in self.structures],
            "matched": [{"devices": list(m.devices), "reasons": m.reasons, "symmetric": m.symmetric}
                        for m in self.matched],
            "symmetric_pairs": [list(p) for p in self.symmetric_pairs],
            "self_symmetric": list(self.self_symmetric),
        }

    def summary(self) -> str:
        lines = [f"{self.schematic}: {len(self.blocks)} blocks"]
        for b in self.blocks:
            tag = " self-symmetric" if b.name in self.self_symmetric else ""
            lines.append(f"  {b.name:<7} {b.kind:<5} {b.device_kind:<5} {' '.join(b.devices)}{tag}")
        for a, b in self.symmetric_pairs:
            lines.append(f"  symmetric pair {a} / {b}")
        for m in self.matched:
            lines.append(f"  matched {' '.join(m.devices)} ({', '.join(m.reasons)}){' sym' if m.symmetric else ''}")
        return "\n".join(lines)


class _Index:
    """Pin lookups over one schematic."""

    def __init__(self, sch: Schematic):
        self.sch = sch
        self.dev = {d.name: d for d in sch.devices}
        self.order = {d.name: i for i, d in enumerate(sch.devices)}
        self.on = defaultdict(list)            # (pin, net) -> [device]
        self.pins_on = defaultdict(list)       # net -> [(device, pin)]
        for d in sch.devices:
            for pin, net in d.pins.items():
                self.on[(pin, net)].append(d)
                self.pins_on[net].append((d.name, pin))

    def mos(self) -> list[Device]:
        return [d for d in self.sch.devices if d.is_mos]

    @staticmethod
    def diode(d: Device) -> bool:
        return d.is_mos and d.net("D") == d.net("G")

    @staticmethod
    def supply(net: str) -> bool:
        return net_class(net) in _SUPPLY

    def passives_between(self, net: str) -> list[tuple[Device, str]]:
        """(passive, other terminal net) for every passive with one terminal on ``net``."""
        out = []
        for d in self.sch.devices:
            if d.kind in PASSIVES and net in d.nets and d.nets[0] != d.nets[1]:
                out.append((d, d.nets[1] if d.nets[0] == net else d.nets[0]))
        return out


def _sorted(idx: _Index, names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(names), key=idx.order.__getitem__))


# ---- 1. seeds ----------------------------------------------------------------
def _diff_pairs(idx: _Index) -> list[tuple[Structure, list[tuple[Device, Device]]]]:
    """Differential and cross-coupled pairs; with each, the passive pairs it implies (degeneration)."""
    out = []
    used: set[str] = set()
    by_source = defaultdict(list)
    for d in idx.mos():
        by_source[(d.kind, d.net("B"), d.net("S"))].append(d)
    for (kind, body, src), devs in by_source.items():
        if idx.supply(src):
            cands = [d for d in devs if not idx.diode(d)]
            for i, a in enumerate(cands):
                for b in cands[i + 1:]:
                    if a.net("G") == b.net("D") and b.net("G") == a.net("D") and a.name not in used and b.name not in used:
                        out.append((Structure("ccp", _sorted(idx, (a.name, b.name))), []))
                        used.update((a.name, b.name))
            continue
        cands = [d for d in devs if not idx.diode(d)]
        if len(cands) == 2:
            a, b = cands
            if a.net("G") != b.net("G") and a.net("D") != b.net("D"):
                kind_ = "ccp" if (a.net("G") == b.net("D") and b.net("G") == a.net("D")) else "dp"
                out.append((Structure(kind_, _sorted(idx, (a.name, b.name))), []))
                used.update((a.name, b.name))
    # degenerated: sources apart, each through one passive to a common net
    mos = [d for d in idx.mos() if d.name not in used and not idx.diode(d) and not idx.supply(d.net("S"))]
    for i, a in enumerate(mos):
        for b in mos[i + 1:]:
            if (a.kind, a.net("B")) != (b.kind, b.net("B")) or a.name in used or b.name in used:
                continue
            if a.net("S") == b.net("S") or a.net("G") == b.net("G") or a.net("D") == b.net("D"):
                continue
            pa = idx.passives_between(a.net("S"))
            pb = idx.passives_between(b.net("S"))
            hits = [(p, q) for p, n in pa for q, m in pb if n == m and p.kind == q.kind and p is not q]
            if len(hits) == 1:
                out.append((Structure("degenerated_dp", _sorted(idx, (a.name, b.name))), hits))
                used.update((a.name, b.name))
    return out


def _mirror_groups(idx: _Index) -> list[tuple[str, ...]]:
    """Devices sharing gate, source and body, when something says the gate is a mirror's.

    Evidence: a diode-connected member; a gate on a bias net or a port; or a
    cascode diode (a device of the same type from a member's drain up to the
    gate net, as in a wide-swing or low-voltage cascode mirror). Two devices
    that merely share an input signal (e.g. both driven by a tank node) are
    not a mirror.
    """
    groups = defaultdict(list)
    for d in idx.mos():
        groups[(d.kind, d.net("G"), d.net("S"), d.net("B"))].append(d)
    out = []
    for (kind, gate, _, _), g in groups.items():
        if len(g) < 2:
            continue
        drains = {d.net("D") for d in g}
        evidence = (any(idx.diode(d) for d in g) or net_class(gate) == "bias" or gate in idx.sch.ports
                    or any(c.kind == kind and c.net("S") in drains for c in idx.on[("D", gate)]))
        if evidence:
            out.append(_sorted(idx, (d.name for d in g)))
    return out


def _gate_diode_pairs(idx: _Index) -> list[tuple[str, str]]:
    out = []
    for d in idx.mos():
        if not idx.diode(d):
            continue
        for other in idx.on[("G", d.net("G"))]:
            if other is not d and other.kind == d.kind and other.net("S") != d.net("S"):
                out.append(_sorted(idx, (d.name, other.name)))
    return out


# ---- 2. propagation ---------------------------------------------------------
def _unique_pair(cands_a: list[Device], cands_b: list[Device], exclude: set[str]) -> Optional[tuple[Device, Device]]:
    hits = [(c, d) for c in cands_a for d in cands_b
            if c is not d and c.name not in exclude and d.name not in exclude
            and c.kind == d.kind and c.is_mos and c.net("G") == d.net("G")]
    return hits[0] if len(hits) == 1 else None


def _propagate(idx: _Index, matched: dict[frozenset, Matched]) -> None:
    changed = True
    while changed:
        changed = False
        for key, m in list(matched.items()):
            if len(m.devices) != 2:
                continue
            a, b = (idx.dev[n] for n in m.devices)
            if not (a.is_mos and b.is_mos):
                continue
            found: list[tuple[tuple[str, ...], str]] = []
            pair = set(m.devices)
            if a.net("D") != b.net("D"):
                up = _unique_pair(idx.on[("S", a.net("D"))], idx.on[("S", b.net("D"))], pair)
                if up:
                    found.append(((up[0].name, up[1].name), "cascode"))
                load = _unique_pair(idx.on[("D", a.net("D"))], idx.on[("D", b.net("D"))], pair)
                if load:
                    found.append(((load[0].name, load[1].name), "load"))
            if a.net("S") != b.net("S"):
                down = _unique_pair(idx.on[("D", a.net("S"))], idx.on[("D", b.net("S"))], pair)
                if down:
                    found.append(((down[0].name, down[1].name), "cascode"))
            for pin in ("D", "S"):
                if a.net(pin) == b.net(pin):
                    continue
                pa, pb = idx.passives_between(a.net(pin)), idx.passives_between(b.net(pin))
                hits = [(p, q) for p, n in pa for q, k in pb if n == k and p.kind == q.kind and p is not q]
                if len(hits) == 1:
                    found.append(((hits[0][0].name, hits[0][1].name), "passive"))
            for names, reason in found:
                k = frozenset(names)
                if k not in matched:
                    matched[k] = Matched(_sorted(idx, names), [reason], m.symmetric)
                    changed = True
                else:
                    t = matched[k]
                    if reason not in t.reasons:
                        t.reasons.append(reason)
                    if m.symmetric and not t.symmetric:
                        t.symmetric = True
                        changed = True


# ---- 3. blocks ---------------------------------------------------------------
def _single_kind(idx: _Index, d: Device, structures: list[Structure]) -> str:
    for s in structures:
        if d.name in s.devices and s.kind in ("inverter", "tg"):
            return "INV" if s.kind == "inverter" else "TG"
    if d.kind == "res":
        return "RES"
    if d.kind == "cap":
        return "CAP"
    if not d.is_mos:
        return "OTHER"
    if idx.diode(d):
        return "BIAS"
    if idx.supply(d.net("S")):
        return "BIAS" if net_class(d.net("G")) == "bias" or any(
            idx.diode(o) for o in idx.on[("G", d.net("G"))] if o.kind == d.kind) else "CS"
    if idx.supply(d.net("D")):
        return "SF"
    stacked = any(o.kind == d.kind for o in idx.on[("D", d.net("S"))]) or \
        any(o.kind == d.kind for o in idx.on[("S", d.net("D"))])
    return "CASC" if stacked else "OTHER"


def _inverters_and_tgs(idx: _Index) -> list[Structure]:
    out = []
    n = [d for d in idx.mos() if d.kind == "nfet"]
    p = [d for d in idx.mos() if d.kind == "pfet"]
    for a in n:
        for b in p:
            if a.net("G") == b.net("G") and a.net("D") == b.net("D") and net_class(a.net("S")) == "ground" \
                    and net_class(b.net("S")) == "power":
                out.append(Structure("inverter", _sorted(idx, (a.name, b.name))))
            elif {a.net("D"), a.net("S")} == {b.net("D"), b.net("S")} and a.net("G") != b.net("G") \
                    and a.net("D") != a.net("S"):
                out.append(Structure("tg", _sorted(idx, (a.name, b.name))))
    return out


def annotate(sch: Schematic) -> Annotation:
    idx = _Index(sch)
    structures: list[Structure] = []
    matched: dict[frozenset, Matched] = {}

    def add(names, reason, symmetric=False):
        k = frozenset(names)
        if k in matched:
            if reason not in matched[k].reasons:
                matched[k].reasons.append(reason)
            matched[k].symmetric |= symmetric
        else:
            matched[k] = Matched(_sorted(idx, names), [reason], symmetric)

    dps = _diff_pairs(idx)
    for s, passive_pairs in dps:
        structures.append(s)
        add(s.devices, "ccp" if s.kind == "ccp" else "dp", True)
        for p, q in passive_pairs:
            add((p.name, q.name), "passive", True)
    mirrors = _mirror_groups(idx)
    for g in mirrors:
        structures.append(Structure("mirror", g))
        add(g, "mirror")
    for pair in _gate_diode_pairs(idx):
        add(pair, "gate_diode")
    _propagate(idx, matched)
    structures += _inverters_and_tgs(idx)

    # blocks
    blocks: list[AnnotatedBlock] = []
    used: set[str] = set()
    counters: dict[str, int] = defaultdict(int)

    def block(kind: str, names: Iterable[str]):
        names = _sorted(idx, names)
        first = idx.dev[names[0]]
        counters[kind] += 1
        body = first.net("B") if first.is_mos else None
        blocks.append(AnnotatedBlock(f"{kind}{counters[kind]}", kind, names, first.kind, body))
        used.update(names)

    for s, _ in dps:
        block("DP", s.devices)
    for g in mirrors:
        free = [n for n in g if n not in used]
        if len(free) >= 3:
            block("BIAS", free)
        elif len(free) == 2:
            block("CM" if any(idx.diode(idx.dev[n]) for n in free) else "CMB", free)
    # remaining matched pairs, strongest first: symmetric, then structural, then gate_diode
    for m in sorted(matched.values(), key=lambda m: (not m.symmetric, m.reasons == ["gate_diode"])):
        names = [n for n in m.devices if n not in used]
        if len(m.devices) != 2 or len(names) != 2:
            continue
        a, b = (idx.dev[n] for n in names)
        if a.is_mos and b.is_mos and (a.kind, a.net("B")) == (b.kind, b.net("B")):
            kind = "CM" if idx.diode(a) or idx.diode(b) else "CMB" if a.net("S") == b.net("S") else "CASC"
            block(kind, names)
    # two-device stacks of leftovers: the node between them is internal and has no other pin
    for d in idx.mos():
        if d.name in used or idx.supply(d.net("S")) or d.net("S") in sch.ports:
            continue
        below = [o for o in idx.on[("D", d.net("S"))]
                 if o.name not in used and o is not d and (o.kind, o.net("B")) == (d.kind, d.net("B"))]
        if len(below) == 1 and len(idx.pins_on[d.net("S")]) == 2:
            structures.append(Structure("stack", _sorted(idx, (d.name, below[0].name))))
            block("CASC", (d.name, below[0].name))
    for d in sch.devices:
        if d.name not in used:
            block(_single_kind(idx, d, structures), (d.name,))

    ann = Annotation(sch.name, blocks, structures, list(matched.values()))
    _floorplan_symmetry(ann)
    return ann


def _floorplan_symmetry(ann: Annotation) -> None:
    taken: set[str] = set()
    for m in ann.matched:
        if not m.symmetric or len(m.devices) != 2:
            continue
        ba, bb = (ann.block_of(n) for n in m.devices)
        if ba is bb:
            if ba.name not in ann.self_symmetric and ba.name not in taken:
                ann.self_symmetric.append(ba.name)
        elif len(ba.devices) == 1 and len(bb.devices) == 1 and ba.device_kind == bb.device_kind \
                and ba.name not in taken and bb.name not in taken and ba.name not in ann.self_symmetric \
                and bb.name not in ann.self_symmetric:
            ann.symmetric_pairs.append((ba.name, bb.name))
            taken.update((ba.name, bb.name))

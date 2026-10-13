"""Schematic → floorplanner ``Circuit``: one finger-sequence block per annotated block.

``schematic_to_circuit(pdk, sch)`` annotates and sizes the schematic (unless
given an annotation and a sizing), builds every block's variants with
``finger_sequence_cell`` and returns a ``Circuit`` ready for
``anneal_floorplan`` / ``closed_loop_floorplan``. ``schematic_routed`` runs
the whole flow to a routed layout with its LVS netlist.

Blocks are built on canonical names: devices ``A``, ``B``, … and nets ``n0``,
``n1``, … in order of first use, plus the bulk ``B``. The instance maps them
back to circuit nets. The same mirror in two circuits is then the same cell,
built once per process (``BLOCK_CACHE``). Every instance still gets its own
``Block`` and ``Variant`` objects, because DRR padding and the closed loop's
per-block scales are per instance.

A block exports a net only if something outside it uses that net (another
device, any device's bulk, or a port); nets that stay inside the block are
its own rails and get no pin. The bulk is always exported.

Supported for now: MOS-only schematics whose nfet bodies sit on ground
(a single p-substrate; an nfet body elsewhere needs an isolated p-well) and
with no transistor whose source is its drain (a MOS capacitor or varactor,
which a finger row cannot draw). pfet bodies may be on any net: every pfet
block has its own n-well and tie ring. Anything else raises
``UnsupportedCircuit`` with a ``code``.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from typing import Optional, Union

from glayout.cells.elementary.finger_sequence import finger_sequence_cell
from glayout.floorplan.block import Block, Variant, extract_pins
from glayout.floorplan.circuit import Circuit, Instance
from glayout.pdk.mappedpdk import MappedPDK
from glayout.schematic.annotate import AnnotatedBlock, Annotation, annotate
from glayout.schematic.model import Schematic, net_class
from glayout.schematic.sizing import BlockSize, Sizing, size_schematic
from glayout.util.process_table import ProcessTable

BLOCK_CACHE: dict[str, list[tuple[dict, float, float, dict, object]]] = {}


class UnsupportedCircuit(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def pdk_limits(pdk: MappedPDK) -> dict:
    """Numbers the sizing needs: minimum length and the narrowest finger it may draw, µm."""
    return {"lmin": float(pdk.get_grule("poly")["min_width"]),
            "wmin": max(0.5, float(pdk.get_grule("active_diff")["min_width"]))}


def check_supported(sch: Schematic) -> None:
    other = sorted({d.kind for d in sch.devices if not d.is_mos})
    if other:
        raise UnsupportedCircuit("non_mos", f"{sch.name} has {', '.join(other)} devices (MOS only for now)")
    bad = [d.name for d in sch.devices if d.kind == "nfet" and net_class(d.net("B")) != "ground"]
    if bad:
        raise UnsupportedCircuit("nfet_body", f"{sch.name}: nfet body off ground for {', '.join(bad)}")
    caps = [d.name for d in sch.devices if d.is_mos and d.net("D") == d.net("S")]
    if caps:
        raise UnsupportedCircuit("mos_cap", f"{sch.name}: source = drain (MOS capacitor or varactor) for "
                                            f"{', '.join(caps)}; a finger row cannot draw it")


@contextmanager
def _pin_labels(on: bool):
    old = os.environ.get("GLAYOUT_NO_PIN_LABELS")
    if on:
        os.environ.pop("GLAYOUT_NO_PIN_LABELS", None)
    else:
        os.environ["GLAYOUT_NO_PIN_LABELS"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GLAYOUT_NO_PIN_LABELS", None)
        else:
            os.environ["GLAYOUT_NO_PIN_LABELS"] = old


def _letters(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def canonical_block(sch: Schematic, blk: AnnotatedBlock, external: set[str]) -> tuple[list[dict], dict[str, str], list[str]]:
    """(devices with canonical nets, exported canonical pin -> circuit net, exported pins incl. ``B``).

    Nets that stay inside the block are in ``devices`` only: they have no pin and no circuit net.
    """
    cnet: dict[str, str] = {}
    devices = []
    for i, name in enumerate(blk.devices):
        d = sch[name]
        spec = {"name": _letters(i)}
        for pin, key in (("D", "drain"), ("G", "gate"), ("S", "source")):
            net = d.net(pin)
            if net not in cnet:
                cnet[net] = f"n{len(cnet)}"
            spec[key] = cnet[net]
        devices.append(spec)
    pins = [c for n, c in cnet.items() if n in external] + ["B"]
    net_map = {c: n for n, c in cnet.items() if c in pins}
    net_map["B"] = blk.body
    return devices, net_map, pins


def _external_nets(sch: Schematic, blk: AnnotatedBlock) -> set[str]:
    inside = set(blk.devices)
    ext = set(sch.ports)
    for d in sch.devices:
        if d.name in inside:
            ext.add(d.net("B"))
            continue
        ext.update(d.nets)
    return ext


def build_block_variants(pdk: MappedPDK, device_kind: str, devices: list[dict], base: list[int], pins: list[str],
                         size: BlockSize, labels: bool = False,
                         cache: Optional[dict] = None) -> list[tuple[dict, float, float, dict, object]]:
    """Build (or fetch) every variant: (params, width, height, pins, component).

    ``devices`` are canonical (``canonical_block``); ``base`` their base finger counts.
    """
    cache = BLOCK_CACHE if cache is None else cache
    key_body = {"pdk": pdk.name, "kind": device_kind, "devices": [[d["drain"], d["gate"], d["source"]] for d in devices],
                "base": base, "unit": size.unit, "length": size.length, "splits": list(size.splits),
                "pins": pins, "labels": labels}
    key = hashlib.sha256(json.dumps(key_body, sort_keys=True).encode()).hexdigest()[:12]
    if key in cache:
        return cache[key]
    built = []
    with _pin_labels(labels):
        for k, width in size.variants():
            params = {
                "devices": [{**d, "fingers": b * k} for d, b in zip(devices, base)],
                "device": device_kind, "width": float(pdk.snap_to_2xgrid(width)), "length": size.length,
                "pins": list(pins), "circuit_name": f"FS_{key}_K{k}".upper(), "name": f"fs_{key}_k{k}",
            }
            comp = finger_sequence_cell(pdk, **params)
            built.append((params, float(comp.xsize), float(comp.ysize),
                          extract_pins(comp, comp.info["pins"], pdk), comp))
    cache[key] = built
    return built


def schematic_to_circuit(pdk: MappedPDK, sch: Schematic, ann: Optional[Annotation] = None,
                         sizing: Optional[Sizing] = None, sizing_seed: int = 0, labels: bool = False,
                         cache: Optional[dict] = None) -> Circuit:
    """Floorplanner circuit for ``sch``. ``labels=False`` builds cells for a routed composite."""
    check_supported(sch)
    ann = ann or annotate(sch)
    if sizing is None:
        lim = pdk_limits(pdk)
        sizing = size_schematic(sch, ann, lim["lmin"], lim["wmin"], seed=sizing_seed)
    instances = []
    with ProcessTable(f"Schematic → circuit {sch.name}", f"{len(ann.blocks)} blocks, sizing seed {sizing.seed}") as pt:
        for blk in ann.blocks:
            with pt.stage(blk.name) as st:
                size = sizing.blocks[blk.name]
                devices, net_map, pins = canonical_block(sch, blk, _external_nets(sch, blk))
                n_cached = len(BLOCK_CACHE if cache is None else cache)
                base = [size.base[n] for n in blk.devices]
                built = build_block_variants(pdk, blk.device_kind, devices, base, pins, size, labels, cache)
                variants = [Variant(dict(p), w, h, pin_map, comp) for p, w, h, pin_map, comp in built]
                instances.append(Instance(blk.name, Block(blk.name, blk.kind, variants), net_map))
                hit = len(BLOCK_CACHE if cache is None else cache) == n_cached
                st.detail(f"{blk.kind} {' '.join(blk.devices)}: {len(variants)} variant(s), "
                          f"pins {','.join(net_map[p] for p in pins)}{' (cached)' if hit else ''}")
        circuit = Circuit(sch.name, instances, symmetric_pairs=list(ann.symmetric_pairs),
                          self_symmetric=list(ann.self_symmetric), ports=list(sch.ports),
                          schematic=sch, annotation=ann, sizing=sizing)
        pt.result("instances", len(instances))
    return circuit


def schematic_routed(
    pdk: MappedPDK,
    circuit: Union[str, Schematic] = "C08",
    sizing_seed: int = 0,
    seed: int = 0,
    mode: str = "closed_loop",
    drr_scale: float = 1.0,
    strict: bool = True,
    return_result: bool = False,
    name: Optional[str] = None,
):
    """Annotate, size, build, floorplan and route a schematic. Builder-compatible with the DRC/LVS runner.

    ``circuit``: a curated circuit id (``C01``…``C18``, ``two_stage_ota``) or a ``Schematic``.
    ``mode``: ``closed_loop`` (router feedback on the padding) or ``drr`` (fixed padding).
    """
    from glayout.floorplan.anneal import anneal_floorplan
    from glayout.floorplan.drr import apply_drr
    from glayout.floorplan.route_flow import closed_loop_floorplan, route_floorplan
    from glayout.floorplan.router import RouterTech
    from glayout.schematic.textbook import curated
    sch = curated()[circuit] if isinstance(circuit, str) else circuit
    fp_circuit = schematic_to_circuit(pdk, sch, sizing_seed=sizing_seed, labels=False)
    blocks = [inst.block for inst in fp_circuit.instances]
    tech = RouterTech.from_pdk(pdk)
    if mode == "closed_loop":
        loop = closed_loop_floorplan(pdk, fp_circuit, blocks, tech, seed=seed)
        fp = loop.floorplan
    elif mode == "drr":
        apply_drr(pdk, blocks, scale=drr_scale, layers=tech.layers)
        fp = anneal_floorplan(fp_circuit, seed=seed)
        loop = None
    else:
        raise ValueError(f"unknown mode {mode!r}")
    comp, rr = route_floorplan(pdk, fp_circuit, fp, tech, name=name or f"{sch.name.lower()}_routed")
    if strict and not rr.success:
        raise RuntimeError(f"routing failed: nets {rr.failed_nets}, unreachable pins {rr.unreachable}")
    return (comp, fp_circuit, fp, rr, loop) if return_result else comp

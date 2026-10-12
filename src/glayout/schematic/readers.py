"""Readers that turn external netlists into ``Schematic``s.

AnalogGenie (https://github.com/xz-group/AnalogGenie, MIT) stores each circuit
in ``Dataset/<id>/``:

- ``<id>.cir``: one device per line, ``M4 (net23 VB1 VSS VSS) nmos4``: name,
  nets in pin order, type. Topology only, no sizes.
- ``Port<id>.txt``: the external nets on the first line.
- ``Pagenumber<id>.txt``: book page or paper reference.
- ``<id>_full.cir`` (32 entries): the same netlist in Spectre form with
  ``subckt`` definitions of the composite cells it uses (inverter,
  transmission gate, NAND, NOR, XOR, PFD), so it flattens to transistors.
- ``Opamp<id>.cir`` (the bandgap and LDO entries): the error amplifier on its
  own; its ports are not listed anywhere, so they are inferred from net names.
- ``Book<id>.png``, ``Cadence<id>.png``: textbook scan and schematic capture;
  never read here.

Differences from AnalogGenie's own ``SPICE2GRAPH_full.py``, which they need
care with: BJT lines carry a fourth substrate net (it is kept, not dropped);
flattened subcircuits reuse instance names in one file (the second ``MM0``
becomes ``MM0_2`` and the rename is recorded); blank lines are skipped.

``from_textbook_circuit`` reads the 18 curated circuits of the textbook
sharing benchmark (gLayout PR #127, ``Circuit.to_json()``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Union

from glayout.schematic.model import PINS, Device, Schematic

AG_KINDS = {
    "nmos4": "nfet", "pmos4": "pfet", "npn": "npn", "pnp": "pnp",
    "resistor": "res", "capacitor": "cap", "inductor": "ind", "diode": "diode",
    "INVERTER": "composite:INVERTER", "TRANSMISSION_GATE": "composite:TRANSMISSION_GATE",
    "XOR": "composite:XOR", "PFD": "composite:PFD",
}

# Dataset/data_categorization.md: (first id, last id, category)
AG_CATEGORIES = [
    (1, 444, "book:razavi_analog_cmos"),
    (445, 603, "book:razavi_rf"),
    (604, 665, "book:camenzind"),
    (666, 821, "book:johns_martin"),
    (822, 932, "book:gray"),
    (933, 1044, "book:allen_holberg"),
    (1045, 1080, "paper:comparator"),
    (1081, 1090, "paper:lna"),
    (1091, 1099, "paper:mixer"),
    (1100, 1108, "paper:pa"),
    (1109, 1190, "paper:vco"),
    (1191, 1460, "paper:power_converter"),
    (1461, 1780, "paper:bandgap"),
    (1781, 2180, "paper:opamp"),
    (2181, 2630, "paper:ldo"),
    (2631, 3502, "paper:sc_sampler"),
]
_UNLISTED = range(1030, 1039)   # inside the Allen range but not listed in the categorisation file

_LINE = re.compile(r"^\s*(\S+)\s*\(([^()]*)\)\s*(\S+)\s*$")


class NetlistParseError(ValueError):
    pass


def analoggenie_category(ag_id: int) -> Optional[str]:
    if ag_id in _UNLISTED:
        return None
    for lo, hi, cat in AG_CATEGORIES:
        if lo <= ag_id <= hi:
            return cat
    return None


def parse_ports(text: str) -> tuple[str, ...]:
    """First line that is not blank and not a ``---`` rule."""
    for line in text.splitlines():
        s = line.strip()
        if s and set(s) != {"-"}:
            return tuple(s.split())
    return ()


_SUBCKT = re.compile(r"^\s*subckt\s+(\S+)\s+(.*?)\s*$")
_ENDS = re.compile(r"^\s*ends\b")
_GLOBAL_NETS = {"0"}
_COMMENT = ("*", "//", ";")


def parse_subckts(text: str) -> tuple[dict[str, tuple[tuple[str, ...], list[tuple[int, str]]]], list[tuple[int, str]]]:
    """Spectre ``subckt NAME pins ... ends NAME`` blocks, and the (line number, text) lines outside them."""
    subckts: dict[str, tuple[tuple[str, ...], list[tuple[int, str]]]] = {}
    top: list[tuple[int, str]] = []
    current = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        m = _SUBCKT.match(raw)
        if m:
            current = m.group(1)
            subckts[current] = (tuple(m.group(2).split()), [])
        elif current is not None and _ENDS.match(raw):
            current = None
        elif current is not None:
            subckts[current][1].append((lineno, raw))
        else:
            top.append((lineno, raw))
    return subckts, top


def _expand(lines, subckts, prefix: str, pinmap: dict, depth: int, out: list, problem, flattened: list) -> None:
    """Device lines with subcircuit instances replaced by their (renamed, re-netted) contents."""
    for lineno, raw in lines:
        line = raw.strip()
        if not line or line.startswith(_COMMENT):
            continue
        m = _LINE.match(line)
        if not m:
            out.append((lineno, None, line, None))
            continue
        dname, nets, dtype = m.group(1), m.group(2).split(), m.group(3)
        nets = tuple(pinmap[n] if n in pinmap else n if (n in _GLOBAL_NETS or not prefix) else f"{prefix}{n}"
                     for n in nets)
        if dtype in subckts:
            pins, body = subckts[dtype]
            if len(pins) != len(nets) or depth > 8:
                problem(f"line {lineno}: cannot expand {dname} ({dtype}: {len(pins)} pins, {len(nets)} nets)")
                continue
            flattened.append(f"{prefix}{dname}:{dtype}")
            _expand(body, subckts, f"{prefix}{dname}/", dict(zip(pins, nets)), depth + 1, out, problem, flattened)
        else:
            out.append((lineno, f"{prefix}{dname}", nets, dtype))


def parse_analoggenie(text: str, name: str, ports: Sequence[str] = (), strict: bool = True,
                      provenance: Optional[dict] = None, flatten: bool = True) -> Schematic:
    """Parse the body of an AnalogGenie ``.cir`` (or ``_full.cir``) file.

    ``strict``: a line that does not parse, an unknown device type or a pin count
    that does not fit raises ``NetlistParseError``; otherwise it becomes a warning
    (unknown types are kept as ``unknown:<type>``, unparsable lines are skipped).
    ``flatten``: instances of ``subckt`` definitions found in the text are replaced
    by their devices (``I5/M0``, local nets ``I5/net8``); otherwise composites stay whole.
    """
    devices: list[Device] = []
    warnings: list[str] = []
    renamed: list[dict] = []
    seen: dict[str, int] = {}

    def problem(msg: str):
        if strict:
            raise NetlistParseError(f"{name}: {msg}")
        warnings.append(msg)

    subckts, top = parse_subckts(text)
    flattened: list[str] = []
    items: list = []
    _expand(top, subckts if flatten else {}, "", {}, 0, items, problem, flattened)
    for lineno, dname, nets, dtype in items:
        if dname is None:
            problem(f"line {lineno}: cannot parse {nets!r}")
            continue
        kind = AG_KINDS.get(dtype)
        if kind is None:
            problem(f"line {lineno}: unknown device type {dtype!r} ({dname})")
            kind = f"unknown:{dtype}"
        seen[dname] = seen.get(dname, 0) + 1
        if seen[dname] > 1:
            new = f"{dname}_{seen[dname]}"
            while new in seen:
                seen[dname] += 1
                new = f"{dname}_{seen[dname]}"
            seen[new] = 1
            renamed.append({"line": lineno, "from": dname, "to": new})
            dname = new
        try:
            devices.append(Device(dname, kind, nets))
        except ValueError as e:
            problem(f"line {lineno}: {e}")
    if renamed:
        warnings.append(f"{len(renamed)} repeated instance name(s) renamed")
    prov = dict(provenance or {})
    if renamed:
        prov["renamed"] = renamed
    if flattened:
        prov["flattened"] = flattened
    return Schematic(name, tuple(ports), tuple(devices), prov, {}, warnings)


def read_analoggenie(path: Union[str, Path], ag_id: Optional[int] = None, strict: bool = True,
                     flatten: bool = True) -> Schematic:
    """Read one AnalogGenie entry.

    ``path`` is the entry folder (``.../Dataset/157``), or the dataset root
    (``.../Dataset``) together with ``ag_id``. ``flatten``: read ``<id>_full.cir``
    when it exists, so composite cells become transistors.
    """
    folder = Path(path) / str(ag_id) if ag_id is not None else Path(path)
    ag = int(folder.name)
    cir = folder / f"{ag}.cir"
    if flatten and (folder / f"{ag}_full.cir").exists():
        cir = folder / f"{ag}_full.cir"
    if not cir.exists():
        raise FileNotFoundError(cir)
    port_file = folder / f"Port{ag}.txt"
    page_file = folder / f"Pagenumber{ag}.txt"
    ports = parse_ports(port_file.read_text()) if port_file.exists() else ()
    prov = {"dataset": "AnalogGenie", "analoggenie_id": ag, "category": analoggenie_category(ag),
            "page_ref": page_file.read_text().strip() if page_file.exists() else None, "file": cir.name}
    sch = parse_analoggenie(cir.read_text(errors="replace"), f"ag{ag}", ports, strict=strict, provenance=prov,
                            flatten=flatten)
    if not port_file.exists():
        sch.warnings.append(f"no Port{ag}.txt")
    return sch


_INTERNAL_NET = re.compile(r"^net\d+$", re.I)


def infer_ports(sch: Schematic) -> tuple[str, ...]:
    """AnalogGenie names internal nets ``net<N>``; every other net is taken as a port."""
    return tuple(n for n in sch.nets() if not _INTERNAL_NET.match(n.split("/")[-1]) and "/" not in n)


def read_analoggenie_opamp(path: Union[str, Path], ag_id: Optional[int] = None, strict: bool = True) -> Schematic:
    """The ``Opamp<id>.cir`` extract of a bandgap or LDO entry, with inferred ports."""
    folder = Path(path) / str(ag_id) if ag_id is not None else Path(path)
    ag = int(folder.name)
    cir = folder / f"Opamp{ag}.cir"
    if not cir.exists():
        raise FileNotFoundError(cir)
    prov = {"dataset": "AnalogGenie", "analoggenie_id": ag, "category": analoggenie_category(ag),
            "variant": "opamp_extract", "file": cir.name, "ports_inferred": True}
    sch = parse_analoggenie(cir.read_text(errors="replace"), f"ag{ag}_opamp", (), strict=strict, provenance=prov)
    sch.ports = infer_ports(sch)
    return sch


_PR127_KINDS = {"nfet": "nfet", "pfet": "pfet", "res": "res", "cap": "cap"}


def from_textbook_circuit(d: dict, matched: Optional[Sequence[Sequence[str]]] = None) -> Schematic:
    """A curated circuit of the textbook sharing benchmark (PR #127 ``Circuit.to_json()``).

    ``matched``: that benchmark's declared matched sets for this circuit
    (``matching.MATCHED[circuit_id]``), stored as a declared constraint.
    """
    devices = tuple(Device(x["name"], _PR127_KINDS[x["kind"]], tuple(x["nets"]), dict(x.get("params", {})))
                    for x in d["devices"])
    prov = {"dataset": "textbook_sharing", "circuit_id": d["circuit_id"], "topology": d["name"]}
    src = d.get("source", {})
    for k in ("analoggenie_id", "book", "page_ref", "device_name_map", "source_verification_level", "modifications"):
        if k in src:
            prov[k] = src[k]
    constraints = {"matched": [list(m) for m in matched]} if matched else {}
    return Schematic(d["circuit_id"], tuple(d["ports"]), devices, prov, constraints, list(d.get("notes", [])))

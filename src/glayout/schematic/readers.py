"""Readers that turn external netlists into ``Schematic``s.

AnalogGenie (https://github.com/xz-group/AnalogGenie, MIT) stores each circuit
in ``Dataset/<id>/``:

- ``<id>.cir``: one device per line, ``M4 (net23 VB1 VSS VSS) nmos4``: name,
  nets in pin order, type. Topology only, no sizes.
- ``Port<id>.txt``: the external nets on the first line.
- ``Pagenumber<id>.txt``: book page or paper reference.
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


def parse_analoggenie(text: str, name: str, ports: Sequence[str] = (), strict: bool = True,
                      provenance: Optional[dict] = None) -> Schematic:
    """Parse the body of an AnalogGenie ``.cir`` file.

    ``strict``: a line that does not parse, an unknown device type or a pin count
    that does not fit raises ``NetlistParseError``; otherwise it becomes a warning
    (unknown types are kept as ``unknown:<type>``, unparsable lines are skipped).
    """
    devices: list[Device] = []
    warnings: list[str] = []
    renamed: list[dict] = []
    seen: dict[str, int] = {}

    def problem(msg: str):
        if strict:
            raise NetlistParseError(f"{name}: {msg}")
        warnings.append(msg)

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("*", "//", ";")):
            continue
        m = _LINE.match(line)
        if not m:
            problem(f"line {lineno}: cannot parse {line!r}")
            continue
        dname, nets, dtype = m.group(1), tuple(m.group(2).split()), m.group(3)
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
    return Schematic(name, tuple(ports), tuple(devices), prov, {}, warnings)


def read_analoggenie(path: Union[str, Path], ag_id: Optional[int] = None, strict: bool = True) -> Schematic:
    """Read one AnalogGenie entry.

    ``path`` is the entry folder (``.../Dataset/157``), or the dataset root
    (``.../Dataset``) together with ``ag_id``.
    """
    folder = Path(path) / str(ag_id) if ag_id is not None else Path(path)
    ag = int(folder.name)
    cir = folder / f"{ag}.cir"
    if not cir.exists():
        raise FileNotFoundError(cir)
    port_file = folder / f"Port{ag}.txt"
    page_file = folder / f"Pagenumber{ag}.txt"
    ports = parse_ports(port_file.read_text()) if port_file.exists() else ()
    prov = {"dataset": "AnalogGenie", "analoggenie_id": ag, "category": analoggenie_category(ag),
            "page_ref": page_file.read_text().strip() if page_file.exists() else None}
    sch = parse_analoggenie(cir.read_text(errors="replace"), f"ag{ag}", ports, strict=strict, provenance=prov)
    if not port_file.exists():
        sch.warnings.append(f"no Port{ag}.txt")
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

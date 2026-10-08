"""SPICE text transforms that put the three netlists of a layout into one form.

The reference netlist (glayout), the LVS-extracted netlist and the parasitic
netlist (kpex) differ in device dialect, net naming, substrate handling and
pin order. These functions remove each difference; all are pure text so they
are tested without any tool.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Mapping, Optional, Sequence

from glayout.evaluation.config import SimConfig

_SUBCKT = re.compile(r"^\s*\.subckt\s+(\S+)(.*)$", re.IGNORECASE)
_NUMBER = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(meg|[afpnumkgt])?", re.IGNORECASE)
_SCALE = {"a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
          "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def logical_lines(text: str) -> list[str]:
    """Lines with ``+`` continuations joined onto the line they continue."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("+") and out:
            out[-1] += " " + stripped[1:].strip()
        else:
            out.append(line.rstrip())
    return out


def subckt_ports(text: str, name: Optional[str] = None) -> tuple[str, list[str]]:
    """Name and pins of ``.subckt name``; without ``name`` the last one (glayout's top)."""
    found = None
    for line in logical_lines(text):
        m = _SUBCKT.match(line)
        if m and (name is None or m.group(1).lower() == name.lower()):
            found = (m.group(1), [t for t in m.group(2).split() if "=" not in t and t.lower() != "params:"])
            if name is not None:
                break
    if found is None:
        raise ValueError(f"no .subckt {name} in netlist" if name else "no .subckt in netlist")
    return found


def to_sim_dialect(text: str, cfg: SimConfig) -> str:
    """glayout's ``X<name> d g s b <fet> l=.. w=..`` lines in the PDK's device dialect.

    The gf180 LVS runner (tests/lvs/klayout_gf180.py) applies the same two
    rewrites to the schematic: ``M`` for primitive FETs, and ``u`` on bare
    ``w``/``l``. Subcircuit instances (``X0 ... CC_DIFFPAIR``) are left alone.
    """
    if cfg.fet_prefix != "X":
        alt = "|".join(re.escape(f) for f in cfg.fets)
        text = re.sub(rf"^X(\S+)((?:\s+\S+){{4}}\s+(?:{alt})\b)", rf"{cfg.fet_prefix}\1\2", text,
                      flags=re.MULTILINE | re.IGNORECASE)
    if cfg.si_units:
        text = re.sub(r"(\b[wl]=)([0-9.]+)(?=\s|$)", r"\1\2u", text, flags=re.MULTILINE | re.IGNORECASE)
    return text


def sanitize_names(text: str) -> str:
    """KLayout escapes generated names (``\\$11``); ngspice does not unescape them.

    ``\\$11`` (unnamed net 11) becomes ``_11``, any other escaped character
    an underscore, so every node is a plain identifier.
    """
    return re.sub(r"\\(.)", "_", text.replace("\\$", "_"))


def rename_nodes(text: str, mapping: Mapping[str, str]) -> str:
    """Rename whole-token nodes on element lines; headers, comments and ``name=value`` stay.

    Used to put kpex's substrate node (``VSUBS``, not a subckt pin) on the
    cell's bulk net, so wire-to-substrate capacitance does not float.
    """
    upper = {k.upper(): v for k, v in mapping.items()}
    out = []
    for line in logical_lines(text):
        if not line.strip() or line.lstrip()[0] in "*.":
            out.append(line)
            continue
        first, *rest = line.split()
        out.append(" ".join([first] + [upper.get(t.upper(), t) if "=" not in t else t for t in rest]))
    return "\n".join(out) + "\n"


def wrapper(cell: str, cell_pins: Sequence[str], name: str, ports: Sequence[str],
            ties: Optional[Mapping[str, str]] = None) -> str:
    """``.subckt <name> <ports>`` around ``cell``, its pins matched by name, not position.

    Pin order differs between the netlists (the reference keeps the circuit's
    order, the extractor sorts), so the testbench only ever sees ``ports``. A
    cell pin that is not a port goes to ``ties`` (the substrate to ground) or
    else stays an internal node of the same name: a labelled net the
    testbench does not drive. A port the cell lacks is an error: the layout
    has no label for it.
    """
    ties = {k.upper(): v for k, v in (ties or {}).items()}
    by_upper = {p.upper(): p for p in ports}
    have = {p.upper() for p in cell_pins}
    missing = [p for p in ports if p.upper() not in have]
    if missing:
        raise ValueError(f"{cell} has no pin {', '.join(missing)} (pins: {' '.join(cell_pins)}); "
                         f"is the net labelled in the layout?")
    nodes = [by_upper.get(p.upper()) or ties.get(p.upper()) or p for p in cell_pins]
    return f".subckt {name} {' '.join(ports)}\nXcell {' '.join(nodes)} {cell}\n.ends {name}\n"


def spice_number(token: str) -> float:
    """``1.61812f`` -> 1.61812e-15; trailing unit letters (``pF``) are ignored."""
    m = _NUMBER.match(token)
    if not m:
        raise ValueError(f"not a SPICE number: {token!r}")
    return float(m.group(1)) * _SCALE.get((m.group(2) or "").lower(), 1.0)


def capacitors(text: str) -> list[tuple[str, str, float]]:
    """``(net a, net b, farads)`` for every C element."""
    caps = []
    for line in logical_lines(text):
        toks = line.split()
        if len(toks) >= 4 and toks[0][0] in "cC":
            caps.append((toks[1], toks[2], spice_number(toks[3])))
    return caps


def capacitance_by_net(caps: Sequence[tuple[str, str, float]]) -> dict[str, float]:
    """Total capacitance on each net (a coupling cap counts on both of its nets), largest first."""
    total: dict[str, float] = defaultdict(float)
    for a, b, c in caps:
        total[a] += c
        total[b] += c
    return dict(sorted(total.items(), key=lambda kv: -kv[1]))

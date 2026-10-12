"""Transistor-level schematic: devices, the nets their pins connect to, ports.

This is the netlist-in side of the floorplanning flow. A ``Schematic`` holds
no geometry and no sizing it does not know: AnalogGenie netlists carry
topology only, so ``Device.params`` stays empty until a sizing step fills it.

Pin order per kind follows the SPICE convention (MOS: D G S B; BJT: C B E,
optional substrate S; two-terminal devices: P N). Composite cells that
AnalogGenie draws as one symbol (inverter, transmission gate, XOR, PFD) are
kept whole as ``composite:<NAME>`` with their own pin names.

Pure Python: nothing here needs a PDK.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

PINS: dict[str, tuple[str, ...]] = {
    "nfet": ("D", "G", "S", "B"),
    "pfet": ("D", "G", "S", "B"),
    "npn": ("C", "B", "E", "S"),
    "pnp": ("C", "B", "E", "S"),
    "res": ("P", "N"),
    "cap": ("P", "N"),
    "ind": ("P", "N"),
    "diode": ("P", "N"),
    "composite:INVERTER": ("A", "Q", "VDD", "VSS"),
    "composite:TRANSMISSION_GATE": ("A", "B", "C", "VDD", "VSS"),
    "composite:XOR": ("A", "B", "VDD", "VSS", "Y"),
    "composite:PFD": ("A", "B", "QA", "QB", "VDD", "VSS"),
}
# kinds whose last pin may be left out (a BJT without its substrate terminal)
OPTIONAL_LAST_PIN = {"npn", "pnp"}
MOS = ("nfet", "pfet")

NET_CLASSES = ("power", "ground", "bias", "clock", "signal")
_NET_PATTERNS = [
    ("ground", re.compile(r"^(0|GND\w*|A?VSS\w*|VEE\w*)$", re.I)),
    ("power", re.compile(r"^(A?VDD\w*|VCC\w*)$", re.I)),
    ("bias", re.compile(r"^(VB\d*|IB\d*|VBIAS\w*|IBIAS\w*|IREF\w*|VREF\w*)$", re.I)),
    ("clock", re.compile(r"^(V?CLK\w*|PHI\w*)$", re.I)),
]


def net_class(net: str) -> str:
    """Class of a net from its name: power, ground, bias, clock or signal."""
    for cls, pattern in _NET_PATTERNS:
        if pattern.match(net):
            return cls
    return "signal"


@dataclass(frozen=True)
class Device:
    """One schematic element. ``nets`` follow ``PINS[kind]`` (or P1..Pn for an unknown kind)."""
    name: str
    kind: str
    nets: tuple[str, ...]
    params: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "nets", tuple(self.nets))
        expected = PINS.get(self.kind)
        if expected is None:
            return
        n = len(self.nets)
        if n != len(expected) and not (self.kind in OPTIONAL_LAST_PIN and n == len(expected) - 1):
            raise ValueError(f"device {self.name}: kind {self.kind} has pins {expected}, got {n} nets {self.nets}")

    @property
    def pin_names(self) -> tuple[str, ...]:
        names = PINS.get(self.kind)
        if names is None:
            return tuple(f"P{i + 1}" for i in range(len(self.nets)))
        return names[:len(self.nets)]

    @property
    def pins(self) -> dict[str, str]:
        """pin name -> net"""
        return dict(zip(self.pin_names, self.nets))

    def net(self, pin: str) -> str:
        return self.pins[pin]

    @property
    def is_mos(self) -> bool:
        return self.kind in MOS

    def to_json(self) -> dict:
        d = {"name": self.name, "kind": self.kind, "nets": list(self.nets)}
        if self.params:
            d["params"] = dict(self.params)
        return d

    @staticmethod
    def from_json(d: dict) -> "Device":
        return Device(d["name"], d["kind"], tuple(d["nets"]), dict(d.get("params", {})))


@dataclass
class Schematic:
    """A flat circuit: devices, external ports, where it came from.

    ``constraints`` holds what a source *declares* (e.g. the matched pairs of a
    curated benchmark), never what a tool inferred. ``warnings`` lists what a
    reader had to repair or skip.
    """
    name: str
    ports: tuple[str, ...]
    devices: tuple[Device, ...]
    provenance: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.ports = tuple(self.ports)
        self.devices = tuple(self.devices)
        names = [d.name for d in self.devices]
        dup = sorted(n for n, c in Counter(names).items() if c > 1)
        if dup:
            raise ValueError(f"schematic {self.name}: duplicate device names {dup}")

    def __getitem__(self, name: str) -> Device:
        for d in self.devices:
            if d.name == name:
                return d
        raise KeyError(name)

    def nets(self) -> list[str]:
        """Every net, ports first (in port order), then by first use."""
        out = list(self.ports)
        for d in self.devices:
            for n in d.nets:
                if n not in out:
                    out.append(n)
        return out

    def net_pins(self) -> dict[str, list[tuple[str, str]]]:
        """net -> [(device, pin)], ports included even when nothing connects to them."""
        out: dict[str, list[tuple[str, str]]] = {n: [] for n in self.ports}
        for d in self.devices:
            for pin, net in d.pins.items():
                out.setdefault(net, []).append((d.name, pin))
        return out

    def net_class(self, net: str) -> str:
        return net_class(net)

    def kinds(self) -> Counter:
        return Counter(d.kind for d in self.devices)

    @property
    def is_mos_only(self) -> bool:
        return bool(self.devices) and all(d.is_mos for d in self.devices)

    def check(self) -> list[str]:
        """Structural problems that do not stop a reader: unconnected ports, single-pin nets, unknown kinds."""
        problems = []
        pins = self.net_pins()
        for p in self.ports:
            if not pins[p]:
                problems.append(f"port {p} connects to nothing")
        for n, ps in pins.items():
            if n not in self.ports and len(ps) == 1:
                problems.append(f"net {n} has one pin ({ps[0][0]}.{ps[0][1]})")
        for d in self.devices:
            if d.kind not in PINS:
                problems.append(f"device {d.name} has unknown kind {d.kind}")
        return problems

    # ---- serialisation ---------------------------------------------------
    def to_json(self) -> dict:
        return {"name": self.name, "ports": list(self.ports), "devices": [d.to_json() for d in self.devices],
                "provenance": self.provenance, "constraints": self.constraints, "warnings": list(self.warnings)}

    @staticmethod
    def from_json(d: dict) -> "Schematic":
        return Schematic(d["name"], tuple(d["ports"]), tuple(Device.from_json(x) for x in d["devices"]),
                         dict(d.get("provenance", {})), dict(d.get("constraints", {})), list(d.get("warnings", [])))

    def sha(self) -> str:
        """Hash of ports and devices (names included). Not an isomorphism class."""
        body = {"ports": list(self.ports), "devices": [d.to_json() for d in self.devices]}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def summary(self, max_devices: Optional[int] = None) -> str:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(self.kinds().items()))
        lines = [f"{self.name}: {len(self.devices)} devices ({kinds}), ports {' '.join(self.ports)}"]
        for d in self.devices[:max_devices]:
            lines.append(f"  {d.name:<8} {d.kind:<6} " + " ".join(f"{p}={n}" for p, n in d.pins.items()))
        return "\n".join(lines)

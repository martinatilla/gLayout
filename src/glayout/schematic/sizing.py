"""Synthetic sizing for layout generation. **Layout only: no electrical meaning.**

AnalogGenie netlists carry topology only. To draw them, every block gets:

- ``length``: k × the PDK's minimum length, k drawn per block kind (mirrors
  and bias devices longer, input pairs and cascodes shorter);
- ``unit``: total width per *base finger*, in µm, drawn per block kind;
- ``base``: a base finger count per device (1, or 2 for some mirror outputs
  and bias-bank branches), so a device's total width is ``base × unit``;
- ``splits``: the shapes the floorplanner may choose from. Split k draws
  ``base × k`` fingers of width ``unit / k``: the same total width as 2, 4 or
  8 fingers, like ``floorplan.examples._SPLITS``.

Matched devices get equal sizes: every device of a self-symmetric block has
the same base, and the two blocks of a symmetric pair are sized alike. The
random choices are seeded by the schematic name and the seed, so a sizing
is reproducible on any machine.

Pure Python: the PDK enters only as numbers (``lmin``, ``wmin``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from glayout.schematic.annotate import Annotation
from glayout.schematic.model import Schematic

# block kind -> (unit widths in µm, length multiples of lmin)
KIND_RULES: dict[str, tuple[tuple[float, ...], tuple[int, ...]]] = {
    "DP": ((8.0, 12.0, 16.0), (1, 2)),
    "CM": ((4.0, 6.0, 8.0), (2, 4)),
    "CMB": ((4.0, 6.0, 8.0), (2, 4)),
    "BIAS": ((4.0, 6.0, 8.0), (2, 4)),
    "CASC": ((6.0, 8.0, 12.0), (1, 2)),
}
DEFAULT_RULE = ((4.0, 6.0, 8.0, 12.0), (1, 2))
SPLITS = (2, 4, 8)
NOTE = "synthetic sizing for layout generation; no electrical meaning"


@dataclass
class BlockSize:
    unit: float                       # µm of width per base finger
    length: float                     # µm
    base: dict[str, int]              # device -> base finger count
    splits: tuple[int, ...]           # variant i draws base*k fingers of width unit/k

    def variants(self) -> list[tuple[int, float]]:
        """(k, finger width) per variant."""
        return [(k, round(self.unit / k, 6)) for k in self.splits]

    def total_width(self, device: str) -> float:
        return self.base[device] * self.unit

    def to_json(self) -> dict:
        return {"unit": self.unit, "length": self.length, "base": dict(self.base), "splits": list(self.splits)}

    @staticmethod
    def from_json(d: dict) -> "BlockSize":
        return BlockSize(float(d["unit"]), float(d["length"]), {k: int(v) for k, v in d["base"].items()},
                         tuple(int(k) for k in d["splits"]))


@dataclass
class Sizing:
    schematic: str
    seed: int
    lmin: float
    wmin: float
    blocks: dict[str, BlockSize]                  # annotation block name -> size
    skipped: list[str] = field(default_factory=list)   # blocks without MOS devices (not sized)
    note: str = NOTE

    def device(self, name: str) -> tuple[float, float]:
        """(total width, length) of a device, µm."""
        for b in self.blocks.values():
            if name in b.base:
                return b.total_width(name), b.length
        raise KeyError(name)

    def to_json(self) -> dict:
        return {"schematic": self.schematic, "seed": self.seed, "lmin": self.lmin, "wmin": self.wmin,
                "blocks": {k: v.to_json() for k, v in self.blocks.items()}, "skipped": list(self.skipped),
                "note": self.note}

    @staticmethod
    def from_json(d: dict) -> "Sizing":
        return Sizing(d["schematic"], int(d["seed"]), float(d["lmin"]), float(d["wmin"]),
                      {k: BlockSize.from_json(v) for k, v in d["blocks"].items()}, list(d.get("skipped", [])),
                      d.get("note", NOTE))


def _splits(unit: float, fingers: int, wmin: float, max_fingers: int, splits: Sequence[int]) -> tuple[int, ...]:
    ok = tuple(k for k in splits if unit / k >= wmin - 1e-9 and fingers * k <= max_fingers)
    if ok:
        return ok
    # nothing fits the finger cap: the fewest fingers that are still wide enough, else one finger each
    wide = [k for k in sorted(splits) if unit / k >= wmin - 1e-9]
    return (wide[0],) if wide else (1,)


def size_schematic(sch: Schematic, ann: Annotation, lmin: float, wmin: float = 0.5, seed: int = 0,
                   max_fingers: int = 24, splits: Sequence[int] = SPLITS,
                   rules: Optional[dict] = None) -> Sizing:
    """Sizes for every MOS block of ``ann``. Passive and other blocks are listed in ``skipped``."""
    rules = {**KIND_RULES, **(rules or {})}
    rng = random.Random(f"{sch.name}:{seed}")
    partner = {}
    for a, b in ann.symmetric_pairs:
        partner[b] = a
    out: dict[str, BlockSize] = {}
    skipped = []
    for blk in ann.blocks:
        if blk.device_kind not in ("nfet", "pfet"):
            skipped.append(blk.name)
            continue
        units, lmults = rules.get(blk.kind, DEFAULT_RULE)
        unit, lmult = rng.choice(units), rng.choice(lmults)
        if blk.name in partner and partner[blk.name] in out:
            twin = out[partner[blk.name]]
            out[blk.name] = BlockSize(twin.unit, twin.length, {blk.devices[0]: next(iter(twin.base.values()))},
                                      twin.splits)
            continue
        base = {d: 1 for d in blk.devices}
        if blk.name not in ann.self_symmetric and blk.kind in ("CM", "BIAS"):
            for d in blk.devices:
                dev = sch[d]
                if dev.net("D") != dev.net("G"):          # outputs: 1:1 or 1:2 against the diode
                    base[d] = rng.choice((1, 2))
        out[blk.name] = BlockSize(unit, round(lmult * lmin, 6), base,
                                  _splits(unit, sum(base.values()), wmin, max_fingers, splits))
    return Sizing(sch.name, seed, lmin, wmin, out, skipped)

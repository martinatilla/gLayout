"""Per-PDK settings for extraction and simulation.

Everything PDK-specific in the harness is here, so a second PDK is one more
``SimConfig`` plus whatever its extractor needs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SimConfig:
    """How to extract and simulate one PDK.

    glayout's reference netlists write every primitive FET as an ``X``
    instance with unitless µm ``w``/``l`` (right for sky130, whose models are
    subcircuits in µm). ``fet_prefix`` and ``si_units`` say how this PDK's
    models want them instead.
    """
    name: str                       # glayout PDK name
    kpex_pdk: str                   # kpex --pdk
    vdd: float
    fets: tuple[str, ...]           # primitive FET model names
    fet_prefix: str                 # "M": .model cards; "X": subcircuits
    si_units: bool                  # bare w=/l= need a "u"
    model_lines: tuple[str, ...]    # deck lines that load the models ({pdk_root}, {corner})
    default_corner: str
    default_substrate: str          # the LVS deck's substrate net when a cell has no bulk pin

    def models(self, corner: Optional[str] = None, pdk_root: Optional[str] = None) -> str:
        """The deck lines that load the device models for ``corner``."""
        root = pdk_root or os.environ.get("PDK_ROOT")
        if not root:
            raise EnvironmentError("PDK_ROOT is not set")
        lines = [line.format(pdk_root=root, corner=corner or self.default_corner) for line in self.model_lines]
        for line in lines:
            path = Path(line.split()[1])
            if not path.is_file():
                raise FileNotFoundError(f"{self.name} model file missing: {path}")
        return "\n".join(lines)


GF180 = SimConfig(
    name="gf180",
    kpex_pdk="gf180mcuD",
    vdd=3.3,
    fets=("nfet_03v3", "pfet_03v3"),
    fet_prefix="M",
    si_units=True,
    model_lines=(
        ".include {pdk_root}/gf180mcuD/libs.tech/ngspice/design.ngspice",
        ".lib {pdk_root}/gf180mcuD/libs.tech/ngspice/sm141064.ngspice {corner}",
    ),
    default_corner="typical",       # also ff, ss, fs, sf
    default_substrate="gf180mcu_gnd",
)

_CONFIGS = {c.name: c for c in (GF180,)}


def sim_config(pdk) -> SimConfig:
    """``SimConfig`` for a ``MappedPDK`` or a PDK name."""
    name = (pdk if isinstance(pdk, str) else pdk.name).lower()
    try:
        return _CONFIGS[name]
    except KeyError:
        raise NotImplementedError(
            f"no evaluation config for PDK {name!r} (available: {', '.join(_CONFIGS)})") from None

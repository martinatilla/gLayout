"""Testbenches: an ngspice deck around a DUT subcircuit, and the metrics read back.

A testbench knows the DUT only by its port names (``ports``), so the same
deck runs unchanged on the schematic, the extracted devices and the parasitic
netlist; the views differ only in the file the deck includes.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from glayout.evaluation.config import SimConfig
from glayout.evaluation.ngspice import read_raw

Metrics = dict[str, Optional[float]]


class Testbench(ABC):
    name: str
    ground: str         # the port the substrate is tied to

    @property
    @abstractmethod
    def ports(self) -> tuple[str, ...]:
        """DUT pins, in the order the deck instantiates them."""

    @abstractmethod
    def deck(self, dut: str, include: str, cfg: SimConfig) -> str:
        """Deck text that includes ``include``, instances subckt ``dut`` and writes rawfiles."""

    @abstractmethod
    def metrics(self, workdir: Path, cfg: SimConfig) -> Metrics:
        """Read the rawfiles the deck wrote in ``workdir``."""

    def params(self) -> dict:
        return asdict(self) if hasattr(self, "__dataclass_fields__") else {}


@dataclass
class OtaTestbench(Testbench):
    """Open-loop gain and phase plus the DC operating point of a single-ended OTA.

    The loop is closed at DC and open in AC: 1 GH from the output to the
    inverting input, 1 GF from that input to ground. At DC the OTA is a
    unity-gain follower, so it sits at the operating point it has in feedback
    (VOUT = VCM + systematic offset) instead of railing on its own offset; at
    every AC frequency the inductor is open and the capacitor holds the
    inverting input at AC ground, so V(out)/V(ac) is the open-loop gain.

    ``ibias`` > 0 flows from the supply into the bias pin (an NMOS diode, as in
    the finger-sequence mirrors); ``ibias`` < 0 flows out of it to ground.
    ``inp`` must be the non-inverting input, or the DC loop is positive feedback.
    """
    ibias: float = 20e-6
    cl: float = 1e-12
    vcm: Optional[float] = None         # default VDD/2
    vdd: Optional[float] = None         # default from the PDK config
    corner: Optional[str] = None        # default from the PDK config
    fstart: float = 1.0
    fstop: float = 10e9
    points_per_decade: int = 20
    inp: str = "VINP"
    inn: str = "VINN"
    out: str = "VOUT"
    bias: str = "IBIAS"
    supply: str = "VDD"
    ground: str = "VSS"
    name: str = "ota"

    @property
    def ports(self) -> tuple[str, ...]:
        return (self.inp, self.inn, self.out, self.bias, self.supply, self.ground)

    def levels(self, cfg: SimConfig) -> tuple[float, float]:
        vdd = cfg.vdd if self.vdd is None else self.vdd
        return vdd, vdd / 2 if self.vcm is None else self.vcm

    def deck(self, dut: str, include: str, cfg: SimConfig) -> str:
        vdd, vcm = self.levels(cfg)
        if self.ibias >= 0:
            bias = f"IB {self.supply} {self.bias} DC {self.ibias:g}"
        else:
            bias = f"IB {self.bias} {self.ground} DC {-self.ibias:g}"
        return "\n".join([
            f"* {self.name} testbench: {dut}",
            cfg.models(self.corner),
            f".include {include}",
            f"XDUT {' '.join(self.ports)} {dut}",
            f"VSUP {self.supply} 0 DC {vdd:g}",
            f"VGND {self.ground} 0 DC 0",
            bias,
            f"VCM vcm 0 DC {vcm:g}",
            f"VAC {self.inp} vcm DC 0 AC 1",
            f"LFB {self.out} {self.inn} 1G",
            f"CFB {self.inn} 0 1G",
            f"CL {self.out} 0 {self.cl:g}",
            ".control",
            "set filetype=ascii",
            "op",
            "write op.raw all",
            f"ac dec {self.points_per_decade} {self.fstart:g} {self.fstop:g}",
            f"write ac.raw v({self.out})",
            "quit",
            ".endc",
            ".end",
            "",
        ])

    def metrics(self, workdir: Path, cfg: SimConfig) -> Metrics:
        vdd, vcm = self.levels(cfg)
        op = read_raw(Path(workdir) / "op.raw")[0]
        ac = read_raw(Path(workdir) / "ac.raw")[0]
        vout = float(op[f"v({self.out})"][0].real)
        m = open_loop_metrics(ac["frequency"].real, ac[f"v({self.out})"])
        m.update({"vout_dc_v": vout, "vos_sys_v": vout - vcm, "isupply_a": -float(op["i(vsup)"][0].real)})
        return m


def open_loop_metrics(f: np.ndarray, h: np.ndarray) -> Metrics:
    """DC gain, -3 dB frequency, unity-gain frequency and phase margin of an open-loop response.

    The phase is unwrapped and referred to its low-frequency value, so a
    non-inverting response starts near 0° and PM = 180° + phase at the UGF.
    ``phase0_deg`` far from 0 means the inputs are swapped. Crossings are
    interpolated on log-log axes; a crossing outside the sweep is ``None``.
    """
    mag = np.abs(h)
    phase = np.degrees(np.unwrap(np.angle(h)))
    phase -= 360.0 * np.round(phase[0] / 360.0)
    ugf = _falling_crossing(f, mag, 1.0)
    pm = None
    if ugf is not None:
        pm = 180.0 + float(np.interp(math.log10(ugf), np.log10(f), phase))
    return {
        "dc_gain_db": 20.0 * math.log10(mag[0]),
        "f3db_hz": _falling_crossing(f, mag, mag[0] / math.sqrt(2.0)),
        "ugf_hz": ugf,
        "pm_deg": pm,
        "phase0_deg": float(phase[0]),
    }


def _falling_crossing(f: np.ndarray, y: np.ndarray, level: float) -> Optional[float]:
    below = np.nonzero(y < level)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = int(below[0])
    x0, x1 = math.log10(f[i - 1]), math.log10(f[i])
    y0, y1 = math.log10(y[i - 1]), math.log10(y[i])
    return 10 ** (x0 + (math.log10(level) - y0) / (y1 - y0) * (x1 - x0))

"""Parasitic extraction with KLayout-PEX (kpex).

kpex runs the PDK's KLayout LVS deck for connectivity, then its 2.5D engine
for capacitance. One run gives two netlists of the same layout:

- ``devices``: the LVS-extracted netlist: one device per finger with its
  drawn AS/AD/PS/PD, no wiring parasitics;
- ``parasitic``: the same devices plus every extracted capacitance (to
  substrate, to other nets, between wires of the same layer).

kpex 0.4.4 (iic-osic-tools 2026.09) needs four workarounds, found bringing
this up on the routed 5T OTA:

1. It calls the LVS deck without ``target_netlist``; the gf180 deck then
   writes ``/<cell>_extracted.cir``, fails, and kpex dies looking for the
   LVSDB. ``KPEX_KLAYOUT_EXE`` points at a shim that adds the path and
   ``lvs_sub``, so the substrate gets the cell's bulk net name instead of
   ``gf180mcu_gnd``.
2. The 2.5D ``RC`` and ``R`` modes write 0-ohm resistors between subnodes no
   device connects to, so only ``CC`` is supported here.
3. Its LVS keeps devices uncombined (what simulation wants), so its verdict
   against a reference that writes ``m=4`` is always a mismatch; it gets a
   dummy schematic, and ``lvs.py`` checks LVS separately.
4. Its copy of the gf180 deck fails on ``metal_level=5LM`` (a nil layer), so
   LVS runs at the deck's 6LM default. Only connectivity comes from there;
   the 2.5D process stack is the 5-metal, thick-top-metal one.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from glayout.evaluation.config import SimConfig

PathLike = Union[str, Path]

_SHIM = """#!/bin/bash
# Written by glayout.evaluation.extract: kpex runs the LVS deck (klayout -r)
# without an extracted-netlist path or substrate name; add them.
for a in "$@"; do
  if [ "$a" = "-r" ]; then exec {klayout} "$@" {extra}; fi
done
exec {klayout} "$@"
"""


class ExtractionError(RuntimeError):
    pass


@dataclass
class Extraction:
    cell: str
    devices: Path           # LVS-extracted devices, no parasitics
    parasitic: Path         # devices + extracted capacitance
    mode: str
    substrate: str          # the substrate net's name in both netlists
    out_dir: Path
    seconds: float


def tool(name: str, env_var: str) -> str:
    exe = os.environ.get(env_var) or shutil.which(name)
    if not exe:
        raise FileNotFoundError(f"{name} not found: put it on PATH or set {env_var}")
    return exe


def kpex_extract(gds: PathLike, cell: str, workdir: PathLike, cfg: SimConfig, substrate: str,
                 mode: str = "CC", timeout: float = 1800.0) -> Extraction:
    """Extract ``cell`` from ``gds`` into ``<workdir>/kpex`` (cleared first)."""
    if mode != "CC":
        raise ValueError(f"kpex 2.5D mode {mode!r} is not usable with kpex 0.4.4 (see module docstring); use 'CC'")
    out = Path(workdir).resolve() / "kpex"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    devices = out / f"{cell}_devices.cir"

    extra = " ".join(f"-rd {shlex.quote(f'{k}={v}')}" for k, v in
                     {"target_netlist": devices, "lvs_sub": substrate}.items())
    shim = out / "klayout_shim.sh"
    shim.write_text(_SHIM.format(klayout=shlex.quote(tool("klayout", "KLAYOUT_EXE")), extra=extra))
    shim.chmod(0o755)
    env = {**os.environ, "KPEX_KLAYOUT_EXE": str(shim), "PDK": cfg.kpex_pdk}
    cmd = [tool("kpex", "KPEX_EXE"), "extract", "--pdk", cfg.kpex_pdk, "--gds", str(Path(gds).resolve()),
           "--cell", cell, "--2.5D", "--mode", mode, "--out_dir", str(out), "--cache-lvs", "false",
           "--log_level", "warning"]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True, timeout=timeout)
    log = out / "kpex_stdout.log"
    log.write_text(proc.stdout + proc.stderr)
    found = sorted(out.glob(f"*/{cell}_k25d_pex_netlist.spice"))
    if proc.returncode != 0 or not found or not devices.is_file():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
        raise ExtractionError(f"kpex failed on {cell} (exit {proc.returncode}, log {log}):\n{tail}")
    return Extraction(cell=cell, devices=devices, parasitic=found[0], mode=mode, substrate=substrate,
                      out_dir=out, seconds=time.perf_counter() - t0)

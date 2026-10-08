"""Run ngspice in batch mode and read the ASCII rawfiles a deck writes.

Decks ``write`` each analysis to its own rawfile (``set filetype=ascii``);
metrics are computed in Python from the vectors, which reports a missing
crossing as ``None`` where a ``.meas`` would just fail.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]

# ngspice exits 0 on most analysis failures, so the log is checked too
_FAILURE = re.compile(r"^\s*error\b|singular matrix|timestep too small|no convergence|gmin stepping failed",
                      re.IGNORECASE | re.MULTILINE)


class SimulationError(RuntimeError):
    pass


@dataclass
class Plot:
    name: str                          # e.g. "AC Analysis"
    vectors: dict[str, np.ndarray]     # lower-case vector name -> values (complex for AC)

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self.vectors[key.lower()]
        except KeyError:
            raise KeyError(f"{key} not in plot {self.name!r} (have {', '.join(self.vectors)})") from None


def read_raw(path: PathLike) -> list[Plot]:
    """Every plot in an ASCII rawfile."""
    plots = []
    for block in re.split(r"^(?=Title:)", Path(path).read_text(), flags=re.MULTILINE):
        if "Values:" not in block:
            continue
        header, values = block.split("Values:", 1)
        fields = dict(re.findall(r"^([A-Za-z. ]+):[ \t]*(.*)$", header, re.MULTILINE))
        nvars, npts = int(fields["No. Variables"]), int(fields["No. Points"])
        names = re.findall(r"^[ \t]+\d+[ \t]+(\S+)", header.split("\nVariables:", 1)[1], re.MULTILINE)[:nvars]
        is_complex = "complex" in fields.get("Flags", "")
        tokens = values.split()
        data = np.empty((npts, nvars), dtype=complex if is_complex else float)
        k = 0
        for p in range(npts):
            k += 1                                     # point index
            for v in range(nvars):
                tok = tokens[k]
                k += 1
                if is_complex:
                    re_, im = tok.split(",")
                    data[p, v] = complex(float(re_), float(im))
                else:
                    data[p, v] = float(tok)
        plots.append(Plot(fields.get("Plotname", "").strip(),
                          {n.lower(): data[:, i] for i, n in enumerate(names)}))
    return plots


def ngspice_exe() -> str:
    exe = os.environ.get("NGSPICE_EXE") or shutil.which("ngspice")
    if not exe:
        raise FileNotFoundError("ngspice not found: put it on PATH or set NGSPICE_EXE")
    return exe


def run_ngspice(deck: str, workdir: PathLike, name: str = "tb", timeout: float = 600.0) -> float:
    """Write ``<name>.spice`` in ``workdir`` and run it there; returns seconds.

    Raises ``SimulationError`` with the failing log lines when ngspice exits
    non-zero or reports an analysis failure.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    deck_path = workdir / f"{name}.spice"
    log_path = workdir / f"{name}.log"
    deck_path.write_text(deck)
    t0 = time.perf_counter()
    proc = subprocess.run([ngspice_exe(), "-b", deck_path.name], cwd=workdir,
                          capture_output=True, text=True, timeout=timeout)
    log = proc.stdout + proc.stderr
    log_path.write_text(log)
    bad = [line.strip() for line in log.splitlines() if _FAILURE.search(line)]
    if proc.returncode != 0 or bad:
        detail = "; ".join(bad[:4]) or "\n".join(log.strip().splitlines()[-8:])
        raise SimulationError(f"ngspice failed on {deck_path} (exit {proc.returncode}): {detail}")
    return time.perf_counter() - t0

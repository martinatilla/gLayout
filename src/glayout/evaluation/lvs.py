"""LVS gate: a layout's simulated numbers mean nothing unless it matches its netlist.

gf180 runs the PDK's KLayout deck with the switches of the repo's LVS runner
(tests/lvs/klayout_gf180.py), which passes on the routed OTAs: flat mode,
device combining, schematic simplification, 5LM / 11K top metal, MIM option
A, the substrate named after the reference's bulk pin, and only the labels
the reference declares as pins. That module lives under tests/ and is not
importable from the package, hence the small copy. Its MIM option A deck
patch is not copied: a layout with MIM caps fails here rather than passing
wrongly.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from glayout.evaluation.config import SimConfig, sim_config
from glayout.evaluation.extract import tool
from glayout.evaluation.netlist import subckt_ports, to_sim_dialect

PathLike = Union[str, Path]


@dataclass
class LvsResult:
    passed: bool
    conclusion: str
    report: Path
    seconds: float


def lvs_reference(reference: str, cell: str, cfg: Optional[SimConfig]) -> str:
    """A glayout reference netlist as the LVS schematic for layout cell ``cell``.

    The top subckt is renamed to the cell (KLayout pairs the schematic's top
    circuit with the layout's top cell by name) and, given a ``cfg``, the
    primitives are put in the PDK's dialect (gf180: ``M`` devices, ``u`` units).
    The raw netlist fails gf180 KLayout LVS, even with the deck's
    ``schematic_translate``.
    """
    top, _ = subckt_ports(reference)
    text = re.sub(rf"\b{re.escape(top)}\b", cell, reference)
    return to_sim_dialect(text, cfg) if cfg else text


def write_lvs_netlist(component, pdk, path: PathLike) -> Path:
    """Write ``component.info["netlist"]`` as an LVS schematic to ``path``.

    Saved as ``<gds name>.spice`` next to the GDS, the gf180 PDK's KLayout LVS
    menu finds it on its own (with its Netlist Path option empty). A PDK
    without a ``SimConfig`` (sky130) gets the reference with only the top
    subckt renamed: the form glayout's own netgen LVS reads.
    """
    netlist = component.info.get("netlist")
    if netlist is None:
        raise ValueError(f"{component.name} has no info['netlist']")
    reference = netlist.generate_netlist() if hasattr(netlist, "generate_netlist") else str(netlist)
    try:
        cfg = sim_config(pdk)
    except NotImplementedError:
        cfg = None
    path = Path(path)
    path.write_text(lvs_reference(reference, component.name, cfg).rstrip("\n") + "\n")
    return path


def substrate_name(ports: Sequence[str], default: str) -> str:
    """The reference's bulk pin, which the deck must name the substrate after.

    Otherwise every NMOS bulk is an unmatched net. VSS comes last: it is
    usually a source rail, and a cell with a separate B pin wants B.
    """
    upper = {p.upper(): p for p in ports}
    for cand in ("B", "VBULK", "VSUB", "GND", "VSS"):
        if cand in upper:
            return upper[cand]
    return default


def _gf180_deck(pdk_root: Path) -> Path:
    candidates = []
    pointer = pdk_root / "ciel" / "gf180mcu" / "current"
    if pointer.is_file():
        candidates.append(pdk_root / "ciel" / "gf180mcu" / "versions" / pointer.read_text().strip() / "gf180mcuD")
    candidates.append(pdk_root / "gf180mcuD")
    for base in candidates:
        deck = base / "libs.tech" / "klayout" / "tech" / "lvs" / "gf180mcu.lvs"
        if deck.is_file():
            return deck
    raise FileNotFoundError(f"gf180mcu.lvs not found under {pdk_root}")


def _keep_pin_labels(gds: Path, ports: Sequence[str]) -> list[str]:
    """Drop labels that are not reference pins (a composite inherits its children's); returns them."""
    import gdstk
    keep = {p.upper() for p in ports}
    lib = gdstk.read_gds(str(gds))
    dropped = []
    for cell in lib.cells:
        for label in list(cell.labels):
            if label.text.upper() not in keep:
                cell.remove(label)
                dropped.append(label.text)
    if dropped:
        lib.write_gds(str(gds))
    return dropped


def lvs_gf180(gds: PathLike, cell: str, reference: str, workdir: PathLike, cfg: SimConfig,
              substrate: str, timeout: float = 1800.0) -> LvsResult:
    """LVS ``cell`` against the glayout ``reference`` netlist text, in ``<workdir>/lvs``."""
    out = Path(workdir).resolve() / "lvs"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    staged = out / f"{cell}.gds"
    shutil.copy(gds, staged)
    _, ports = subckt_ports(reference)
    spice = out / f"{cell}.spice"
    spice.write_text(lvs_reference(reference, cell, cfg))
    _keep_pin_labels(staged, ports)

    switches = {
        "input": staged, "schematic": spice, "topcell": cell,
        "target_netlist": out / f"{cell}.cir", "report": out / f"{cell}.lvsdb",
        "mim_option": "A", "metal_level": "5LM", "metal_top": "11K", "poly_res": "1k", "mim_cap": "2",
        "run_mode": "flat", "combine": "true", "schematic_simplify": "true", "top_lvl_pins": "true",
        "spice_net_names": "true", "lvs_sub": substrate, "thr": str(min(os.cpu_count() or 2, 8)),
    }
    if not os.environ.get("PDK_ROOT"):
        raise EnvironmentError("PDK_ROOT is not set")
    cmd = [tool("klayout", "KLAYOUT_EXE"), "-b", "-r", str(_gf180_deck(Path(os.environ["PDK_ROOT"])))]
    for k, v in switches.items():
        cmd += ["-rd", f"{k}={v}"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=out, capture_output=True, text=True, timeout=timeout)
    log = proc.stdout + proc.stderr
    report = out / f"{cell}_lvs.log"
    report.write_text(log)
    if re.search(r"Congratulations!\s*Netlists\s*match", log):
        passed, conclusion = True, "netlists match"
    elif re.search(r"Netlists\s*don.t\s*match|Netlists do not match", log):
        passed, conclusion = False, "netlists do not match"
    else:
        passed, conclusion = False, f"inconclusive (klayout exit {proc.returncode})"
    return LvsResult(passed, conclusion, report, time.perf_counter() - t0)

"""Cell parasitics: every variant of a floorplan ``Circuit``, extracted on its own.

``extract_cell_parasitics(pdk, circuit)`` fills ``Variant.parasitics`` with the
capacitors kpex finds in the variant's cell alone, nets named by the block's
pin nets and the substrate as ``VSUBS``. The graph (``glayout.schematic.graph``,
schema 2.1) turns them into ``variant__loads__net`` edges and ``cell_coupling``
nodes. A cell is built before floorplanning, so this is schematic-stage
information; on the two feasibility OTAs (gf180) the per-variant values add up
to the extracted layouts' with a fitted scale of 0.99 to ground and 1.01
between nets.

The routed composite uses cells without pin labels; kpex needs the labels to
name the nets, so each variant is rebuilt with them from its parameters
(``finger_sequence_cell`` keyword arguments, as ``schematic_to_circuit`` makes
them). Variants of other builders are skipped. Results are cached per PDK and
parameters for the life of the process.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

from glayout.evaluation.config import sim_config
from glayout.evaluation.extract import kpex_extract
from glayout.evaluation.lvs import substrate_name
from glayout.evaluation.netlist import capacitors, subckt_ports
from glayout.util.process_table import ProcessTable

PathLike = Union[str, Path]
SUBSTRATE = "VSUBS"
_CACHE: dict[str, list[tuple[str, str, float]]] = {}


def _key(pdk, params: dict) -> str:
    body = json.dumps({"pdk": pdk.name, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _labelled_cell(pdk, params: dict, key: str):
    from glayout.cells.elementary.finger_sequence import finger_sequence_cell
    from glayout.schematic.to_circuit import _pin_labels

    with _pin_labels(True):
        return finger_sequence_cell(pdk, **{**params, "name": f"{params.get('name', 'cell')}_px{key[:6]}"})


def extract_cell_parasitics(pdk, circuit, workdir: Optional[PathLike] = None, workers: int = 4) -> dict:
    """Set ``parasitics`` on every variant of ``circuit`` that a kpex run can name; returns counts and seconds."""
    cfg = sim_config(pdk)
    root = Path(workdir) if workdir else Path(tempfile.gettempdir()) / "glayout_cells"
    jobs, skipped = {}, 0
    for inst in circuit.instances:
        for var in inst.block.variants:
            if not {"devices", "device"} <= set(var.params):
                skipped += 1
                continue
            key = _key(pdk, var.params)
            if key not in _CACHE and key not in jobs:
                jobs[key] = var.params

    def build(key: str, params: dict) -> dict:
        comp = _labelled_cell(pdk, params, key)
        out = root / key
        out.mkdir(parents=True, exist_ok=True)
        gds = out / f"{comp.name}.gds"
        comp.write_gds(str(gds))
        nl = comp.info["netlist"]
        _, pins = subckt_ports(nl.generate_netlist() if hasattr(nl, "generate_netlist") else str(nl))
        return {"key": key, "cell": comp.name, "gds": gds, "out": out, "pins": pins,
                "sub": substrate_name(pins, cfg.default_substrate)}

    def run(job: dict):
        ex = kpex_extract(job["gds"], job["cell"], job["out"], cfg, job["sub"])
        text = next((job["out"] / "kpex").glob(f"*/{job['cell']}_k25d_pex_netlist.spice")).read_text()
        # the bulk is a pin of the cell unless it is the substrate itself
        rename = {} if job["sub"] in job["pins"] else {job["sub"]: SUBSTRATE}
        caps = [(rename.get(a, a), rename.get(b, b), c) for a, b, c in capacitors(text)]
        return job["key"], caps, ex.seconds

    with ProcessTable("Cell parasitics", f"{len(jobs)} cells to extract, {workers} at a time") as pt:
        secs = 0.0
        if jobs:
            # cells are built one at a time (the pin-label switch is process-wide); kpex runs in parallel
            built = [build(k, p) for k, p in jobs.items()]
            with ThreadPoolExecutor(workers) as pool:
                for key, caps, s in pool.map(run, built):
                    _CACHE[key] = caps
                    secs += s
        done = 0
        for inst in circuit.instances:
            for var in inst.block.variants:
                key = _key(pdk, var.params) if {"devices", "device"} <= set(var.params) else None
                if key in _CACHE:
                    var.parasitics = list(_CACHE[key])
                    done += 1
        pt.result("variants", done)
        pt.result("skipped", skipped)
        pt.result("kpex_s", round(secs, 1))
    return {"variants": done, "extracted": len(jobs), "skipped": skipped, "kpex_s": secs}

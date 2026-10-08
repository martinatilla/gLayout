"""``evaluate``: one layout, three netlists of it, one testbench.

    from glayout.evaluation import OtaTestbench, evaluate
    ev = evaluate(ota, OtaTestbench(ibias=20e-6, cl=1e-12), pdk)
    ev.metric("pex", "ugf_hz"), ev.gap("ugf_hz", "schematic", "pex")

Stages, recorded in one ProcessTable:

1. Inputs    the GDS top cell must be named as the cell, and the reference
             netlist's top subckt must have every testbench port.
2. LVS       layout vs reference; a failure stops the run unless
             ``require_lvs=False``.
3. Extract   kpex: LVS-extracted devices + 2.5D capacitance.
4. Adapt     each netlist -> ``.subckt DUT <testbench ports>``: device dialect,
             plain net names, substrate on the testbench ground, pins by name.
5. Simulate  the same deck on each view.

Views:

  schematic  the reference netlist: no AS/AD/PS/PD, so the model sizes the
             junctions itself (gf180's BSIM4 default diffusion geometry, about
             2.4 fF per 3 µm finger, not zero); no wiring
  devices    LVS-extracted: the drawn AS/AD/PS/PD (diffusion sharing), no wires
  pex        devices + extracted capacitance (wires, coupling, substrate)

so schematic -> devices is the drawn device layout (finger sequence, variant,
shared diffusion) against the model's default, and devices -> pex the effect
of routing and floorplan.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from glayout.evaluation.config import SimConfig, sim_config
from glayout.evaluation.extract import Extraction, kpex_extract
from glayout.evaluation.lvs import LvsResult, lvs_gf180, lvs_reference, substrate_name
from glayout.evaluation.netlist import (capacitance_by_net, capacitors, rename_nodes, sanitize_names,
                                        subckt_ports, to_sim_dialect, wrapper)
from glayout.evaluation.ngspice import SimulationError, run_ngspice
from glayout.evaluation.testbench import Metrics, Testbench
from glayout.util.process_table import ProcessTable, print_table

PathLike = Union[str, Path]
VIEWS = ("schematic", "devices", "pex")
DUT = "DUT"
KPEX_SUBSTRATE = "VSUBS"        # kpex's node for capacitance to substrate; not a subckt pin
_LVS = {"gf180": lvs_gf180}
_GAPS = (("schematic", "devices"), ("devices", "pex"), ("schematic", "pex"))


class EvaluationError(RuntimeError):
    pass


@dataclass
class ViewResult:
    netlist: Path
    metrics: Metrics = field(default_factory=dict)
    seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class Evaluation:
    cell: str
    pdk: str
    testbench: str
    params: dict
    workdir: Path
    lvs: Optional[LvsResult] = None
    extraction: Optional[Extraction] = None
    views: dict[str, ViewResult] = field(default_factory=dict)
    cap_by_net: dict[str, float] = field(default_factory=dict)     # extracted capacitance per net, F
    cap_total: float = 0.0                                          # sum of extracted capacitors, F

    def metric(self, view: str, key: str) -> Optional[float]:
        v = self.views.get(view)
        return None if v is None else v.metrics.get(key)

    def gap(self, key: str, a: str = "schematic", b: str = "pex") -> Optional[float]:
        """``key`` from view ``a`` to ``b``: a difference for dB, degrees and volts, else relative."""
        x, y = self.metric(a, key), self.metric(b, key)
        if x is None or y is None:
            return None
        if _absolute(key):
            return y - x
        return None if x == 0 else (y - x) / abs(x)

    def keys(self) -> list[str]:
        keys: list[str] = []
        for v in self.views.values():
            keys += [k for k in v.metrics if k not in keys]
        return keys

    def print(self) -> None:
        rows = [[k, *(_fmt(k, self.metric(v, k)) for v in VIEWS), *(_fmt_gap(k, self.gap(k, a, b)) for a, b in _GAPS)]
                for k in self.keys()]
        print_table(f"Evaluation {self.cell} ({self.pdk}, {self.testbench} testbench)",
                    ["metric", *VIEWS, "sch→dev", "dev→pex", "sch→pex"], rows,
                    caption="sch→dev: drawn junction geometry vs the model's default; dev→pex: extracted "
                            "wiring capacitance. dB, ° and V: difference; others: relative")
        if self.cap_by_net:
            print_table(f"Extracted capacitance by net ({self.cell})", ["net", "C"],
                        [(n, _eng(c, "F")) for n, c in list(self.cap_by_net.items())[:10]],
                        caption=f"a coupling capacitor counts on both nets; total extracted "
                                f"{_eng(self.cap_total, 'F')}")

    def to_dict(self) -> dict:
        return {
            "cell": self.cell, "pdk": self.pdk, "testbench": self.testbench, "params": self.params,
            "workdir": str(self.workdir),
            "lvs": None if self.lvs is None else {"passed": self.lvs.passed, "conclusion": self.lvs.conclusion,
                                                   "report": str(self.lvs.report)},
            "extraction": None if self.extraction is None else {
                "mode": self.extraction.mode, "substrate": self.extraction.substrate,
                "devices": str(self.extraction.devices), "parasitic": str(self.extraction.parasitic),
                "seconds": self.extraction.seconds},
            "views": {k: {"metrics": v.metrics, "error": v.error, "netlist": str(v.netlist), "seconds": v.seconds}
                      for k, v in self.views.items()},
            "gaps": {k: {f"{a}_to_{b}": self.gap(k, a, b) for a, b in _GAPS} for k in self.keys()},
            "cap_by_net_f": self.cap_by_net,
            "cap_total_f": self.cap_total,
        }


def evaluate(component, testbench: Testbench, pdk, workdir: Optional[PathLike] = None,
             require_lvs: bool = True) -> Evaluation:
    """Evaluate a glayout ``Component``: its GDS and LVS schematic (``<cell>.spice``) go into ``workdir``."""
    cell = component.name
    workdir = _workdir(workdir, cell)
    netlist = component.info.get("netlist")
    if netlist is None:
        raise EvaluationError(f"{cell} has no info['netlist'] (the reference netlist)")
    reference = netlist.generate_netlist() if hasattr(netlist, "generate_netlist") else str(netlist)
    gds = workdir / f"{cell}.gds"
    component.write_gds(str(gds))
    gds.with_suffix(".spice").write_text(lvs_reference(reference, cell, sim_config(pdk)))     # for GUI LVS
    return evaluate_gds(gds, cell, reference, testbench, pdk, workdir, require_lvs)


def evaluate_gds(gds: PathLike, cell: str, reference: str, testbench: Testbench, pdk,
                 workdir: Optional[PathLike] = None, require_lvs: bool = True) -> Evaluation:
    """Evaluate ``cell`` of ``gds`` against the ``reference`` netlist text.

    ``workdir`` (default ``$TMPDIR/glayout_eval/<cell>``) keeps every
    intermediate file: LVS, kpex, the three DUT netlists, decks, logs,
    rawfiles and ``evaluation.json``.
    """
    cfg = sim_config(pdk)
    if cfg.name not in _LVS:
        raise NotImplementedError(f"no LVS gate for {cfg.name}")
    workdir = _workdir(workdir, cell)
    ev = Evaluation(cell, cfg.name, testbench.name, testbench.params(), workdir)
    with ProcessTable(f"Evaluate {cell}", f"{cfg.name}, {testbench.name} testbench") as pt:
        with pt.stage("Inputs") as st:
            tops = _gds_top_cells(gds)
            if cell not in tops:
                raise EvaluationError(f"GDS top cell(s) {tops}, expected {cell!r}")
            ref_top, ref_ports = subckt_ports(reference)
            have = {p.upper() for p in ref_ports}
            missing = [p for p in testbench.ports if p.upper() not in have]
            if missing:
                raise EvaluationError(f"reference {ref_top}({' '.join(ref_ports)}) lacks testbench port(s) {missing}")
            sub = substrate_name(ref_ports, cfg.default_substrate)
            ties = {} if sub.upper() in {p.upper() for p in testbench.ports} else {sub: testbench.ground}
            st.detail(f"{ref_top}({' '.join(ref_ports)}); substrate {sub}"
                      + (f" tied to {testbench.ground}" if ties else ""))
        with pt.stage("LVS") as st:
            ev.lvs = _LVS[cfg.name](gds, cell, reference, workdir, cfg, sub)
            st.detail(f"{ev.lvs.conclusion}; {ev.lvs.report}")
            if not ev.lvs.passed:
                if require_lvs:
                    raise EvaluationError(f"LVS failed: {ev.lvs.conclusion} (require_lvs=False to simulate anyway)")
                st.warn("simulating anyway (require_lvs=False)")
        with pt.stage("Extract") as st:
            ev.extraction = ex = kpex_extract(gds, cell, workdir, cfg, sub)
            st.detail(f"kpex 2.5D {ex.mode}, {ex.seconds:.1f}s")
        with pt.stage("Adapt") as st:
            files = _dut_netlists(reference, ex, cfg, testbench, ties, workdir / "dut")
            caps = capacitors(files["pex"].read_text())
            ev.cap_by_net = capacitance_by_net(caps)
            ev.cap_total = sum(c for _, _, c in caps)
            st.detail(f"{len(caps)} capacitors, {_eng(ev.cap_total, 'F')} in total")
        for view in VIEWS:
            with pt.stage(f"Simulate {view}") as st:
                res = ev.views[view] = ViewResult(files[view])
                sim_dir = workdir / "sim" / view
                if sim_dir.exists():
                    shutil.rmtree(sim_dir)
                try:
                    res.seconds = run_ngspice(testbench.deck(DUT, str(files[view]), cfg), sim_dir)
                    res.metrics = testbench.metrics(sim_dir, cfg)
                except (SimulationError, KeyError, IndexError, ValueError) as exc:
                    res.error = str(exc)
                    st.fail(res.error[:300])
                    continue
                st.detail(", ".join(f"{k} {_fmt(k, v)}" for k, v in list(res.metrics.items())[:4]))
        pt.result("lvs", "pass" if ev.lvs.passed else "FAIL")
        for key in ("ugf_hz", "pm_deg"):
            if ev.gap(key) is not None:
                pt.result(f"{key} sch→pex", _fmt_gap(key, ev.gap(key)))
    ev.print()
    (workdir / "evaluation.json").write_text(json.dumps(ev.to_dict(), indent=2))
    return ev


def _dut_netlists(reference: str, ex: Extraction, cfg: SimConfig, testbench: Testbench,
                  ties: dict[str, str], out: Path) -> dict[str, Path]:
    """One include per view, each defining ``.subckt DUT <testbench ports>``."""
    out.mkdir(parents=True, exist_ok=True)
    texts = {
        "schematic": to_sim_dialect(reference, cfg),
        "devices": sanitize_names(ex.devices.read_text()),
        "pex": rename_nodes(sanitize_names(ex.parasitic.read_text()), {KPEX_SUBSTRATE: ex.substrate}),
    }
    files = {}
    for view, text in texts.items():
        top, pins = subckt_ports(text, None if view == "schematic" else ex.cell)
        path = out / f"dut_{view}.spice"
        path.write_text(f"* {view} view of {ex.cell}\n{text}\n\n{wrapper(top, pins, DUT, testbench.ports, ties)}")
        files[view] = path
    return files


def _workdir(workdir: Optional[PathLike], cell: str) -> Path:
    path = Path(workdir) if workdir else Path(tempfile.gettempdir()) / "glayout_eval" / cell
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _gds_top_cells(gds: PathLike) -> list[str]:
    import gdstk
    return [c.name for c in gdstk.read_gds(str(gds)).top_level()]


def _absolute(key: str) -> bool:
    return key.endswith(("_db", "_deg", "_v"))


_PREFIXES = ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"),
             (1e-12, "p"), (1e-15, "f"), (1e-18, "a"))


def _eng(value: float, unit: str) -> str:
    if value == 0:
        return f"0 {unit}"
    for scale, prefix in _PREFIXES:
        if abs(value) >= scale:
            return f"{value / scale:.4g} {prefix}{unit}"
    return f"{value:.3g} {unit}"


def _fmt(key: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if key.endswith("_db"):
        return f"{value:.2f} dB"
    if key.endswith("_deg"):
        return f"{value:.2f}°"
    unit = {"_hz": "Hz", "_v": "V", "_a": "A", "_f": "F", "_s": "s"}.get("_" + key.rsplit("_", 1)[-1], "")
    return _eng(value, unit) if unit else f"{value:.4g}"


def _fmt_gap(key: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if not _absolute(key):
        return f"{value:+.2%}"
    if key.endswith("_v"):
        return f"{value * 1e3:+.3f} mV"
    return f"{value:+.2f}" + (" dB" if key.endswith("_db") else "°")

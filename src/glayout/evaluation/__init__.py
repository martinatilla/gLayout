"""Post-layout evaluation: LVS, parasitic extraction (kpex) and ngspice on any generated layout.

``evaluate(component, testbench, pdk)`` simulates the same testbench on three
netlists of the layout (schematic, extracted devices, extracted devices +
capacitance) and reports each metric's gap between them. gf180 only so far;
see ``evaluate`` for the stages and ``extract`` for the kpex workarounds.
"""
from glayout.evaluation.cells import extract_cell_parasitics
from glayout.evaluation.config import GF180, SimConfig, sim_config
from glayout.evaluation.evaluate import VIEWS, Evaluation, EvaluationError, ViewResult, evaluate, evaluate_gds
from glayout.evaluation.extract import Extraction, ExtractionError, kpex_extract
from glayout.evaluation.lvs import LvsResult, lvs_gf180, lvs_reference, write_lvs_netlist
from glayout.evaluation.ngspice import Plot, SimulationError, read_raw, run_ngspice
from glayout.evaluation.testbench import OtaTestbench, Testbench, open_loop_metrics

__all__ = [
    "Evaluation", "EvaluationError", "Extraction", "ExtractionError", "GF180", "LvsResult", "OtaTestbench",
    "Plot", "SimConfig", "SimulationError", "Testbench", "VIEWS", "ViewResult", "evaluate", "evaluate_gds",
    "extract_cell_parasitics",
    "kpex_extract", "lvs_gf180", "lvs_reference", "open_loop_metrics", "read_raw", "run_ngspice", "sim_config",
    "write_lvs_netlist",
]

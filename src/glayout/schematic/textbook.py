"""Curated circuits with known structure, for tests and as the held-out set.

``TEXTBOOK``: the 18 topologies of the textbook sharing benchmark (gLayout
PR #127, ``circuits.py``), each an AnalogGenie entry (MIT licence) checked
against the textbook scan, with that benchmark's device names and declared
matched pairs (``matching.py``). ``TWO_STAGE_OTA``: the circuit of
``glayout.floorplan.examples``, written as a netlist.

Netlists use the AnalogGenie line format (``M1 (D G S B) nmos4``).
"""
from __future__ import annotations

from glayout.schematic.model import Schematic
from glayout.schematic.readers import parse_analoggenie

# id: (AnalogGenie id, topology, ports, netlist)
TEXTBOOK: dict[str, tuple[int, str, str, str]] = {
    "C01": (726, "basic_current_mirror", "VSS IB1 IOUT1",
            "M1 (IB1 IB1 VSS VSS) nmos4\nM2 (IOUT1 IB1 VSS VSS) nmos4"),
    "C02": (672, "cascode_current_mirror", "VSS IIN1 IOUT1",
            "M1 (IIN1 IIN1 net6 VSS) nmos4\nM2 (net6 net6 VSS VSS) nmos4\nM3 (IOUT1 IIN1 net12 VSS) nmos4\n"
            "M4 (net12 net6 VSS VSS) nmos4"),
    "C03": (840, "wilson_current_mirror", "VSS IIN1 IOUT1",
            "M1 (IIN1 Y VSS VSS) nmos4\nM2 (Y Y VSS VSS) nmos4\nM3 (IOUT1 IIN1 Y VSS) nmos4"),
    "C04": (9, "cs_amplifier_current_source_load", "VDD VSS VIN1 VOUT1 VB1",
            "M1 (VOUT1 VB1 VDD VDD) pmos4\nM2 (VOUT1 VIN1 VSS VSS) nmos4"),
    "C05": (20, "source_follower_mos_current_source", "VDD VSS VIN1 VOUT1 VB1",
            "M1 (VOUT1 VB1 VSS VSS) nmos4\nM2 (VDD VIN1 VOUT1 VSS) nmos4"),
    "C06": (38, "cascode_amplifier_cascode_load", "VDD VSS VIN1 VOUT1 VB1 VB2 VB3",
            "M1 (VOUT1 VB2 net10 VDD) pmos4\nM2 (net10 VB3 VDD VDD) pmos4\nM3 (net9 VIN1 VSS VSS) nmos4\n"
            "M4 (VOUT1 VB1 net9 VSS) nmos4"),
    "C07": (192, "diff_pair_current_source_loads", "VDD VSS VIN1 VIN2 VOUT1 VOUT2 VB1 IB1",
            "M1 (VOUT1 VIN1 IB1 VSS) nmos4\nM2 (VOUT2 VIN2 IB1 VSS) nmos4\nM3 (VOUT1 VB1 VDD VDD) pmos4\n"
            "M4 (VOUT2 VB1 VDD VDD) pmos4"),
    "C08": (157, "five_transistor_ota", "VDD VSS VIN1 VIN2 VOUT1 VB1",
            "M1 (net22 VIN1 net23 VSS) nmos4\nM2 (VOUT1 VIN2 net23 VSS) nmos4\nM3 (net23 VB1 VSS VSS) nmos4\n"
            "M4 (net22 net22 VDD VDD) pmos4\nM5 (VOUT1 net22 VDD VDD) pmos4"),
    "C09": (296, "telescopic_ota_single_ended", "VDD VSS VIN1 VIN2 VOUT1 VB1 IB1",
            "M1 (net018 VIN1 IB1 VSS) nmos4\nM2 (net016 VIN2 IB1 VSS) nmos4\nM3 (net04 VB1 net018 VSS) nmos4\n"
            "M4 (VOUT1 VB1 net016 VSS) nmos4\nM5 (net04 net04 net019 VDD) pmos4\nM6 (VOUT1 net04 net017 VDD) pmos4\n"
            "M7 (net019 net019 VDD VDD) pmos4\nM8 (net017 net019 VDD VDD) pmos4"),
    "C10": (294, "folded_cascode_ota_fully_differential", "VDD VSS VIN1 VIN2 VOUT1 VOUT2 VB1 VB2 VB3 VB4 IB1",
            "M1 (net022 VIN1 IB1 VDD) pmos4\nM2 (net018 VIN2 IB1 VDD) pmos4\nM3 (net022 VB4 VSS VSS) nmos4\n"
            "M4 (net018 VB4 VSS VSS) nmos4\nM5 (VOUT1 VB1 net022 VSS) nmos4\nM6 (VOUT2 VB1 net018 VSS) nmos4\n"
            "M7 (VOUT1 VB2 net023 VDD) pmos4\nM8 (VOUT2 VB2 net021 VDD) pmos4\nM9 (net023 VB3 VDD VDD) pmos4\n"
            "M10 (net021 VB3 VDD VDD) pmos4"),
    "C11": (705, "two_stage_miller_opamp", "VDD VSS VIN1 VIN2 VOUT1 IB1",
            "M1 (IB1 IB1 VDD VDD) pmos4\nM2 (net97 IB1 VDD VDD) pmos4\nM3 (net014 VIN1 net97 VDD) pmos4\n"
            "M4 (net016 VIN2 net97 VDD) pmos4\nM5 (net014 net014 VSS VSS) nmos4\nM6 (net016 net014 VSS VSS) nmos4\n"
            "M7 (VOUT1 IB1 VDD VDD) pmos4\nM8 (VOUT1 net016 VSS VSS) nmos4\nR1 (net016 net022) resistor\n"
            "C1 (net022 VOUT1) capacitor"),
    "C12": (195, "cmos_inverter_amplifier", "VDD VSS VIN1 VOUT1",
            "M1 (VOUT1 VIN1 VSS VSS) nmos4\nM2 (VOUT1 VIN1 VDD VDD) pmos4"),
    "C13": (709, "wide_swing_cascode_current_mirror", "VSS IIN1 IOUT1 IB1",
            "M1 (IB1 IB1 VSS VSS) nmos4\nM2 (IIN1 IB1 net15 VSS) nmos4\nM3 (net15 IIN1 VSS VSS) nmos4\n"
            "M4 (IOUT1 IB1 net14 VSS) nmos4\nM5 (net14 IIN1 VSS VSS) nmos4"),
    "C14": (936, "regulated_cascode_current_mirror", "VSS IIN1 IOUT1 IREF1",
            "M1 (IIN1 IIN1 VSS VSS) nmos4\nM2 (net11 IIN1 VSS VSS) nmos4\nM3 (IREF1 net11 VSS VSS) nmos4\n"
            "M4 (IOUT1 IREF1 net11 VSS) nmos4"),
    "C15": (235, "cs_amplifier_resistor_load", "VDD VSS VIN1 VOUT1",
            "M1 (VOUT1 VIN1 VSS VSS) nmos4\nR1 (VDD VOUT1) resistor"),
    "C16": (71, "diff_pair_resistor_loads", "VDD VSS VIN1 VIN2 VOUT1 VOUT2 IB1",
            "M1 (VOUT1 VIN1 IB1 VSS) nmos4\nM2 (VOUT2 VIN2 IB1 VSS) nmos4\nR1 (VDD VOUT1) resistor\n"
            "R2 (VDD VOUT2) resistor"),
    "C17": (220, "degenerated_diff_pair", "VDD VSS VIN1 VIN2 VOUT1 VOUT2 VB1",
            "M1 (VOUT1 VIN1 net07 VSS) nmos4\nM2 (VOUT2 VIN2 net09 VSS) nmos4\nM3 (net06 VB1 VSS VSS) nmos4\n"
            "R1 (VDD VOUT1) resistor\nR2 (VDD VOUT2) resistor\nR3 (net07 net06) resistor\nR4 (net09 net06) resistor"),
    "C18": (877, "class_ab_output_buffer", "VDD VSS VIN1 VOUT1 VB1",
            "M1 (net13 VIN1 VSS VSS) nmos4\nM2 (net9 VB1 VDD VDD) pmos4\nM3 (net9 net9 net22 VSS) nmos4\n"
            "M4 (net13 net13 net22 VDD) pmos4\nM5 (VDD net9 VOUT1 VSS) nmos4\nM6 (VSS net13 VOUT1 VDD) pmos4"),
}

MATCHED: dict[str, list[tuple[str, str]]] = {
    "C01": [("M1", "M2")], "C02": [("M2", "M4"), ("M1", "M3")], "C03": [("M1", "M2")],
    "C07": [("M1", "M2"), ("M3", "M4")], "C08": [("M1", "M2"), ("M4", "M5")],
    "C09": [("M1", "M2"), ("M3", "M4"), ("M5", "M6"), ("M7", "M8")],
    "C10": [("M1", "M2"), ("M3", "M4"), ("M5", "M6"), ("M7", "M8"), ("M9", "M10")],
    "C11": [("M3", "M4"), ("M5", "M6"), ("M1", "M2")], "C13": [("M3", "M5"), ("M2", "M4")],
    "C14": [("M1", "M2")], "C16": [("M1", "M2"), ("R1", "R2")],
    "C17": [("M1", "M2"), ("R1", "R2"), ("R3", "R4")], "C18": [("M3", "M5"), ("M4", "M6")],
}

TWO_STAGE_OTA = ("VINP VINN VOUT IBIAS VDD VSS",
                 "MA (N1 VINP VTAIL VSS) nmos4\nMB (N2 VINN VTAIL VSS) nmos4\nML1 (N1 N1 VDD VDD) pmos4\n"
                 "ML2 (N2 N1 VDD VDD) pmos4\nMR (IBIAS IBIAS VSS VSS) nmos4\nMT (VTAIL IBIAS VSS VSS) nmos4\n"
                 "MK (VOUT IBIAS VSS VSS) nmos4\nMCS (VOUT N2 VDD VDD) pmos4")


def textbook_schematic(cid: str) -> Schematic:
    """One curated circuit, with its provenance and declared matched pairs."""
    ag, topology, ports, body = TEXTBOOK[cid]
    sch = parse_analoggenie(body, cid, ports.split(), provenance={
        "dataset": "textbook_sharing", "circuit_id": cid, "topology": topology, "analoggenie_id": ag})
    if cid in MATCHED:
        sch.constraints["matched"] = [list(p) for p in MATCHED[cid]]
    return sch


def two_stage_ota_schematic() -> Schematic:
    ports, body = TWO_STAGE_OTA
    return parse_analoggenie(body, "two_stage_ota", ports.split(),
                             provenance={"dataset": "glayout", "source": "glayout.floorplan.examples"})


def curated() -> dict[str, Schematic]:
    out = {cid: textbook_schematic(cid) for cid in TEXTBOOK}
    out["two_stage_ota"] = two_stage_ota_schematic()
    return out

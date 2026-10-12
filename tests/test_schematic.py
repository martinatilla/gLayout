"""Tests for glayout.schematic (netlist model and readers). No PDK needed.

The netlists are copied from AnalogGenie entries (MIT licence): 157 (Razavi
p.171, five-transistor OTA), 840 (Gray p.272, bipolar Wilson mirror), and
excerpts of 1150, 1300 and 2200.

Run with pytest, or directly: ``python tests/test_schematic.py``.
"""
import json
import tempfile
from pathlib import Path

from glayout.schematic import (
    Device, NetlistParseError, Schematic, analoggenie_category, from_textbook_circuit, net_class,
    parse_analoggenie, parse_ports, read_analoggenie,
)

AG157 = """M4 (net23 VB1 VSS VSS) nmos4
M3 (VOUT1 VIN2 net23 VSS) nmos4
M2 (net22 VIN1 net23 VSS) nmos4
M0 (net22 net22 VDD VDD) pmos4
M1 (VOUT1 net22 VDD VDD) pmos4
"""
AG840 = """Q2 (IOUT1 IIN1 net5 0) npn
Q1 (net5 net5 VSS 0) npn
Q0 (IIN1 net5 VSS 0) npn
"""
AG2200_EXCERPT = """MM10 (net27 net12 VSS VSS) nmos4
MM2 (net26 VB1 VDD VDD) pmos4

MM2 (net1006 VB3 VSS VSS) nmos4
MM0 (net010 VB3 VSS VSS) nmos4
RR0 (VB3 net010) resistor
C0 (VOUT1 net11) capacitor
"""
AG1300_EXCERPT = """L0 (net33 net20) inductor
D0 (net36 net15) diode
M2 (net15 VCONT3 VIN1 VIN1) pmos4
"""

# PR #127 circuits.py, C08 (Circuit.to_json() form); AnalogGenie 157 with devices renamed
C08 = {
    "circuit_id": "C08", "name": "five_transistor_ota", "ports": ["VDD", "VSS", "VIN1", "VIN2", "VOUT1", "VB1"],
    "devices": [
        {"name": "M1", "kind": "nfet", "nets": ["net22", "VIN1", "net23", "VSS"]},
        {"name": "M2", "kind": "nfet", "nets": ["VOUT1", "VIN2", "net23", "VSS"]},
        {"name": "M3", "kind": "nfet", "nets": ["net23", "VB1", "VSS", "VSS"]},
        {"name": "M4", "kind": "pfet", "nets": ["net22", "net22", "VDD", "VDD"]},
        {"name": "M5", "kind": "pfet", "nets": ["VOUT1", "net22", "VDD", "VDD"]},
    ],
    "source": {"analoggenie_id": 157, "book": "Razavi", "page_ref": "171",
               "device_name_map": {"M1": "M2", "M2": "M3", "M3": "M4", "M4": "M0", "M5": "M1"}},
    "notes": [],
}


def test_five_transistor_ota():
    sch = parse_analoggenie(AG157, "ag157", ("VDD", "VSS", "VIN1", "VIN2", "VOUT1", "VB1"))
    assert [d.name for d in sch.devices] == ["M4", "M3", "M2", "M0", "M1"]
    assert sch.is_mos_only and sch.kinds() == {"nfet": 3, "pfet": 2}
    assert sch["M0"].pins == {"D": "net22", "G": "net22", "S": "VDD", "B": "VDD"}
    assert sorted(sch.net_pins()["net23"]) == [("M2", "S"), ("M3", "S"), ("M4", "D")]
    assert sch.nets()[:6] == ["VDD", "VSS", "VIN1", "VIN2", "VOUT1", "VB1"]
    assert sch.check() == [] and sch.warnings == []


def test_net_classes():
    assert [net_class(n) for n in ("VDD", "AVDD", "VSS", "GND", "0", "VB1", "IB2", "VREF1", "VCLK1", "VIN1", "net22")] == \
        ["power", "power", "ground", "ground", "ground", "bias", "bias", "bias", "clock", "signal", "signal"]


def test_bjt_keeps_substrate():
    sch = parse_analoggenie(AG840, "ag840", ("VSS", "IIN1", "IOUT1"))
    assert sch["Q1"].pins == {"C": "net5", "B": "net5", "E": "VSS", "S": "0"}
    assert not sch.is_mos_only
    assert Device("Q9", "npn", ("C", "B", "E")).pins == {"C": "C", "B": "B", "E": "E"}   # 3-terminal form


def test_repeated_names_are_renamed_and_recorded():
    sch = parse_analoggenie(AG2200_EXCERPT, "ag2200")
    assert [d.name for d in sch.devices] == ["MM10", "MM2", "MM2_2", "MM0", "RR0", "C0"]
    assert sch.provenance["renamed"] == [{"line": 4, "from": "MM2", "to": "MM2_2"}]
    assert sch["MM2_2"].pins["G"] == "VB3"
    assert sch.kinds() == {"nfet": 3, "pfet": 1, "res": 1, "cap": 1}


def test_two_terminal_kinds():
    sch = parse_analoggenie(AG1300_EXCERPT, "ag1300")
    assert sch.kinds() == {"ind": 1, "diode": 1, "pfet": 1}
    assert sch["D0"].pins == {"P": "net36", "N": "net15"}
    assert sch["M2"].pins["B"] == "VIN1"        # body on a signal net is kept as drawn


def test_composite_cell():
    sch = parse_analoggenie("X0 (IN OUT VDD VSS) INVERTER\n", "inv")
    assert sch["X0"].kind == "composite:INVERTER"
    assert sch["X0"].pins == {"A": "IN", "Q": "OUT", "VDD": "VDD", "VSS": "VSS"}


def test_strict_and_lenient():
    bad = "M0 (a b c d) nmos4\nV0 (a 0) vsource\nthis is not a device\nM1 (a b c) nmos4\n"
    for snippet in ("V0 (a 0) vsource", "this is not a device", "M1 (a b c) nmos4"):
        try:
            parse_analoggenie(snippet, "x")
        except NetlistParseError:
            pass
        else:
            raise AssertionError(f"strict parse accepted {snippet!r}")
    sch = parse_analoggenie(bad, "x", strict=False)
    assert [d.name for d in sch.devices] == ["M0", "V0"]
    assert sch["V0"].kind == "unknown:vsource" and sch["V0"].pin_names == ("P1", "P2")
    assert len(sch.warnings) == 3
    assert "device V0 has unknown kind unknown:vsource" in sch.check()


def test_check_reports_dangling_port_and_single_pin_net():
    sch = parse_analoggenie("M0 (D G VSS VSS) nmos4\n", "x", ("VDD", "VSS", "G"))
    assert sch.check() == ["port VDD connects to nothing", "net D has one pin (M0.D)"]


def test_port_file_and_folder_reader():
    assert parse_ports("---\nVDD VSS VIN1 VIN2 VOUT1 VB1\n---\n") == ("VDD", "VSS", "VIN1", "VIN2", "VOUT1", "VB1")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "Dataset" / "157"
        folder.mkdir(parents=True)
        (folder / "157.cir").write_text(AG157)
        (folder / "Port157.txt").write_text("VDD VSS VIN1 VIN2 VOUT1 VB1\n")
        (folder / "Pagenumber157.txt").write_text("171\n")
        a = read_analoggenie(folder)
        b = read_analoggenie(Path(tmp) / "Dataset", 157)
    assert a.to_json() == b.to_json()
    assert a.name == "ag157" and a.ports[0] == "VDD"
    assert a.provenance == {"dataset": "AnalogGenie", "analoggenie_id": 157,
                            "category": "book:razavi_analog_cmos", "page_ref": "171"}


def test_categories():
    assert analoggenie_category(157) == "book:razavi_analog_cmos"
    assert analoggenie_category(705) == "book:johns_martin"
    assert analoggenie_category(1035) is None            # not listed in the categorisation file
    assert analoggenie_category(2200) == "paper:ldo"
    assert analoggenie_category(9999) is None


def test_textbook_circuit_matches_its_analoggenie_entry():
    tb = from_textbook_circuit(C08, matched=[("M1", "M2"), ("M4", "M5")])
    ag = parse_analoggenie(AG157, "ag157", tb.ports)
    names = C08["source"]["device_name_map"]
    for d in tb.devices:
        assert d.kind == ag[names[d.name]].kind
        assert d.nets == ag[names[d.name]].nets, d.name
    assert tb.constraints == {"matched": [["M1", "M2"], ["M4", "M5"]]}
    assert tb.provenance["analoggenie_id"] == 157


def test_json_round_trip_and_hash():
    sch = parse_analoggenie(AG2200_EXCERPT, "ag2200", ("VDD", "VSS"))
    again = Schematic.from_json(json.loads(json.dumps(sch.to_json())))
    assert again.to_json() == sch.to_json()
    assert again.sha() == sch.sha()
    assert parse_analoggenie(AG157, "a").sha() != parse_analoggenie(AG157.replace("VB1", "VB2"), "a").sha()


def test_bad_device_and_duplicate_names_rejected():
    for build in (lambda: Device("M0", "nfet", ("a", "b", "c")),
                  lambda: Schematic("x", (), (Device("R0", "res", ("a", "b")), Device("R0", "res", ("b", "c"))))):
        try:
            build()
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an invalid device or schematic")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

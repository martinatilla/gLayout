"""Tool-free tests for glayout.evaluation (netlist transforms, rawfile reader, metrics).

The fixtures are cut from real outputs on the routed 5T OTA (gf180): glayout's
reference netlist, kpex 0.4.4's LVS and 2.5D netlists, ngspice-47 rawfiles.
Run with pytest, or directly: ``python tests/test_evaluation.py``.
"""
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("GLAYOUT_TABLES", "0")

import numpy as np  # noqa: E402

from glayout.evaluation.config import GF180  # noqa: E402
from glayout.evaluation.lvs import lvs_reference, substrate_name  # noqa: E402
from glayout.evaluation.netlist import (  # noqa: E402
    capacitance_by_net, capacitors, logical_lines, rename_nodes, sanitize_names, spice_number, subckt_ports,
    to_sim_dialect, wrapper,
)
from glayout.evaluation.ngspice import read_raw  # noqa: E402
from glayout.evaluation.testbench import OtaTestbench, open_loop_metrics  # noqa: E402

REFERENCE = """.subckt CC_DIFFPAIR VINP VINN VOUT1 VOUT2 VTAIL B
XA VOUT1 VINP VTAIL B nfet_03v3 l=0.28 w=3.0 m=4
XB VOUT2 VINN VTAIL B nfet_03v3 l=0.28 w=3.0 m=4
XDUMMY B B B B nfet_03v3 l=0.28 w=3.0 m=2
.ends CC_DIFFPAIR

.subckt OTA5T_ROUTED VINP VINN VOUT IBIAS VDD VSS
X0 VINP VINN N1 VOUT VTAIL VSS CC_DIFFPAIR
.ends OTA5T_ROUTED"""

KPEX = r"""* Extracted by KLayout with GF180MCU LVS runset
.SUBCKT ota5t_routed IBIAS VDD VINN VINP VOUT VSS
M$2 \$11 \$11 VDD VDD pfet_03v3 L=0.28U W=3U AS=2.25P AD=1.29P PS=7.5U PD=3.86U
M$12 \$11 VINP \$4 VSS nfet_03v3 L=0.28U W=3U AS=2.25P AD=1.29P
+ PS=7.5U PD=3.86U
Cext_1 \$11 \$4 1.61812f
Cext_6 \$11 VSUBS 2.77978f
Cext_30 VSS VSUBS 13.8767f
.ENDS ota5t_routed
"""

OP_RAW = """Title: * ota testbench
Date: Thu Oct  8 01:54:00  2026
Command: ngspice-47, Build Tue Sep 22 13:38:55 UTC 2026
Plotname: Operating Point
Flags: real
No. Variables: 3
No. Points: 1
Variables:
\t0\tv(vdd)\tvoltage
\t1\tv(vout)\tvoltage
\t2\ti(vsup)\tcurrent
Values:
 0\t3.300000000000000e+00
\t1.694760000000000e+00
\t-4.000000000000000e-05
"""

AC_RAW = """Title: * ota testbench
Date: Thu Oct  8 01:54:00  2026
Command: ngspice-47, Build Tue Sep 22 13:38:55 UTC 2026
Plotname: AC Analysis
Flags: complex
No. Variables: 2
No. Points: 2
Variables:
\t0\tfrequency\tfrequency grid=3
\t1\tv(vout)\tvoltage
Values:
 0\t1.000000000000000e+00,0.000000000000000e+00
\t1.788689982050752e+01,2.259700008782768e-04

 1\t1.122018454301963e+00,0.000000000000000e+00
\t1.788689980000000e+01,2.535400000000000e-04
"""


def test_logical_lines_joins_continuations():
    lines = logical_lines(KPEX)
    assert any(l.startswith("M$12") and l.endswith("PD=3.86U") for l in lines)
    assert not any(l.lstrip().startswith("+") for l in lines)


def test_subckt_ports_top_is_last_and_lookup_ignores_case():
    assert subckt_ports(REFERENCE) == ("OTA5T_ROUTED", ["VINP", "VINN", "VOUT", "IBIAS", "VDD", "VSS"])
    assert subckt_ports(REFERENCE, "cc_diffpair")[1] == ["VINP", "VINN", "VOUT1", "VOUT2", "VTAIL", "B"]
    assert subckt_ports(KPEX, "OTA5T_ROUTED")[1] == ["IBIAS", "VDD", "VINN", "VINP", "VOUT", "VSS"]


def test_gf180_dialect_rewrites_primitives_only():
    out = to_sim_dialect(REFERENCE, GF180)
    assert "MA VOUT1 VINP VTAIL B nfet_03v3 l=0.28u w=3.0u m=4" in out
    assert "MDUMMY B B B B nfet_03v3 l=0.28u w=3.0u m=2" in out
    assert "X0 VINP VINN N1 VOUT VTAIL VSS CC_DIFFPAIR" in out          # subckt instance untouched
    assert to_sim_dialect(out, GF180) == out                              # idempotent: units already there


def test_dialect_is_identity_for_subckt_models_in_microns():
    sky = replace(GF180, fet_prefix="X", si_units=False)
    assert to_sim_dialect(REFERENCE, sky) == REFERENCE


def test_sanitize_and_substrate_rename():
    text = rename_nodes(sanitize_names(KPEX), {"VSUBS": "VSS"})
    assert "\\" not in text and "VSUBS" not in text
    assert "Cext_1 _11 _4 1.61812f" in text
    assert "Cext_30 VSS VSS 13.8767f" in text                             # shorted out: harmless
    assert ".SUBCKT ota5t_routed IBIAS VDD VINN VINP VOUT VSS" in text   # header untouched
    assert "M$12 _11 VINP _4 VSS nfet_03v3 L=0.28U W=3U" in text         # instance name, params untouched


def test_wrapper_matches_pins_by_name():
    pins = ["IBIAS", "VDD", "VINN", "VINP", "VOUT", "VSS", "gf180mcu_gnd", "VTAIL"]
    ports = ("VINP", "VINN", "VOUT", "IBIAS", "VDD", "VSS")
    text = wrapper("ota5t_routed", pins, "DUT", ports, ties={"GF180MCU_GND": "VSS"})
    assert text.splitlines() == [
        ".subckt DUT VINP VINN VOUT IBIAS VDD VSS",
        "Xcell IBIAS VDD VINN VINP VOUT VSS VSS VTAIL ota5t_routed",       # VTAIL stays internal
        ".ends DUT",
    ]
    assert wrapper("x", ["vinp"], "DUT", ("VINP",)).splitlines()[1] == "Xcell VINP x"   # case-insensitive


def test_wrapper_rejects_missing_port():
    try:
        wrapper("ota5t_routed", ["VDD", "VSS"], "DUT", ("VINP", "VDD", "VSS"))
    except ValueError as exc:
        assert "VINP" in str(exc) and "labelled" in str(exc)
    else:
        raise AssertionError("missing port accepted")


def test_spice_numbers_and_capacitance_by_net():
    assert math.isclose(spice_number("1.61812f"), 1.61812e-15)
    assert math.isclose(spice_number("68.8681a"), 68.8681e-18)
    assert math.isclose(spice_number("2.19P"), 2.19e-12)
    assert math.isclose(spice_number("1meg"), 1e6)
    assert math.isclose(spice_number("2pF"), 2e-12)
    assert spice_number("4.7") == 4.7
    caps = capacitors(sanitize_names(KPEX))
    assert len(caps) == 3
    by_net = capacitance_by_net(caps)
    assert list(by_net)[0] == "VSUBS"
    assert math.isclose(by_net["_11"], (1.61812 + 2.77978) * 1e-15)


def test_lvs_reference_names_top_after_cell():
    out = lvs_reference(REFERENCE, "ota5t_routed", GF180)
    assert subckt_ports(out) == ("ota5t_routed", ["VINP", "VINN", "VOUT", "IBIAS", "VDD", "VSS"])
    assert "OTA5T_ROUTED" not in out and ".ends ota5t_routed" in out
    assert "MA VOUT1 VINP VTAIL B nfet_03v3 l=0.28u w=3.0u m=4" in out
    plain = lvs_reference(REFERENCE, "ota5t_routed", None)                # no PDK config: rename only
    assert plain == REFERENCE.replace("OTA5T_ROUTED", "ota5t_routed")


def test_substrate_name_prefers_bulk_pin():
    assert substrate_name(["VREF", "VOUT", "VSS", "B"], "gf180mcu_gnd") == "B"
    assert substrate_name(["VINP", "VINN", "VOUT", "IBIAS", "VDD", "VSS"], "gf180mcu_gnd") == "VSS"
    assert substrate_name(["A", "Y"], "gf180mcu_gnd") == "gf180mcu_gnd"


def test_read_raw_real_and_complex():
    with tempfile.TemporaryDirectory() as d:
        op_path, ac_path = Path(d) / "op.raw", Path(d) / "ac.raw"
        op_path.write_text(OP_RAW)
        ac_path.write_text(AC_RAW)
        op = read_raw(op_path)[0]
        ac = read_raw(ac_path)[0]
    assert op.name == "Operating Point"
    assert op["V(VOUT)"][0] == 1.69476 and op["i(vsup)"][0] == -4e-05
    assert ac["frequency"].real.tolist() == [1.0, 1.122018454301963]
    assert np.iscomplexobj(ac["v(vout)"]) and math.isclose(abs(ac["v(vout)"][0]), 17.886899, rel_tol=1e-6)


def _two_pole(a0, p1, p2, ppd=20):
    f = np.logspace(0, 10, 10 * ppd + 1)
    return f, a0 / ((1 + 1j * f / p1) * (1 + 1j * f / p2))


def test_open_loop_metrics_two_pole():
    a0, p1, p2 = 100.0, 1e5, 3e7
    m = open_loop_metrics(*_two_pole(a0, p1, p2))
    lo, hi = 1e5, 1e9                                            # bisect |H| = 1 analytically
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if a0 / abs((1 + 1j * mid / p1) * (1 + 1j * mid / p2)) > 1:
            lo = mid
        else:
            hi = mid
    ugf = lo
    pm = 180 - math.degrees(math.atan(ugf / p1)) - math.degrees(math.atan(ugf / p2))
    assert math.isclose(m["dc_gain_db"], 40.0, abs_tol=1e-3)
    assert math.isclose(m["ugf_hz"], ugf, rel_tol=0.01)
    assert math.isclose(m["pm_deg"], pm, abs_tol=0.5)
    assert math.isclose(m["f3db_hz"], p1, rel_tol=0.02)
    assert abs(m["phase0_deg"]) < 0.1


def test_open_loop_metrics_without_crossing():
    f, h = _two_pole(0.5, 1e5, 3e7)                              # gain below 1: no UGF, no PM
    m = open_loop_metrics(f, h)
    assert m["ugf_hz"] is None and m["pm_deg"] is None


def test_ota_deck_instantiates_ports_in_order():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        lib = root / "gf180mcuD" / "libs.tech" / "ngspice"
        lib.mkdir(parents=True)
        for name in ("design.ngspice", "sm141064.ngspice"):
            (lib / name).write_text("")
        old = os.environ.get("PDK_ROOT")
        os.environ["PDK_ROOT"] = str(root)
        try:
            deck = OtaTestbench(ibias=20e-6, cl=1e-12).deck("DUT", "dut.spice", GF180)
            sink = OtaTestbench(ibias=-5e-6).deck("DUT", "dut.spice", GF180)
        finally:
            if old is None:
                os.environ.pop("PDK_ROOT")
            else:
                os.environ["PDK_ROOT"] = old
    lines = deck.splitlines()
    assert "XDUT VINP VINN VOUT IBIAS VDD VSS DUT" in lines
    assert f".lib {root}/gf180mcuD/libs.tech/ngspice/sm141064.ngspice typical" in lines
    assert "IB VDD IBIAS DC 2e-05" in lines and "VCM vcm 0 DC 1.65" in lines
    assert "IB IBIAS VSS DC 5e-06" in sink.splitlines()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

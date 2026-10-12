"""Tests for glayout.schematic.annotate. No PDK needed.

Ground truth: the 18 curated circuits of the textbook sharing benchmark
(gLayout PR #127, ``circuits.py``) and their 30 declared matched pairs
(``matching.MATCHED``), copied below; the hand-written two-stage OTA of
``glayout.floorplan.examples``; and two AnalogGenie entries (MIT licence),
1150 (LC VCO) and 2200 (LDO).

Run with pytest, or directly: ``python tests/test_annotate.py``.
"""
import json
import re

from glayout.schematic import parse_analoggenie
from glayout.schematic.annotate import annotate

TEXTBOOK = {
    "C01": ("VSS IB1 IOUT1",
            "M1 (IB1 IB1 VSS VSS) nmos4\nM2 (IOUT1 IB1 VSS VSS) nmos4"),
    "C02": ("VSS IIN1 IOUT1",
            "M1 (IIN1 IIN1 net6 VSS) nmos4\nM2 (net6 net6 VSS VSS) nmos4\nM3 (IOUT1 IIN1 net12 VSS) nmos4\nM4 (net12 net6 VSS VSS) nmos4"),
    "C03": ("VSS IIN1 IOUT1",
            "M1 (IIN1 Y VSS VSS) nmos4\nM2 (Y Y VSS VSS) nmos4\nM3 (IOUT1 IIN1 Y VSS) nmos4"),
    "C04": ("VDD VSS VIN1 VOUT1 VB1",
            "M1 (VOUT1 VB1 VDD VDD) pmos4\nM2 (VOUT1 VIN1 VSS VSS) nmos4"),
    "C05": ("VDD VSS VIN1 VOUT1 VB1",
            "M1 (VOUT1 VB1 VSS VSS) nmos4\nM2 (VDD VIN1 VOUT1 VSS) nmos4"),
    "C06": ("VDD VSS VIN1 VOUT1 VB1 VB2 VB3",
            "M1 (VOUT1 VB2 net10 VDD) pmos4\nM2 (net10 VB3 VDD VDD) pmos4\nM3 (net9 VIN1 VSS VSS) nmos4\nM4 (VOUT1 VB1 net9 VSS) nmos4"),
    "C07": ("VDD VSS VIN1 VIN2 VOUT1 VOUT2 VB1 IB1",
            "M1 (VOUT1 VIN1 IB1 VSS) nmos4\nM2 (VOUT2 VIN2 IB1 VSS) nmos4\nM3 (VOUT1 VB1 VDD VDD) pmos4\nM4 (VOUT2 VB1 VDD VDD) pmos4"),
    "C08": ("VDD VSS VIN1 VIN2 VOUT1 VB1",
            "M1 (net22 VIN1 net23 VSS) nmos4\nM2 (VOUT1 VIN2 net23 VSS) nmos4\nM3 (net23 VB1 VSS VSS) nmos4\nM4 (net22 net22 VDD VDD) pmos4\nM5 (VOUT1 net22 VDD VDD) pmos4"),
    "C09": ("VDD VSS VIN1 VIN2 VOUT1 VB1 IB1",
            "M1 (net018 VIN1 IB1 VSS) nmos4\nM2 (net016 VIN2 IB1 VSS) nmos4\nM3 (net04 VB1 net018 VSS) nmos4\nM4 (VOUT1 VB1 net016 VSS) nmos4\nM5 (net04 net04 net019 VDD) pmos4\nM6 (VOUT1 net04 net017 VDD) pmos4\nM7 (net019 net019 VDD VDD) pmos4\nM8 (net017 net019 VDD VDD) pmos4"),
    "C10": ("VDD VSS VIN1 VIN2 VOUT1 VOUT2 VB1 VB2 VB3 VB4 IB1",
            "M1 (net022 VIN1 IB1 VDD) pmos4\nM2 (net018 VIN2 IB1 VDD) pmos4\nM3 (net022 VB4 VSS VSS) nmos4\nM4 (net018 VB4 VSS VSS) nmos4\nM5 (VOUT1 VB1 net022 VSS) nmos4\nM6 (VOUT2 VB1 net018 VSS) nmos4\nM7 (VOUT1 VB2 net023 VDD) pmos4\nM8 (VOUT2 VB2 net021 VDD) pmos4\nM9 (net023 VB3 VDD VDD) pmos4\nM10 (net021 VB3 VDD VDD) pmos4"),
    "C11": ("VDD VSS VIN1 VIN2 VOUT1 IB1",
            "M1 (IB1 IB1 VDD VDD) pmos4\nM2 (net97 IB1 VDD VDD) pmos4\nM3 (net014 VIN1 net97 VDD) pmos4\nM4 (net016 VIN2 net97 VDD) pmos4\nM5 (net014 net014 VSS VSS) nmos4\nM6 (net016 net014 VSS VSS) nmos4\nM7 (VOUT1 IB1 VDD VDD) pmos4\nM8 (VOUT1 net016 VSS VSS) nmos4\nR1 (net016 net022) resistor\nC1 (net022 VOUT1) capacitor"),
    "C12": ("VDD VSS VIN1 VOUT1",
            "M1 (VOUT1 VIN1 VSS VSS) nmos4\nM2 (VOUT1 VIN1 VDD VDD) pmos4"),
    "C13": ("VSS IIN1 IOUT1 IB1",
            "M1 (IB1 IB1 VSS VSS) nmos4\nM2 (IIN1 IB1 net15 VSS) nmos4\nM3 (net15 IIN1 VSS VSS) nmos4\nM4 (IOUT1 IB1 net14 VSS) nmos4\nM5 (net14 IIN1 VSS VSS) nmos4"),
    "C14": ("VSS IIN1 IOUT1 IREF1",
            "M1 (IIN1 IIN1 VSS VSS) nmos4\nM2 (net11 IIN1 VSS VSS) nmos4\nM3 (IREF1 net11 VSS VSS) nmos4\nM4 (IOUT1 IREF1 net11 VSS) nmos4"),
    "C15": ("VDD VSS VIN1 VOUT1",
            "M1 (VOUT1 VIN1 VSS VSS) nmos4\nR1 (VDD VOUT1) resistor"),
    "C16": ("VDD VSS VIN1 VIN2 VOUT1 VOUT2 IB1",
            "M1 (VOUT1 VIN1 IB1 VSS) nmos4\nM2 (VOUT2 VIN2 IB1 VSS) nmos4\nR1 (VDD VOUT1) resistor\nR2 (VDD VOUT2) resistor"),
    "C17": ("VDD VSS VIN1 VIN2 VOUT1 VOUT2 VB1",
            "M1 (VOUT1 VIN1 net07 VSS) nmos4\nM2 (VOUT2 VIN2 net09 VSS) nmos4\nM3 (net06 VB1 VSS VSS) nmos4\nR1 (VDD VOUT1) resistor\nR2 (VDD VOUT2) resistor\nR3 (net07 net06) resistor\nR4 (net09 net06) resistor"),
    "C18": ("VDD VSS VIN1 VOUT1 VB1",
            "M1 (net13 VIN1 VSS VSS) nmos4\nM2 (net9 VB1 VDD VDD) pmos4\nM3 (net9 net9 net22 VSS) nmos4\nM4 (net13 net13 net22 VDD) pmos4\nM5 (VDD net9 VOUT1 VSS) nmos4\nM6 (VSS net13 VOUT1 VDD) pmos4"),
}
MATCHED = {"C01": [["M1", "M2"]], "C02": [["M2", "M4"], ["M1", "M3"]], "C03": [["M1", "M2"]],
           "C07": [["M1", "M2"], ["M3", "M4"]], "C08": [["M1", "M2"], ["M4", "M5"]],
           "C09": [["M1", "M2"], ["M3", "M4"], ["M5", "M6"], ["M7", "M8"]],
           "C10": [["M1", "M2"], ["M3", "M4"], ["M5", "M6"], ["M7", "M8"], ["M9", "M10"]],
           "C11": [["M3", "M4"], ["M5", "M6"], ["M1", "M2"]], "C13": [["M3", "M5"], ["M2", "M4"]],
           "C14": [["M1", "M2"]], "C16": [["M1", "M2"], ["R1", "R2"]],
           "C17": [["M1", "M2"], ["R1", "R2"], ["R3", "R4"]], "C18": [["M3", "M5"], ["M4", "M6"]]}

# pairs found beyond the declared ones, each checked by hand
EXTRA = {
    "C11": [["M1", "M7"], ["M2", "M7"]],   # M7 is a third output of the IB1 mirror
    "C13": [["M1", "M2"], ["M1", "M4"]],   # gate_diode: the scaled bias diode M1 and the cascodes on its gate
}

# blocks: kind(devices); '*' = self-symmetric
BLOCKS = {
    "C01": "CM(M1 M2)",
    "C02": "CM(M2 M4) CM(M1 M3)",
    "C03": "CM(M1 M2) CASC(M3)",
    "C04": "BIAS(M1) CS(M2)",
    "C05": "BIAS(M1) SF(M2)",
    "C06": "CASC(M1 M2) CASC(M3 M4)",
    "C07": "DP(M1 M2)* CMB(M3 M4)*",
    "C08": "DP(M1 M2)* CM(M4 M5)* BIAS(M3)",
    "C09": "DP(M1 M2)* CM(M7 M8)* CM(M5 M6)* CASC(M3 M4)*",
    "C10": "DP(M1 M2)* CMB(M3 M4)* CMB(M9 M10)* CASC(M5 M6)* CASC(M7 M8)*",
    "C11": "DP(M3 M4)* BIAS(M1 M2 M7) CM(M5 M6)* CS(M8) RES(R1) CAP(C1)",
    "C12": "INV(M1) INV(M2)",
    "C13": "CMB(M3 M5) CASC(M2 M4) BIAS(M1)",
    "C14": "CM(M1 M2) CS(M3) CASC(M4)",
    "C15": "CS(M1) RES(R1)",
    "C16": "DP(M1 M2)* RES(R1) RES(R2)",
    "C17": "DP(M1 M2)* BIAS(M3) RES(R1) RES(R2) RES(R3) RES(R4)",
    "C18": "CM(M3 M5) CM(M4 M6) CS(M1) BIAS(M2)",
}
SYMMETRIC_PAIRS = {"C16": [("R1", "R2")], "C17": [("R3", "R4"), ("R1", "R2")]}

# glayout.floorplan.examples two-stage OTA, written as a netlist
TWO_STAGE_OTA = """MA (N1 VINP VTAIL VSS) nmos4
MB (N2 VINN VTAIL VSS) nmos4
ML1 (N1 N1 VDD VDD) pmos4
ML2 (N2 N1 VDD VDD) pmos4
MR (IBIAS IBIAS VSS VSS) nmos4
MT (VTAIL IBIAS VSS VSS) nmos4
MK (VOUT IBIAS VSS VSS) nmos4
MCS (VOUT N2 VDD VDD) pmos4
"""
AG1150 = """M1 (IB1 IB1 VDD VDD) pmos4
M0 (net12 IB1 VDD VDD) pmos4
L2 (net22 net10) inductor
L1 (net12 net10) inductor
L0 (net12 net22) inductor
M7 (IB3 net10 VSS VSS) nmos4
M6 (IB2 net22 VSS VSS) nmos4
M5 (net22 net10 VSS VSS) nmos4
M4 (net10 net22 VSS VSS) nmos4
M3 (VB1 net22 VB1 VSS) nmos4
M2 (VB1 net10 VB1 VSS) nmos4
"""
AG2200 = """MM10 (net27 net12 VSS VSS) nmos4
MM9 (net21 net12 VSS VSS) nmos4
MM8 (net12 net1003 net27 VSS) nmos4
MM7 (net10 net1003 net21 VSS) nmos4
MM3 (net32 VREF1 net10 VSS) nmos4
MM0 (net26 net7 net10 VSS) nmos4
MM4 (net10 net1001 VSS VSS) nmos4
MM6 (net12 VB2 net26 VDD) pmos4
MM5 (net10 VB2 net32 VDD) pmos4
MM1 (net32 VB1 VDD VDD) pmos4
MM2 (net26 VB1 VDD VDD) pmos4
MM15 (net1003 net1003 net1001 VSS) nmos4
MM13 (net1001 net1001 VSS VSS) nmos4
MM14 (net1003 net1006 VDD VDD) pmos4
MM2 (net1006 VB3 VSS VSS) nmos4
MM0 (net010 VB3 VSS VSS) nmos4
MM3 (net1006 net1006 VDD VDD) pmos4
MM1 (VB3 net1006 VDD VDD) pmos4
RR0 (VB3 net010) resistor
M0 (VIN1 net10 VOUT1 VOUT1) nmos4
R2 (net11 VSS) resistor
R1 (net7 VSS) resistor
R0 (VOUT1 net7) resistor
C0 (VOUT1 net11) capacitor
"""


def _schematic(cid):
    ports, body = TEXTBOOK[cid]
    return parse_analoggenie(body, cid, ports.split())


def _pairs(pairs):
    return {frozenset(p) for p in pairs}


def _blocks_str(ann):
    return " ".join(f"{b.kind}({' '.join(b.devices)}){'*' if b.name in ann.self_symmetric else ''}" for b in ann.blocks)


def _check_invariants(sch, ann):
    seen = [d for b in ann.blocks for d in b.devices]
    assert sorted(seen) == sorted(d.name for d in sch.devices), "every device in exactly one block"
    for b in ann.blocks:
        devs = [sch[n] for n in b.devices]
        assert len({d.kind for d in devs}) == 1, f"{b.name} mixes device kinds"
        if devs[0].is_mos:
            assert len({d.net("B") for d in devs}) == 1, f"{b.name} mixes bodies"
    in_pairs = [n for p in ann.symmetric_pairs for n in p]
    assert len(in_pairs) == len(set(in_pairs)) and not set(in_pairs) & set(ann.self_symmetric)
    json.dumps(ann.to_json())


def test_declared_matched_pairs_are_all_recovered():
    total = 0
    for cid in TEXTBOOK:
        ann = annotate(_schematic(cid))
        truth = _pairs(MATCHED.get(cid, []))
        got = ann.matched_pairs()
        assert truth <= got, f"{cid}: missed {truth - got}"
        assert got - truth == _pairs(EXTRA.get(cid, [])), f"{cid}: unexpected extra pairs {got - truth - _pairs(EXTRA.get(cid, []))}"
        total += len(truth)
    assert total == 30


def test_blocks_of_the_textbook_circuits():
    for cid in TEXTBOOK:
        sch = _schematic(cid)
        ann = annotate(sch)
        _check_invariants(sch, ann)
        assert sorted(re.findall(r"\w+\([^)]*\)\*?", _blocks_str(ann))) == \
            sorted(re.findall(r"\w+\([^)]*\)\*?", BLOCKS[cid])), f"{cid}: {_blocks_str(ann)}"
        by_name = {b.name: b for b in ann.blocks}
        pairs = {frozenset(by_name[n].devices[0] for n in p) for p in ann.symmetric_pairs}
        assert pairs == _pairs(SYMMETRIC_PAIRS.get(cid, [])), cid


def test_differential_pairs_are_symmetric_mirrors_alone_are_not():
    ann = annotate(_schematic("C08"))
    assert ann.matched_pairs(symmetric_only=True) == _pairs([["M1", "M2"], ["M4", "M5"]])
    assert annotate(_schematic("C01")).matched_pairs(symmetric_only=True) == set()


def test_degenerated_pair_and_its_resistors():
    ann = annotate(_schematic("C17"))
    assert [s.kind for s in ann.structures if "M1" in s.devices] == ["degenerated_dp"]
    assert ann.matched_pairs(symmetric_only=True) == _pairs([["M1", "M2"], ["R1", "R2"], ["R3", "R4"]])


def test_two_stage_ota_matches_the_hand_written_floorplan_circuit():
    # examples.two_stage_ota_circuit: DP, LOAD (CM), BIAS (3-device bank), CS; DP and LOAD self-symmetric
    sch = parse_analoggenie(TWO_STAGE_OTA, "two_stage_ota", ["VINP", "VINN", "VOUT", "IBIAS", "VDD", "VSS"])
    ann = annotate(sch)
    _check_invariants(sch, ann)
    assert _blocks_str(ann) == "DP(MA MB)* CM(ML1 ML2)* BIAS(MR MT MK) CS(MCS)"
    assert ann.symmetric_pairs == []


def test_lc_vco_cross_coupled_pair_and_tank():
    sch = parse_analoggenie(AG1150, "ag1150", ["VDD", "VSS", "IB1", "IB2", "IB3", "VB1"])
    ann = annotate(sch)
    _check_invariants(sch, ann)
    assert [s.kind for s in ann.structures if "M4" in s.devices] == ["ccp"]
    assert ann.matched_pairs(symmetric_only=True) == _pairs([["M4", "M5"], ["L0", "L1"]])
    # M5 and M7 share only the tank node as their gate: not a mirror
    assert frozenset(("M5", "M7")) not in ann.matched_pairs()
    assert ann.matched_pairs() == _pairs([["M4", "M5"], ["L0", "L1"], ["M0", "M1"]])


def test_ldo_low_voltage_cascode_mirror():
    sch = parse_analoggenie(AG2200, "ag2200", ["VDD", "VSS", "VIN1", "VOUT1", "VREF1", "VB1", "VB2"])
    ann = annotate(sch)
    _check_invariants(sch, ann)
    # MM10/MM9 has no diode-connected member: the diode runs through the cascode MM8
    assert frozenset(("MM10", "MM9")) in ann.matched_pairs(symmetric_only=True)
    assert [s.devices for s in ann.structures if s.kind == "dp"] == [("MM3", "MM0")]


def test_transmission_gate():
    sch = parse_analoggenie("MN (B CLK A VSS) nmos4\nMP (B CLKB A VDD) pmos4\n", "tg", ["A", "B", "CLK", "CLKB", "VDD", "VSS"])
    ann = annotate(sch)
    assert [s.kind for s in ann.structures] == ["tg"]
    assert _blocks_str(ann) == "TG(MN) TG(MP)"


def test_random_netlists_keep_the_block_invariants():
    import random
    rng = random.Random(0)
    nets = ["VDD", "VSS", "VB1", "VIN1", "VIN2", "VOUT1"] + [f"n{i}" for i in range(8)]
    kinds = ["nmos4"] * 4 + ["pmos4"] * 4 + ["resistor", "capacitor", "inductor", "diode"]
    for case in range(300):
        lines = []
        for i in range(rng.randint(1, 14)):
            k = rng.choice(kinds)
            n = 4 if k.endswith("mos4") else 2
            body = [] if n == 2 else [rng.choice(["VDD", "VSS", "n0"])]
            lines.append(f"X{i} ({' '.join([rng.choice(nets) for _ in range(n - len(body))] + body)}) {k}")
        sch = parse_analoggenie("\n".join(lines), f"fuzz{case}", ["VDD", "VSS", "VIN1", "VOUT1"])
        ann = annotate(sch)
        _check_invariants(sch, ann)
        for m in ann.matched:
            assert len(set(m.devices)) == len(m.devices) >= 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

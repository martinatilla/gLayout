"""Tests for glayout.schematic.annotate. No PDK needed.

Ground truth: the 18 curated circuits of the textbook sharing benchmark
(gLayout PR #127) and their 30 declared matched pairs, and the two-stage OTA
of ``glayout.floorplan.examples`` (all in ``glayout.schematic.textbook``);
and two AnalogGenie entries (MIT licence), 1150 (LC VCO) and 2200 (LDO).

Run with pytest, or directly: ``python tests/test_annotate.py``.
"""
import json
import re

from glayout.schematic import parse_analoggenie
from glayout.schematic.annotate import annotate
from glayout.schematic.textbook import MATCHED, TEXTBOOK, textbook_schematic, two_stage_ota_schematic

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
    return textbook_schematic(cid)


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
    sch = two_stage_ota_schematic()
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


def test_matching_classes():
    c08 = annotate(_schematic("C08"))
    assert {m.devices: m.precision for m in c08.matched} == {("M1", "M2"): "precise", ("M4", "M5"): "moderate"}
    assert {b.name: b.matching for b in c08.blocks} == {"DP1": "precise", "CM1": "moderate", "BIAS1": "none"}
    c11 = annotate(_schematic("C11"))
    assert {m.devices: m.precision for m in c11.matched}[("M1", "M2", "M7")] == "minimal"      # bias bank
    c13 = {m.devices: m.precision for m in annotate(_schematic("C13")).matched}
    assert c13[("M3", "M5")] == c13[("M2", "M4")] == "moderate" and c13[("M1", "M2")] == "minimal"
    c16 = {m.devices: m.precision for m in annotate(_schematic("C16")).matched}
    assert c16 == {("M1", "M2"): "precise", ("R1", "R2"): "moderate"}                           # passive loads


def test_net_sensitivity():
    assert annotate(_schematic("C08")).net_sensitivity == {
        "VDD": "supply", "VSS": "supply", "VIN1": "sensitive", "VIN2": "sensitive", "VOUT1": "sensitive",
        "VB1": "other", "net22": "other", "net23": "other"}        # net22 is held by the diode, net23 by sources
    c13 = annotate(_schematic("C13")).net_sensitivity
    assert c13["IIN1"] == "other" and c13["IOUT1"] == "sensitive"  # IIN1 is held through the cascode diode
    sens = annotate(parse_analoggenie("MN (A VCLK B VSS) nmos4\nMI1 (CLKB VCLK VSS VSS) nmos4\n"
                                      "MI2 (CLKB VCLK VDD VDD) pmos4\nMP (A CLKB B VDD) pmos4\n", "tg",
                                      ["A", "B", "VCLK", "VDD", "VSS"])).net_sensitivity
    assert sens["VCLK"] == "noisy" and sens["CLKB"] == "noisy"      # an inverter of a clock is a clock


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

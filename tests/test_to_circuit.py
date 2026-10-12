"""Tests for glayout.schematic.to_circuit. Needs the gf180 PDK (builds cells; about a minute).

Run with pytest, or directly: ``python tests/test_to_circuit.py``.
"""
import os

os.environ.setdefault("GLAYOUT_TABLES", "0")

from glayout import gf180 as pdk  # noqa: E402
from glayout.schematic import (  # noqa: E402
    UnsupportedCircuit, annotate, parse_analoggenie, pdk_limits, schematic_to_circuit, size_schematic,
    textbook_schematic, two_stage_ota_schematic,
)
from glayout.schematic.to_circuit import BLOCK_CACHE  # noqa: E402

# glayout.floorplan.examples.two_stage_ota_circuit: each instance's circuit nets
HAND_WRITTEN_OTA = {
    "DP": {"VINP", "VINN", "N1", "N2", "VTAIL", "VSS"},
    "CM": {"N1", "N2", "VDD"},
    "BIAS": {"IBIAS", "VTAIL", "VOUT", "VSS"},
    "CS": {"N2", "VOUT", "VDD"},
}


def test_five_transistor_ota_circuit():
    sch = textbook_schematic("C08")
    c = schematic_to_circuit(pdk, sch)
    assert [(i.name, i.block.kind) for i in c.instances] == [("DP1", "DP"), ("CM1", "CM"), ("BIAS1", "BIAS")]
    assert c.self_symmetric == ["DP1", "CM1"] and c.symmetric_pairs == []
    assert c.ports == list(sch.ports) and c.schematic is sch and c.sizing is not None
    dp = c["DP1"]
    assert set(dp.nets.values()) == {"net22", "VIN1", "net23", "VOUT1", "VIN2", "VSS"}
    assert dp.nets["B"] == "VSS"
    blocks = {b.name: b for b in c.annotation.blocks}
    for inst in c.instances:
        assert len(inst.block.variants) == 3
        assert set(inst.block.nets) == set(inst.nets)              # every pin is mapped
        size = c.sizing.blocks[inst.name]
        for v in inst.block.variants:                              # every shape keeps each device's total W
            for spec, dev in zip(v.params["devices"], blocks[inst.name].devices):
                assert abs(spec["fingers"] * v.params["width"] - size.total_width(dev)) < 0.02


def test_internal_nets_get_no_pin():
    c = schematic_to_circuit(pdk, textbook_schematic("C06"))    # two cascode stacks; net9 and net10 stay inside
    for inst in c.instances:
        assert "net9" not in inst.nets.values() and "net10" not in inst.nets.values(), inst.nets


def test_cache_reuses_cells_and_instances_stay_separate():
    sch = textbook_schematic("C08")
    ann = annotate(sch)
    sz = size_schematic(sch, ann, **pdk_limits(pdk), seed=0)
    a = schematic_to_circuit(pdk, sch, ann, sz)
    n = len(BLOCK_CACHE)
    b = schematic_to_circuit(pdk, sch, ann, sz)
    assert len(BLOCK_CACHE) == n
    assert a["DP1"].block is not b["DP1"].block                     # own padding per instance
    assert a["DP1"].block.variants[0] is not b["DP1"].block.variants[0]
    assert a["DP1"].block.variants[0].component is b["DP1"].block.variants[0].component


def test_two_stage_ota_matches_the_hand_written_circuit():
    c = schematic_to_circuit(pdk, two_stage_ota_schematic())
    got = {i.block.kind: set(i.nets.values()) for i in c.instances}
    assert got == HAND_WRITTEN_OTA
    assert sorted(c[n].block.kind for n in c.self_symmetric) == ["CM", "DP"]


def test_unsupported_circuits_say_why():
    for sch, code in ((textbook_schematic("C15"), "non_mos"),
                      (parse_analoggenie("M0 (VOUT1 VIN1 VOUT1 VOUT1) nmos4\nM1 (VOUT1 VB1 VSS VSS) nmos4", "x",
                                         ["VIN1", "VOUT1", "VB1", "VSS"]), "nfet_body")):
        try:
            schematic_to_circuit(pdk, sch)
        except UnsupportedCircuit as e:
            assert e.code == code
        else:
            raise AssertionError(f"{sch.name} accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

"""Tests for glayout.schematic.sizing (synthetic, layout-only sizes). No PDK needed.

Run with pytest, or directly: ``python tests/test_sizing.py``.
"""
import json

from glayout.schematic import Sizing, annotate, curated, size_schematic, textbook_schematic
from glayout.schematic.annotate import AnnotatedBlock, Annotation

LMIN, WMIN = 0.28, 0.5


def _sized(cid, seed=0, **kw):
    sch = textbook_schematic(cid) if cid.startswith("C") else curated()[cid]
    ann = annotate(sch)
    return sch, ann, size_schematic(sch, ann, LMIN, WMIN, seed=seed, **kw)


def test_reproducible_per_seed_and_varies_across_seeds():
    a = _sized("C10", seed=3)[2].to_json()
    b = _sized("C10", seed=3)[2].to_json()
    assert a == b
    assert any(_sized("C10", seed=s)[2].to_json() != a for s in range(4, 10))


def test_every_mos_block_sized_and_passives_skipped():
    sch, ann, sz = _sized("C11")
    assert sorted(sz.blocks) == sorted(b.name for b in ann.blocks if b.device_kind in ("nfet", "pfet"))
    assert sorted(sz.skipped) == ["CAP1", "RES1"]


def test_splits_keep_width_and_respect_limits():
    for cid in ("C01", "C08", "C09", "C10", "C11", "two_stage_ota"):
        for seed in range(5):
            sch, ann, sz = _sized(cid, seed, max_fingers=24)
            for name, b in sz.blocks.items():
                assert b.splits, name
                for k, w in b.variants():
                    assert w >= WMIN - 1e-9, (cid, name, k, w)
                    assert sum(b.base.values()) * k <= 24, (cid, name, k)
                    for d, n in b.base.items():
                        assert abs(n * k * w - b.total_width(d)) < 1e-6      # same total W in every shape
                assert b.length in {round(m * LMIN, 6) for m in (1, 2, 4)}


def test_matched_devices_get_equal_sizes():
    for cid in ("C07", "C08", "C09", "C10", "two_stage_ota"):
        for seed in range(5):
            sch, ann, sz = _sized(cid, seed)
            for b in ann.blocks:
                if b.name in ann.self_symmetric:
                    assert len(set(sz.blocks[b.name].base.values())) == 1, (cid, b.name)


def test_mirror_outputs_may_be_ratioed():
    seen = set()
    for seed in range(20):
        sch, ann, sz = _sized("C01", seed)
        seen.add(sz.blocks["CM1"].base["M2"])
        assert sz.blocks["CM1"].base["M1"] == 1                 # the diode stays the unit
    assert seen == {1, 2}


def test_symmetric_block_pair_is_sized_alike():
    sch = textbook_schematic("C08")
    ann = Annotation("pair", [AnnotatedBlock("X1", "CS", ("M1",), "nfet", "VSS"),
                              AnnotatedBlock("X2", "CS", ("M2",), "nfet", "VSS")], [], [],
                     symmetric_pairs=[("X1", "X2")])
    for seed in range(10):
        sz = size_schematic(sch, ann, LMIN, WMIN, seed=seed)
        a, b = sz.blocks["X1"], sz.blocks["X2"]
        assert (a.unit, a.length, a.splits, a.base["M1"]) == (b.unit, b.length, b.splits, b.base["M2"])


def test_finger_cap_falls_back_to_fewest_fingers():
    sch, ann, sz = _sized("C11", max_fingers=4)
    bank = sz.blocks["BIAS1"]                  # three devices, so k=2 would already draw at least 6 fingers
    assert bank.splits == (2,)


def test_json_round_trip():
    sz = _sized("C09", seed=2)[2]
    again = Sizing.from_json(json.loads(json.dumps(sz.to_json())))
    assert again.to_json() == sz.to_json()
    assert sz.device("M1") == (sz.blocks["DP1"].unit, sz.blocks["DP1"].length)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

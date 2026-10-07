"""PDK-free tests for glayout.floorplan.router (unit grid: pitch 1, wire/pad/spacing 0.5).

Run with pytest, or directly: ``python tests/test_router.py``.
"""
import os

os.environ.setdefault("GLAYOUT_TABLES", "0")

from glayout.floorplan.router import (  # noqa: E402
    AccessOption, RouterPin, RouterTech, RoutingProblem, _f, _too_close, route,
)


def _pin(net, owner, cell, side, x, y):
    return RouterPin(net, owner, cell, [AccessOption(side, x, y, "met2", 0.5)])


def _two_cells(extra_cells=(), width=20.0, height=20.0):
    cells = [(2.0, 8.0, 4.0, 12.0), (16.0, 8.0, 18.0, 12.0), *extra_cells]
    nets = {"N": [_pin("N", "A.N", 0, "E", 4.0, 10.0), _pin("N", "B.N", 1, "W", 16.0, 10.0)]}
    return RoutingProblem(width, height, cells, nets)


def _shapes_ok(rr):
    """No two nets' drawn landing shapes break spacing (wires are legal by pitch)."""
    lands = [l for ls in rr.landings.values() for l in ls]
    for i, a in enumerate(lands):
        for b in lands[i + 1:]:
            if a.pin.net == b.pin.net:
                continue
            for la, ra in a.shapes:
                for lb, rb in b.shapes:
                    if la == lb and _too_close(ra, rb, 0.5):
                        return False
    return True


def test_straight_connection():
    rr = route(_two_cells(), RouterTech.abstract())
    assert rr.success and rr.iterations == 1
    assert 12.0 <= rr.wirelength <= 16.0


def test_detour_around_obstacle():
    wall = (9.0, 0.0, 11.0, 17.0)
    rr = route(_two_cells([wall]), RouterTech.abstract())
    assert rr.success
    assert rr.wirelength > 20.0          # had to go over the wall


def test_unroutable_is_reported():
    wall = (9.0, 0.0, 11.0, 20.0)
    rr = route(_two_cells([wall]), RouterTech.abstract())
    assert not rr.success and rr.failed_nets == ["N"]


def test_single_track_gap_fails_two_nets_but_routes_one():
    # walls leave exactly one legal track (y = 9.5) through x = 9..11
    walls = [(9.0, 0.0, 11.0, 8.75), (9.0, 10.25, 11.0, 20.0)]
    cells = [(2.0, 8.0, 4.0, 12.0), (16.0, 8.0, 18.0, 12.0), *walls]
    one = {"N": [_pin("N", "A.N", 0, "E", 4.0, 10.0), _pin("N", "B.N", 1, "W", 16.0, 10.0)]}
    rr1 = route(RoutingProblem(20, 20, cells, one), RouterTech.abstract())
    assert rr1.success
    two = {"N": [_pin("N", "A.N", 0, "E", 4.0, 10.0), _pin("N", "B.N", 1, "W", 16.0, 10.0)],
           "M": [_pin("M", "A.M", 0, "E", 4.0, 9.0), _pin("M", "B.M", 1, "W", 16.0, 9.0)]}
    rr2 = route(RoutingProblem(20, 20, cells, two), RouterTech.abstract(max_iterations=6))
    assert not rr2.success and rr2.iterations == 6 and rr2.conflicts > 0


def test_rip_up_and_reroute_resolves_shared_channel():
    # three-track gap: two nets fit once they negotiate
    walls = [(9.0, 0.0, 11.0, 8.0), (9.0, 12.0, 11.0, 20.0)]
    cells = [(2.0, 8.0, 4.0, 12.0), (16.0, 8.0, 18.0, 12.0), *walls]
    nets = {"N": [_pin("N", "A.N", 0, "E", 4.0, 10.0), _pin("N", "B.N", 1, "W", 16.0, 9.0)],
            "M": [_pin("M", "A.M", 0, "E", 4.0, 9.0), _pin("M", "B.M", 1, "W", 16.0, 10.0)]}
    rr = route(RoutingProblem(20, 20, cells, nets), RouterTech.abstract())
    assert rr.success, rr.summary()
    assert _shapes_ok(rr)


def test_close_pins_on_one_side_are_staggered():
    cells = [(2.0, 4.0, 4.0, 16.0), (16.0, 4.0, 18.0, 16.0)]
    nets = {n: [_pin(n, f"A.{n}", 0, "E", 4.0, y), _pin(n, f"B.{n}", 1, "W", 16.0, y)]
            for n, y in (("P", 9.0), ("Q", 10.0), ("R", 11.0))}   # legal rail pitch: pad + spacing
    rr = route(RoutingProblem(20, 20, cells, nets), RouterTech.abstract())
    assert rr.success
    assert _shapes_ok(rr)
    xs = sorted(l.x for ls in rr.landings.values() for l in ls if l.option.side == "E")
    assert len(set(xs)) > 1              # could not all land on the same column


def test_dynamic_weighting_eq5():
    assert _f(1.0, 3.0, 3.0) == 4.0                 # g < h: plain A*
    assert _f(3.0, 1.0, 3.0) == (3.0 + 5.0) / 3.0   # g >= h: (g + (2κ-1)h)/κ


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")

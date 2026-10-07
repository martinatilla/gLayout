"""Cells built from one common-centroid, diffusion-shared finger row.

``finger_sequence_cell`` is the generic form: give it devices (nets + finger
counts) and it searches for the sequence, draws the row, adds the well tie
ring, the well, gf180 pin labels and the LVS reference netlist.
``cc_current_mirror`` and ``cc_diff_pair`` are the two common-source cases
from the notes (§2.4), where common centroid and shared diffusion agree.
"""
from __future__ import annotations

import os
from typing import Literal, Optional, Sequence, Union

from glayout.backend import Component, rectangle
from glayout.pdk.mappedpdk import MappedPDK
from glayout.placement.finger_sequence.layout import finger_sequence_netlist, finger_sequence_row
from glayout.placement.finger_sequence.model import DeviceSpec, FingerSequence
from glayout.placement.finger_sequence.search import SequenceWeights, search_sequences
from glayout.primitives.guardring import tapring
from glayout.routing.straight_route import straight_route
from glayout.util.comp_utils import align_comp_to_port
from glayout.util.port_utils import add_ports_perimeter
from glayout.util.process_table import ProcessTable

DeviceLike = Union[DeviceSpec, dict]


def _specs(devices: Sequence[DeviceLike]) -> list[DeviceSpec]:
    return [d if isinstance(d, DeviceSpec) else DeviceSpec(**d) for d in devices]


def _label(pdk: MappedPDK, top: Component, text: str, port, alignment, size: float = 0.27) -> None:
    pin = rectangle(layer=pdk.get_glayer("met2_pin"), size=(size, size), centered=True).copy()
    pin.add_label(text=text, layer=pdk.get_glayer("met2_label"))
    top.add(align_comp_to_port(pin, port, alignment=alignment))


def finger_sequence_cell(
    pdk: MappedPDK,
    devices: Sequence[DeviceLike],
    sequence: Optional[str] = None,
    device: Literal["nfet", "pfet"] = "nfet",
    width: float = 3,
    length: Optional[float] = None,
    with_dummy: Union[bool, tuple[bool, bool]] = True,
    with_tie: bool = True,
    tie_layers: tuple[str, str] = ("met2", "met1"),
    rmult: Optional[int] = None,
    pins: Optional[Sequence[str]] = None,
    circuit_name: str = "FSCELL",
    weights: Optional[SequenceWeights] = None,
    name: str = "finger_sequence_cell",
) -> Component:
    """Generic one-row cell. ``sequence`` (e.g. ``"A+ A- B+ B-"``) skips the search.

    ``pins`` lists the external nets (default: every net plus ``B``, the bulk).
    """
    specs = _specs(devices)
    if isinstance(with_dummy, list):
        with_dummy = tuple(with_dummy)
    nets = []
    for d in specs:
        for n in (d.drain, d.gate, d.source):
            if n not in nets:
                nets.append(n)
    pins = list(pins) if pins else nets + ["B"]
    with ProcessTable(f"{name}: cell", f"{device}, pins {pins}") as pt:
        with pt.stage("Sequence") as st:
            if sequence is None:
                seq = search_sequences(specs, weights=weights, title=f"{name}: sequence search").best.sequence
                st.detail(f"searched: {seq.compact()}")
            else:
                seq = FingerSequence.from_compact({d.name: d for d in specs}, sequence)
                st.detail(f"given: {seq.compact()}")
        with pt.stage("Row") as st:
            row = finger_sequence_row(pdk, seq, device=device, width=width, length=length, with_dummy=with_dummy,
                                      rmult=rmult, title=f"{name}: row")
            st.detail(f"{row.xsize:.2f}x{row.ysize:.2f}µm, centroid error {row.info['centroid_error_um']}µm")
        # the final name goes on the flattened cell; gdsfactory suffixes "$1"
        # to any name assigned twice, which hides the cell from magic and LVS
        top = Component(f"{name}_hier")
        rref = top << row
        top.add_ports(rref.get_ports_list(), prefix="row_")
        tap_layer = "p+s/d" if device == "nfet" else "n+s/d"
        well = "pwell" if device == "nfet" else "nwell"
        with pt.stage("Tie ring") as st:
            if with_tie:
                tap_sep = max(float(pdk.util_max_metal_seperation()),
                              float(pdk.get_grule("active_diff", "active_tap")["min_separation"]))
                tap_sep += float(pdk.get_grule(tap_layer, "active_tap")["min_enclosure"])
                enc = (2 * (tap_sep + row.xmax), 2 * (tap_sep + row.ymax))
                tie = top << tapring(pdk, enclosed_rectangle=enc, sdlayer=tap_layer,
                                     horizontal_glayer=tie_layers[0], vertical_glayer=tie_layers[1])
                top.add_ports(tie.get_ports_list(), prefix="welltie_")
                tied = []
                for side, ring in (("L", "W"), ("R", "E")):
                    p = f"row_dummy_{side}_gsdcon_top_met_{ring}"
                    if p in top.ports:
                        top << straight_route(pdk, top.ports[p], top.ports[f"welltie_{ring}_top_met_{ring}"], glayer2="met1")
                        tied.append(side)
                st.detail(f"enclosing {enc[0]:.2f}x{enc[1]:.2f}µm, dummies tied {tied or 'none'}")
            else:
                st.status = "skip"
        with pt.stage("Well") as st:
            top.add_padding(layers=(pdk.get_glayer(well),), default=pdk.get_grule(well, "active_tap")["min_enclosure"])
            top = add_ports_perimeter(top, layer=pdk.get_glayer(well), prefix="well_")
            st.detail(well)
        with pt.stage("Pin labels") as st:
            if pdk.name.lower() == "gf180" and with_tie and not os.environ.get("GLAYOUT_NO_PIN_LABELS"):
                placed = []
                for net in pins:
                    if net == "B":
                        _label(pdk, top, "B", top.ports["welltie_S_top_met_S"], ("c", "c"), size=0.5)
                    elif f"row_{net}_sd_N" in top.ports:
                        _label(pdk, top, net, top.ports[f"row_{net}_sd_N"], ("c", "b"))
                    elif f"row_{net}_gate_S" in top.ports:
                        _label(pdk, top, net, top.ports[f"row_{net}_gate_S"], ("c", "t"))
                    else:
                        st.warn(f"no rail for pin {net}")
                        continue
                    placed.append(net)
                st.detail(f"{placed}")
            else:
                st.status = "skip"
                st.detail("gf180 only (klayout LVS needs named labels)")
        with pt.stage("Netlist") as st:
            # dummies sit on the bulk when tied to the ring, else on a local net
            dummy_net = "B" if with_tie else "DUM"
            nl = finger_sequence_netlist(pdk, seq, device, width, length, with_dummy=with_dummy,
                                         dummy_net=dummy_net, circuit_name=circuit_name, pins=pins)
            top.info["netlist"] = nl
            st.detail(f"{circuit_name}({', '.join(pins)})")
        with pt.stage("Flatten") as st:
            # magic types devices per GDS cell: a pfet row in a subcell, with the
            # nwell drawn in the parent, reads as p+ substrate taps (P-Ohmic)
            top = top.flatten()
            top.name = name
            top.info["netlist"] = nl
            st.detail(f"{len(top.get_polygons())} polygons")
        pt.result("sequence", seq.labels())
        pt.result("size_um", f"{top.xsize:.2f}x{top.ysize:.2f}")
    top.info["sequence"] = seq.compact()
    top.info["diagram"] = seq.diagram()
    top.info["centroid_error_um"] = row.info["centroid_error_um"]
    # external pins for the floorplanner: both ends of each net's rail; the
    # bulk is reachable from every side of the tie ring
    pin_ports = {}
    for net in pins:
        if net == "B":
            ring = [f"welltie_{s}_top_met_{s}" for s in "NESW"] if with_tie else [f"well_{s}" for s in "NESW"]
            pin_ports[net] = [p for p in ring if p in top.ports]
            continue
        for kind in ("sd", "gate"):
            ends = [f"row_{net}_{kind}_W", f"row_{net}_{kind}_E"]
            if all(p in top.ports for p in ends):
                pin_ports[net] = ends
                break
    top.info["pins"] = pin_ports
    return top


def cc_current_mirror(
    pdk: MappedPDK,
    fingers: int = 4,
    ratio: int = 1,
    device: Literal["nfet", "pfet"] = "nfet",
    width: float = 3,
    length: Optional[float] = None,
    with_dummy: Union[bool, tuple[bool, bool]] = True,
    with_tie: bool = True,
    sequence: Optional[str] = None,
    **kwargs,
) -> Component:
    """Common-centroid current mirror in one diffusion-shared row.

    A is the diode-connected reference (``fingers`` fingers), B the output
    (``fingers * ratio``). Pins: VREF, VOUT, VSS, B — the same as ``current_mirror``.
    """
    devices = [DeviceSpec("A", "VSS", "VREF", "VREF", fingers),
               DeviceSpec("B", "VSS", "VOUT", "VREF", fingers * ratio)]
    return finger_sequence_cell(pdk, devices, sequence=sequence, device=device, width=width, length=length,
                                with_dummy=with_dummy, with_tie=with_tie, pins=["VREF", "VOUT", "VSS", "B"],
                                circuit_name="CC_CMIRROR", name=f"cc_cmirror_{device}", **kwargs)


def cc_diff_pair(
    pdk: MappedPDK,
    fingers: int = 4,
    device: Literal["nfet", "pfet"] = "nfet",
    width: float = 3,
    length: Optional[float] = None,
    with_dummy: Union[bool, tuple[bool, bool]] = True,
    with_tie: bool = True,
    sequence: Optional[str] = None,
    **kwargs,
) -> Component:
    """Common-centroid differential pair in one row with a shared tail diffusion.

    Pins: VINP, VINN, VOUT1 (A drain), VOUT2 (B drain), VTAIL, B.
    """
    devices = [DeviceSpec("A", "VTAIL", "VOUT1", "VINP", fingers),
               DeviceSpec("B", "VTAIL", "VOUT2", "VINN", fingers)]
    return finger_sequence_cell(pdk, devices, sequence=sequence, device=device, width=width, length=length,
                                with_dummy=with_dummy, with_tie=with_tie,
                                pins=["VINP", "VINN", "VOUT1", "VOUT2", "VTAIL", "B"],
                                circuit_name="CC_DIFFPAIR", name=f"cc_diffpair_{device}", **kwargs)

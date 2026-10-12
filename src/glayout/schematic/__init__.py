"""Netlist-in: transistor-level schematics for the floorplanning flow.

- ``model``: ``Device``, ``Schematic``, pin names per kind, net classes
- ``readers``: AnalogGenie ``.cir`` entries, the curated textbook circuits
- ``annotate``: structures, matched devices and layout blocks
- ``sizing``: synthetic, layout-only sizes per block
- ``textbook``: the curated circuits (PR #127's 18 and the two-stage OTA)
- ``graph``: device + block heterograph for learning (schema v1); ``vocab``: its one-hot vocabularies
- ``to_circuit``: schematic → floorplanner ``Circuit`` (imported lazily, needs a PDK)

Everything except ``to_circuit`` is pure Python.
"""
from glayout.schematic.model import MOS, NET_CLASSES, PINS, Device, Schematic, net_class
from glayout.schematic.annotate import AnnotatedBlock, Annotation, Matched, Structure, annotate
from glayout.schematic.readers import (
    AG_CATEGORIES, AG_KINDS, NetlistParseError, analoggenie_category, from_textbook_circuit,
    parse_analoggenie, parse_ports, read_analoggenie,
)
from glayout.schematic.sizing import BlockSize, Sizing, size_schematic
from glayout.schematic.textbook import curated, textbook_schematic, two_stage_ota_schematic
from glayout.schematic.graph import SCHEMA_VERSION, HeteroGraph, circuit_graph, schematic_graph, validate_graph

_LAZY = ("schematic_to_circuit", "schematic_routed", "UnsupportedCircuit", "pdk_limits")

__all__ = [
    "AG_CATEGORIES", "AG_KINDS", "AnnotatedBlock", "Annotation", "BlockSize", "Device", "HeteroGraph", "MOS",
    "Matched", "NET_CLASSES", "NetlistParseError", "PINS", "SCHEMA_VERSION", "Schematic", "Sizing", "Structure",
    "analoggenie_category", "annotate", "circuit_graph", "curated", "from_textbook_circuit", "net_class",
    "parse_analoggenie", "parse_ports", "read_analoggenie", "schematic_graph", "size_schematic",
    "textbook_schematic", "two_stage_ota_schematic", "validate_graph", *_LAZY,
]


def __getattr__(name):
    if name in _LAZY:
        from glayout.schematic import to_circuit
        return getattr(to_circuit, name)
    raise AttributeError(name)

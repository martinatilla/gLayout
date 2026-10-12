"""Netlist-in: transistor-level schematics for the floorplanning flow.

- ``model``: ``Device``, ``Schematic``, pin names per kind, net classes
- ``readers``: AnalogGenie ``.cir`` entries, the curated textbook circuits

Pure Python; nothing here needs a PDK.
"""
from glayout.schematic.model import MOS, NET_CLASSES, PINS, Device, Schematic, net_class
from glayout.schematic.readers import (
    AG_CATEGORIES, AG_KINDS, NetlistParseError, analoggenie_category, from_textbook_circuit,
    parse_analoggenie, parse_ports, read_analoggenie,
)

__all__ = [
    "AG_CATEGORIES", "AG_KINDS", "Device", "MOS", "NET_CLASSES", "NetlistParseError", "PINS", "Schematic",
    "analoggenie_category", "from_textbook_circuit", "net_class", "parse_analoggenie", "parse_ports",
    "read_analoggenie",
]

"""Fixed vocabularies for one-hot features. Changing any of them changes the graph schema.

``BLOCK_KINDS`` must equal ``glayout.floorplan.circuit.KINDS`` (checked in the tests);
it is repeated here so that netlist-level graphs need no PDK or layout backend.
"""
DEVICE_KINDS = ("nfet", "pfet", "res", "cap", "ind", "diode", "npn", "pnp", "composite", "unknown")
TERMINALS = ("drain", "gate", "source", "body", "collector", "base", "emitter", "substrate", "p", "n", "other")
# pin name -> terminal role, per device kind (the same letter means different things: MOS B is the
# body, BJT B the base)
_ROLES = {
    "mos": {"D": "drain", "G": "gate", "S": "source", "B": "body"},
    "bjt": {"C": "collector", "B": "base", "E": "emitter", "S": "substrate"},
    "two": {"P": "p", "N": "n"},
}
_FAMILY = {"nfet": "mos", "pfet": "mos", "npn": "bjt", "pnp": "bjt", "res": "two", "cap": "two", "ind": "two",
           "diode": "two"}
NET_CLASSES = ("power", "ground", "bias", "clock", "signal")
BLOCK_KINDS = ("DP", "CM", "CMB", "CASC", "CS", "SF", "INV", "TG", "CAP", "RES", "BIAS", "OTHER")
STRUCTURES = ("dp", "degenerated_dp", "ccp", "mirror", "inverter", "tg", "stack")
SIDES = "NESW"
LAYERS = ("met1", "met2", "met3", "met4", "met5")


def device_kind_class(kind: str) -> str:
    if kind in DEVICE_KINDS:
        return kind
    if kind.startswith("composite:"):
        return "composite"
    return "unknown"


def terminal_role(kind: str, pin: str) -> str:
    return _ROLES.get(_FAMILY.get(kind, ""), {}).get(pin, "other")


def one_hot(value, vocab) -> list[float]:
    return [float(value == v) for v in vocab]

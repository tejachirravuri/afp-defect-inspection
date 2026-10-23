"""Standard defect classes 0 to 5.

Different methods in the group used their own class lists (three classes in
one, four in another, different names for the same thing), so data labelled
for one could not be reused for the other. This catalogue maps the terms of
the inspection system manual and of the literature onto one numbering. Every
label, detector output and database row uses these ids.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple


class DefectClass(IntEnum):
    OK = 0
    GAP = 1
    OVERLAP = 2
    OUT_OF_PLANE = 3
    FOREIGN_OBJECT = 4
    NO_DATA = 5


@dataclass(frozen=True)
class ClassInfo:
    name: str
    description: str
    ais_terms: Tuple[str, ...]
    literature_terms: Tuple[str, ...]
    weight: float              # used in the quality factor
    color: Tuple[float, float, float]


CATALOGUE = {
    DefectClass.OK: ClassInfo(
        "ok", "Tow surface or tool surface without findings.",
        (), (), 0.0, (0.85, 0.85, 0.85)),
    DefectClass.GAP: ClassInfo(
        "gap", "Missing material between or instead of tows: gaps between "
        "neighbouring tows, missing tows, gaps at ply edges.",
        ("GAPS",), ("gap", "narrow gap", "wide gap", "missing tow", "lap gap"),
        1.0, (0.12, 0.47, 0.71)),
    DefectClass.OVERLAP: ClassInfo(
        "overlap", "A tow lying partly on its neighbour, one extra ply of "
        "material along the tow edge.",
        (), ("overlap", "lap"), 1.0, (1.0, 0.5, 0.05)),
    DefectClass.OUT_OF_PLANE: ClassInfo(
        "out_of_plane", "Tow lifted out of the layup plane: twists, folds, "
        "wrinkles, puckers, bridging over a concave tool, pull-ups.",
        ("TWISTS", "FOLDINGS", "Z-defects"),
        ("twist", "fold", "wrinkle", "pucker", "bridging", "blister", "pull-up"),
        2.0, (0.84, 0.15, 0.16)),
    DefectClass.FOREIGN_OBJECT: ClassInfo(
        "foreign_object", "Small raised spot not belonging to a tow: foreign "
        "object debris, fuzz balls, resin balls.",
        ("CAULIFLOWERS",), ("FOD", "foreign object debris", "fuzzball", "resin ball"),
        2.0, (0.58, 0.40, 0.74)),
    DefectClass.NO_DATA: ClassInfo(
        "no_data", "No valid measurement inside the tape area (laser out of the "
        "field of view, shadowing). Not a material defect itself, but the area "
        "cannot be released without another look.",
        ("Loss of information",), ("occlusion", "missing data"),
        0.5, (0.2, 0.2, 0.2)),
}

NAMES = {c: CATALOGUE[c].name for c in DefectClass}
DEFECTS = [c for c in DefectClass if c != DefectClass.OK]


def from_name(name: str) -> DefectClass:
    for c, info in CATALOGUE.items():
        if info.name == name or name.upper() == c.name:
            return c
    raise ValueError("unknown defect class %r" % name)

"""Defect statistics and the quality factor Q of one tape course.

    Q = 1 - sum_c(w_c * A_c) / A_tape,   clipped to [0, 1]

A_c is the defect area of class c, A_tape the inspected tape area and w_c the
class weight from the catalogue (out of plane defects and foreign objects
count double, missing data half). Q = 1 means no findings. A course with 5 %
of its area covered by gaps gets Q = 0.95.

The weights are a proposal. In the end they should come from mechanical
tests that relate defect type and size to the loss of strength.
"""
from __future__ import annotations

from typing import Dict, Iterable

from .catalogue import CATALOGUE, DEFECTS, NAMES, DefectClass


def class_summary(defects: Iterable) -> Dict[str, Dict[str, float]]:
    summary = {NAMES[c]: {"count": 0, "area_mm2": 0.0} for c in DEFECTS}
    for d in defects:
        s = summary[NAMES[DefectClass(d.cls)]]
        s["count"] += 1
        s["area_mm2"] += d.area_mm2
    return summary


def quality_factor(defects: Iterable, tape_area_mm2: float) -> float:
    if tape_area_mm2 <= 0:
        return 0.0
    weighted = sum(CATALOGUE[DefectClass(d.cls)].weight * d.area_mm2 for d in defects)
    return max(0.0, min(1.0, 1.0 - weighted / tape_area_mm2))

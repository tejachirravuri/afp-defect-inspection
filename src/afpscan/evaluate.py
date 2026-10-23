"""Baseline and U-Net against the ground truth of cached synthetic scans."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .calibration import DEFAULT
from .catalogue import CATALOGUE, DefectClass
from .detect.baseline import BaselineConfig, TapeGeometry, detect, features
from .metrics import add_counts, confusion, object_counts, pixel_scores, rates
from .preprocess import HeightMap

BIN = 4


def q_from_labels(labels, tape, cell_area):
    area = tape.sum() * cell_area
    if area <= 0:
        return 0.0
    weighted = sum(CATALOGUE[c].weight * (labels[tape] == c).sum() * cell_area
                   for c in DefectClass if c != DefectClass.OK)
    return max(0.0, 1.0 - weighted / area)


def evaluate_cached(files: Iterable[Path], model_path: Optional[str] = None,
                    min_area_mm2: float = 1.0) -> dict:
    cal = DEFAULT.binned(BIN)
    cell = cal.dx_mm * cal.dy_mm
    min_cells = max(1, math.ceil(min_area_mm2 / cell))

    model = None
    if model_path and Path(model_path).is_file():
        from .detect import unet
        model = unet.load(model_path)

    methods = {"baseline": BaselineConfig(),
               "cross_section": BaselineConfig(surface_method="cross_section")}
    if model is not None:
        methods["unet"] = model
    conf = {m: 0 for m in methods}
    objects = {m: {} for m in methods}
    q_err = {m: [] for m in methods}
    n = 0
    for f in files:
        z = np.load(f)
        y = z["y"]
        tape = z["x"][3] > 0
        n += 1
        q_gt = q_from_labels(y, tape, cell)
        h = z["h"].astype(np.float32)
        h[h < -8] = np.nan
        hm = HeightMap(h, cal, (0.0, 0.0, 0.0))
        geo = TapeGeometry(200.0, 50.8, 8)
        f_median = features(hm, BaselineConfig(), geo)
        for name, method in methods.items():
            if isinstance(method, BaselineConfig):
                pred = detect(hm, method, geo).labels
            else:
                from .detect import unet
                pred = unet.detect(model, hm, f_median, 8, min_area_mm2).labels
            conf[name] = conf[name] + confusion(y, pred, tape)
            add_counts(objects[name], object_counts(y, pred, tape, min_cells))
            q_err[name].append(abs(q_from_labels(pred, tape, cell) - q_gt))

    report = {"scans": n, "min_area_mm2": min_area_mm2}
    for name in methods:
        report[name] = {
            "pixel": pixel_scores(conf[name]),
            "objects": rates(objects[name]),
            "q_mean_abs_error": float(np.mean(q_err[name])) if q_err[name] else float("nan"),
            "confusion": conf[name].tolist(),
        }
    return report

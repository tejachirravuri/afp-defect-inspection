"""Comparing label maps: per-class pixel scores and object level detection rates."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage as ndi

from .catalogue import DEFECTS, NAMES, DefectClass

N = len(DefectClass)
EIGHT = np.ones((3, 3), dtype=bool)


def confusion(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g, p = gt[mask].astype(np.int64), pred[mask].astype(np.int64)
    return np.bincount(g * N + p, minlength=N * N).reshape(N, N)


def pixel_scores(conf: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for c in DefectClass:
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        out[NAMES[c]] = {
            "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan"),
            "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
            "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "cells": int(conf[c, :].sum()),
        }
    defect = conf[1:, 1:].sum()          # any defect class vs ok
    fp = conf[0, 1:].sum()
    fn = conf[1:, 0].sum()
    out["any_defect"] = {"iou": float(defect / (defect + fp + fn)) if defect + fp + fn else float("nan"),
                         "precision": float(defect / (defect + fp)) if defect + fp else float("nan"),
                         "recall": float(defect / (defect + fn)) if defect + fn else float("nan"),
                         "cells": int(conf[1:, :].sum())}
    return out


def _components(binary, min_cells):
    lab, n = ndi.label(binary, structure=EIGHT)
    if n == 0:
        return lab, []
    sizes = ndi.sum(binary, lab, index=np.arange(1, n + 1))
    return lab, [i + 1 for i in range(n) if sizes[i] >= min_cells]


def object_counts(gt, pred, mask, min_cells=8, tolerance=2) -> Dict[str, Dict[str, int]]:
    """A ground truth defect counts as found if a prediction of the same class
    touches it (within ``tolerance`` cells), and the other way round for
    predictions. ``any_defect`` ignores the class."""
    gt = np.where(mask, gt, 0)
    pred = np.where(mask, pred, 0)
    grow = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    out = {}
    targets = [(NAMES[c], gt == c, pred == c) for c in DEFECTS]
    targets.append(("any_defect", gt > 0, pred > 0))
    for name, g, p in targets:
        g_near = ndi.binary_dilation(g, grow)
        p_near = ndi.binary_dilation(p, grow)
        g_lab, g_ids = _components(g, min_cells)
        p_lab, p_ids = _components(p, min_cells)
        found = sum(1 for i in g_ids if p_near[g_lab == i].any())
        correct = sum(1 for i in p_ids if g_near[p_lab == i].any())
        out[name] = {"gt": len(g_ids), "found": found, "pred": len(p_ids), "correct": correct}
    return out


def add_counts(total, counts):
    for name, c in counts.items():
        t = total.setdefault(name, {"gt": 0, "found": 0, "pred": 0, "correct": 0})
        for k, v in c.items():
            t[k] += v
    return total


def rates(counts):
    out = {}
    for name, c in counts.items():
        out[name] = {
            "recall": c["found"] / c["gt"] if c["gt"] else float("nan"),
            "precision": c["correct"] / c["pred"] if c["pred"] else float("nan"),
            **c,
        }
    return out

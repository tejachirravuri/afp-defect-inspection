"""From raw counts to a height map above the tool surface in millimetres."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .calibration import DEFAULT, Calibration

PLY_MM = 0.18  # nominal ply thickness used for thresholds


@dataclass
class HeightMap:
    h: np.ndarray                 # (rows, cols) float32, mm above tool, NaN = invalid
    cal: Calibration              # scale of this (binned) grid
    plane: Tuple[float, float, float]  # tool plane in counts: a*col + b*row + c, raw grid

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.h)

    @property
    def cell_area_mm2(self) -> float:
        return self.cal.dx_mm * self.cal.dy_mm


def bin_columns(counts: np.ndarray, factor: int, min_valid: float = 0.5) -> np.ndarray:
    """Mean of each block of ``factor`` points along the laser line.
    Blocks with less than ``min_valid`` valid points become NaN."""
    rows, cols = counts.shape
    cols -= cols % factor
    z = counts[:, :cols].astype(np.float32).reshape(rows, cols // factor, factor)
    valid = z > 0
    n = valid.sum(axis=2)
    total = np.where(valid, z, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / n
    out[n < max(1, int(np.ceil(min_valid * factor)))] = np.nan
    return out


def _fit_plane(cols, rows, z):
    a = np.column_stack([cols, rows, np.ones_like(cols)])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    return coef


def ransac_plane(cols, rows, z, threshold, iterations=300, rng=None):
    """Plane z = a*col + b*row + c with the most inliers, refined by least squares."""
    rng = rng or np.random.default_rng(0)
    best, best_count = None, -1
    n = len(z)
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        try:
            coef = np.linalg.solve(
                np.column_stack([cols[idx], rows[idx], np.ones(3)]), z[idx])
        except np.linalg.LinAlgError:
            continue
        count = np.count_nonzero(np.abs(cols * coef[0] + rows * coef[1] + coef[2] - z) < threshold)
        if count > best_count:
            best, best_count = coef, count
    inliers = np.abs(cols * best[0] + rows * best[1] + best[2] - z) < threshold
    return _fit_plane(cols[inliers], rows[inliers], z[inliers]), inliers


def tool_plane(counts: np.ndarray, cal: Calibration, sample: int = 40000,
               rng=None) -> Tuple[float, float, float]:
    """Plane of the tool surface in raw counts.

    RANSAC finds the largest plane. If the tape covers most of the view that
    plane is the tape top, recognisable by many points clearly below it; the
    fit is then repeated on those lower points.
    """
    rng = rng or np.random.default_rng(0)
    rows, cols = np.nonzero(counts)
    if len(rows) < 100:
        raise ValueError("scan has almost no valid points")
    pick = rng.choice(len(rows), min(sample, len(rows)), replace=False)
    r, c = rows[pick].astype(float), cols[pick].astype(float)
    z = counts[rows[pick], cols[pick]].astype(float)

    threshold = 0.4 * PLY_MM / cal.dz_mm
    coef, _ = ransac_plane(c, r, z, threshold, rng=rng)
    residual = z - (c * coef[0] + r * coef[1] + coef[2])
    lower = residual < -0.6 * PLY_MM / cal.dz_mm
    if lower.mean() > 0.10 and lower.sum() > 1000:
        coef, _ = ransac_plane(c[lower], r[lower], z[lower], threshold, rng=rng)
    return tuple(float(v) for v in coef)


def preprocess(scan_or_counts, cal: Calibration = DEFAULT, bin_factor: int = 4,
               plane: Optional[Tuple[float, float, float]] = None) -> HeightMap:
    counts = getattr(scan_or_counts, "counts", scan_or_counts)
    if plane is None:
        plane = tool_plane(counts, cal)
    z = bin_columns(counts, bin_factor)
    rows, cols = np.indices(z.shape, dtype=np.float32)
    raw_cols = cols * bin_factor + (bin_factor - 1) / 2.0
    ref = plane[0] * raw_cols + plane[1] * rows + plane[2]
    h = ((z - ref) * cal.dz_mm).astype(np.float32)
    return HeightMap(h, cal.binned(bin_factor), plane)

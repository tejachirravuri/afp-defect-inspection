"""Rule based defect detection on the height map.

1. Tool surface: modelled as row offset + column offset (waviness along the
   travel direction, crown across), fitted on points classified as tool with
   Otsu's threshold, a few iterations.
2. Tape step (height of one ply) taken from the scan itself. Thresholds are
   fractions of this step or a multiple of the measured noise, whichever is
   larger, so the unknown height calibration does not change the result.
3. Tape area: with the layup program the programmed length and width are
   placed where the most material is, then extended over material laid right
   next to it. Without a program, the largest region that is mostly material.
4. Local tape surface by a median filter about 10 mm along and 15 mm across,
   wider than two tows so a missing tow or a twist cannot pull it along.
   Alternatively ("cross_section") a quadratic fitted to each laser profile
   after quantile screening, as in Tang et al. (2021). Residual = height -
   surface.
5. Candidates: residual below -gap threshold (gap), above +raise threshold
   (raised material), no measurement inside the tape (no data). With
   hysteresis: regions grown at a lower threshold are kept only if they
   contain enough cells above the full one.
6. Candidates are grouped with DBSCAN, each group is classified by shape and
   height.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN

from ..catalogue import DefectClass
from ..preprocess import HeightMap


@dataclass
class BaselineConfig:
    smooth: Tuple[int, int] = (5, 5)          # rows, cols of the noise median
    segment: Tuple[int, int] = (31, 31)       # smoothing used to find the tape
    surface_mm: Tuple[float, float] = (10.0, 15.0)  # local tape surface window, along x across
    surface_method: str = "median"            # or "cross_section" (per profile quadratic)
    quantiles: Tuple[float, float] = (0.25, 0.75)
    gap_depth: float = 0.5                    # in tape steps, group median must reach this
    raise_height: float = 0.75                # in tape steps, group median must reach this
    noise_k: float = 3.5                      # ... and at least k * noise
    low_fraction: float = 0.6                 # candidate cells: this fraction of the above
    noise_k_low: float = 2.5
    min_strong: int = 12                      # strong cells a region needs to be kept
    edge_margin_mm: float = 1.0               # band along the tape edges that is not inspected
    end_margin_mm: float = 2.0                # same at the add and cut ends
    eps_mm: float = 0.8                       # DBSCAN neighbourhood
    min_samples: int = 6
    min_area_mm2: float = 1.0
    min_gap_length_mm: float = 3.0
    min_gap_width_mm: float = 0.4             # narrower seams between tows are not counted
    overlap_min_length_mm: float = 15.0
    overlap_max_width_mm: float = 4.0
    overlap_max_height: float = 1.6           # in tape steps
    fod_max_size_mm: float = 4.0
    tape_mode: str = "auto"                   # "full": whole valid area is laid material
    nominal_step_mm: float = 0.27             # step used in "full" mode


@dataclass
class TapeGeometry:
    """What is known about the tape from the layup program (all optional)."""
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    n_tows: Optional[int] = 8


@dataclass
class Features:
    rel: np.ndarray        # smoothed height above tool, mm
    resid: np.ndarray      # height above local tape surface, mm
    valid: np.ndarray
    tape: np.ndarray
    inner: np.ndarray      # tape without the edge margin
    step: float
    noise: float


@dataclass
class Defect:
    cls: DefectClass
    area_mm2: float
    length_mm: float          # extent along travel
    width_mm: float           # extent across the tape
    max_height_mm: float      # signed extreme of the residual
    x_mm: float               # centroid across
    y_mm: float               # centroid along travel
    bbox: Tuple[int, int, int, int]  # row0, row1, col0, col1 on the analysis grid
    tow: Optional[int] = None
    score: float = 1.0


@dataclass
class Detection:
    labels: np.ndarray                 # class per cell
    tape: np.ndarray                   # bool, tape area
    defects: List[Defect]
    step_mm: float
    tape_area_mm2: float
    edges: dict = field(default_factory=dict)
    noise_mm: float = 0.0


# --- helpers ---------------------------------------------------------------

def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    return centers[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]


def _otsu_split(a, valid):
    sample = a[valid][:: max(1, int(valid.sum()) // 200000)]
    return otsu(np.clip(sample, *np.percentile(sample, [0.5, 99.5])))


def _interp_nan(a):
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    return np.interp(idx, idx[ok], a[ok])


def _fill_nan_rows(a):
    out = a.copy()
    with np.errstate(all="ignore"):
        med = np.nanmedian(out, axis=1)
    med = np.where(np.isfinite(med), med, np.nanmedian(out))
    idx = np.where(~np.isfinite(out))
    out[idx] = med[idx[0]]
    return out


def _best_window(frac, size):
    csum = np.concatenate([[0.0], np.cumsum(frac - frac.mean())])
    start = int(np.argmax(csum[size:] - csum[:-size]))
    return start, start + size


# --- steps -----------------------------------------------------------------

def relative_height(hm: HeightMap, cfg: BaselineConfig, iterations: int = 3):
    """Smoothed height above the fitted tool surface."""
    filled = _fill_nan_rows(hm.h)
    hs = ndi.median_filter(filled, size=cfg.smooth)
    coarse = ndi.uniform_filter(hs, size=cfg.segment)
    valid = hm.valid
    # a low percentile per profile is the tool level right next to the tape,
    # which keeps the first Otsu split from mixing tool and tape
    row_off = ndi.median_filter(np.percentile(coarse, 10, axis=1), size=31,
                                mode="nearest").astype(np.float32)
    col_off = np.zeros(hs.shape[1], dtype=np.float32)
    with np.errstate(all="ignore"):
        for _ in range(iterations):
            rel = coarse - row_off[:, None] - col_off[None, :]
            tool = valid & (rel < _otsu_split(rel, valid))
            resid = np.where(tool, coarse - col_off[None, :], np.nan)
            enough = tool.sum(axis=1)[:, None] > 20
            r = _interp_nan(np.nanmedian(np.where(enough, resid, np.nan), axis=1))
            row_off = ndi.median_filter(r, size=31, mode="nearest").astype(np.float32)
            resid = np.where(tool, coarse - row_off[:, None], np.nan)
            enough = tool.sum(axis=0)[None, :] > 20
            c = _interp_nan(np.nanmedian(np.where(enough, resid, np.nan), axis=0))
            col_off = ndi.median_filter(c, size=15, mode="nearest").astype(np.float32)
    return hs - row_off[:, None] - col_off[None, :]


def tape_area(rel, valid, cfg: BaselineConfig, geometry: TapeGeometry, dx_mm, dy_mm):
    coarse = ndi.uniform_filter(rel, size=cfg.segment)
    above = coarse > _otsu_split(coarse, valid)
    step = float(np.median(coarse[above & valid]) - np.median(coarse[~above & valid]))
    material = coarse > 0.5 * step
    n_rows, n_cols = material.shape
    tape = np.zeros_like(material)

    length = int(round(geometry.length_mm / dy_mm)) if geometry.length_mm else None
    width = int(round(geometry.width_mm / dx_mm)) if geometry.width_mm else None

    row_frac = material.mean(axis=1)
    if length and length < n_rows:
        r0, r1 = _best_window(row_frac, length)
    else:
        rows = np.nonzero(row_frac > 0.5 * row_frac.max())[0]
        if len(rows) == 0:
            return tape, step
        r0, r1 = int(rows[0]), int(rows[-1]) + 1

    frac = material[r0:r1].mean(axis=0)
    if width and width < n_cols:
        c0, c1 = _best_window(frac, width)
        tow = max(1, width // max(1, geometry.n_tows or 1))
        # material laid earlier right next to the tape is inspected as well
        while c0 - tow >= 0 and frac[c0 - tow:c0].mean() > 0.7:
            c0 -= tow
        while c1 + tow <= n_cols and frac[c1:c1 + tow].mean() > 0.7:
            c1 += tow
    else:
        cols = ndi.binary_closing(frac > 0.5, structure=np.ones(31))
        lab, n = ndi.label(cols)
        if n == 0:
            return tape, step
        sizes = ndi.sum(cols, lab, index=np.arange(1, n + 1))
        run = np.nonzero(lab == 1 + int(np.argmax(sizes)))[0]
        c0, c1 = int(run[0]), int(run[-1]) + 1
    if length and length < n_rows:
        r0, r1 = _refine_ends(coarse[:, c0:c1].mean(axis=1), r0, length)
    if width and c1 - c0 < n_cols:
        c0, c1 = _refine_ends(coarse[r0:r1].mean(axis=0), c0, c1 - c0, search=16, k=8)
    tape[r0:r1, c0:c1] = True
    return tape, step


def _refine_ends(profile, r0, length, search=120, k=20):
    """Shifts a window of fixed size to where the height jumps at both of its
    ends are largest (add and cut along travel, tape edges across)."""
    n = len(profile)
    csum = np.concatenate([[0.0], np.cumsum(profile)])

    def mean(a, b):
        a, b = max(0, a), min(n, b)
        return (csum[b] - csum[a]) / max(1, b - a)

    best, best_r = -np.inf, r0
    for r in range(max(k, r0 - search), min(n - length - k, r0 + search) + 1):
        e = r + length
        score = (mean(r, r + k) - mean(r - k, r)) + (mean(e - k, e) - mean(e, e + k))
        if score > best:
            best, best_r = score, r
    return best_r, best_r + length


def _surface(a, cfg, cal, decimate=4):
    """Median filter with a large window, computed on a decimated grid."""
    small = a[::decimate, ::decimate]
    size = (max(3, int(round(cfg.surface_mm[0] / (cal.dy_mm * decimate))) | 1),
            max(3, int(round(cfg.surface_mm[1] / (cal.dx_mm * decimate))) | 1))
    med = ndi.median_filter(small, size=size, mode="nearest")
    zoom = (a.shape[0] / med.shape[0], a.shape[1] / med.shape[1])
    up = ndi.zoom(med, zoom, order=1, mode="nearest")
    out = np.empty_like(a)
    r, c = min(a.shape[0], up.shape[0]), min(a.shape[1], up.shape[1])
    out[:r, :c] = up[:r, :c]
    out[r:, :] = out[r - 1:r, :]
    out[:, c:] = out[:, c - 1:c]
    return out


def _cross_section_surface(rel, tape, n_tows, cfg):
    """Nominal tape surface fitted line by line, after Tang et al. (2021).

    For every laser profile the tape columns are split into n_tows // 2 + 1
    groups. In each group only heights between the two quantiles are kept,
    a quadratic is fitted to the kept points of all groups, then refitted
    once without points more than 2.5 sigma off. Outside the tape the nearest
    profile value is used.
    """
    surface = rel.copy()
    rows = np.nonzero(tape.any(axis=1))[0]
    groups = (n_tows or 8) // 2 + 1
    lo, hi = cfg.quantiles
    for r in rows:
        cols = np.nonzero(tape[r])[0]
        if len(cols) < 3 * groups:
            continue
        z = rel[r, cols]
        keep = np.zeros(len(cols), dtype=bool)
        for idx in np.array_split(np.arange(len(cols)), groups):
            ql, qh = np.quantile(z[idx], [lo, hi])
            keep[idx] = (z[idx] >= ql) & (z[idx] <= qh)
        u = (cols - cols.mean()) / max(1.0, np.ptp(cols) / 2)
        coef = np.polyfit(u[keep], z[keep], 2)
        res = z - np.polyval(coef, u)
        sigma = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        good = np.abs(res) <= 2.5 * max(sigma, 1e-6)
        if good.sum() > 10:
            coef = np.polyfit(u[good], z[good], 2)
        all_u = (np.arange(rel.shape[1]) - cols.mean()) / max(1.0, np.ptp(cols) / 2)
        line = np.polyval(coef, all_u)
        line[:cols[0]] = line[cols[0]]
        line[cols[-1] + 1:] = line[cols[-1]]
        surface[r] = line
    return surface


def features(hm: HeightMap, cfg: BaselineConfig = BaselineConfig(),
             geometry: TapeGeometry = TapeGeometry()) -> Features:
    cal, valid = hm.cal, hm.valid
    rel = relative_height(hm, cfg)
    if cfg.tape_mode == "full":
        tape = ndi.binary_fill_holes(ndi.binary_closing(valid, structure=np.ones((15, 15))))
        tape = ndi.binary_erosion(tape, structure=np.ones((15, 15)))
        step = cfg.nominal_step_mm
    else:
        tape, step = tape_area(rel, valid, cfg, geometry, cal.dx_mm, cal.dy_mm)
        step = _measured_step(rel, valid, tape, step)

    margin = max(1, int(round(cfg.edge_margin_mm / cal.dx_mm)))
    end = max(1, int(round(cfg.end_margin_mm / cal.dy_mm)))
    inner = ndi.binary_erosion(tape, structure=np.ones((2 * end + 1, 2 * margin + 1)))
    if tape.any() and cfg.surface_method == "cross_section":
        surface = _cross_section_surface(rel, tape, geometry.n_tows, cfg)
    elif tape.any():
        padded = rel.copy()
        _, (ri, ci) = ndi.distance_transform_edt(~tape, return_indices=True)
        padded[~tape] = rel[ri[~tape], ci[~tape]]
        surface = _surface(padded, cfg, cal)
    else:
        surface = _surface(rel, cfg, cal)
    resid = rel - surface
    inside = resid[inner & valid]
    noise = float(1.4826 * np.median(np.abs(inside - np.median(inside)))) if inside.size else 0.0
    return Features(rel, resid, valid, tape, inner, step, noise)


def _measured_step(rel, valid, tape, fallback):
    """Tape median minus the tool right beside the tape, in the tape rows."""
    if not tape.any():
        return fallback
    rows = tape.any(axis=1)
    near = ndi.binary_dilation(tape, structure=np.ones((1, 61))) & ~ndi.binary_dilation(
        tape, structure=np.ones((1, 9)))
    tool = near & valid & rows[:, None]
    if tool.sum() < 500:
        return fallback
    step = float(np.median(rel[tape & valid]) - np.median(rel[tool]))
    return step if step > 0 else fallback


def _hysteresis(low, high, min_strong):
    lab, n = ndi.label(low, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return low
    strong = ndi.sum(high, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], strong >= min_strong])
    return keep[lab]


def classify(kind, rows, cols, resid, step, cal, cfg) -> Optional[DefectClass]:
    length = (rows.max() - rows.min() + 1) * cal.dy_mm
    width = (cols.max() - cols.min() + 1) * cal.dx_mm
    area = len(rows) * cal.dx_mm * cal.dy_mm
    if area < cfg.min_area_mm2:
        return None
    if kind == "gap":
        return DefectClass.GAP if length >= cfg.min_gap_length_mm else None
    if kind == "nodata":
        return DefectClass.NO_DATA
    peak = np.percentile(resid, 90) / step
    if (length >= cfg.overlap_min_length_mm and width <= cfg.overlap_max_width_mm
            and peak <= cfg.overlap_max_height and length / max(width, 1e-6) > 5):
        return DefectClass.OVERLAP
    if max(length, width) <= cfg.fod_max_size_mm:
        return DefectClass.FOREIGN_OBJECT
    return DefectClass.OUT_OF_PLANE


def _tow_index(col_center, tape_cols, n_tows):
    if len(tape_cols) == 0 or not n_tows:
        return None
    rel = (col_center - tape_cols[0]) / max(1, tape_cols[-1] - tape_cols[0] + 1)
    return int(np.clip(rel * n_tows, 0, n_tows - 1)) + 1


def defects_from_labels(labels, f: Features, hm: HeightMap, n_tows=None,
                        eps_mm=0.8, min_samples=6, min_area_mm2=1.0,
                        min_gap_width_mm=0.4) -> List[Defect]:
    """Group a label map into defect instances (also used for model output).
    Groups below the minimum area and gaps narrower than the minimum width
    are dropped."""
    cal = hm.cal
    tape_cols = np.nonzero(f.tape.any(axis=0))[0]
    width_mm = len(tape_cols) * cal.dx_mm
    tows = n_tows
    if n_tows and width_mm > 0:
        tows = max(1, int(round(width_mm / 6.35)))
    out = []
    for cls in np.unique(labels):
        if cls == 0:
            continue
        rows, cols = np.nonzero(labels == cls)
        coords = np.column_stack([cols * cal.dx_mm, rows * cal.dy_mm])
        groups = DBSCAN(eps=eps_mm, min_samples=min_samples).fit_predict(coords)
        for g in np.unique(groups):
            if g < 0:
                continue
            sel = groups == g
            r, c = rows[sel], cols[sel]
            area = len(r) * hm.cell_area_mm2
            if area < min_area_mm2:
                continue
            if cls == DefectClass.GAP and _median_width(r, c, cal) < min_gap_width_mm:
                continue
            res = f.resid[r, c]
            res = res[np.isfinite(res) & f.valid[r, c]]
            extreme = float(res[np.argmax(np.abs(res))]) if len(res) and cls != DefectClass.NO_DATA else 0.0
            out.append(Defect(
                cls=DefectClass(int(cls)), area_mm2=area,
                length_mm=(r.max() - r.min() + 1) * cal.dy_mm,
                width_mm=(c.max() - c.min() + 1) * cal.dx_mm,
                max_height_mm=extreme,
                x_mm=float(c.mean() * cal.dx_mm), y_mm=float(r.mean() * cal.dy_mm),
                bbox=(int(r.min()), int(r.max()), int(c.min()), int(c.max())),
                tow=_tow_index(c.mean(), tape_cols, tows)))
    return out


def _median_width(rows, cols, cal):
    """Median width across the tape of a group, profile by profile."""
    order = np.argsort(rows, kind="stable")
    r, c = rows[order], cols[order]
    starts = np.r_[0, np.nonzero(np.diff(r))[0] + 1, len(r)]
    widths = [(c[a:b].max() - c[a:b].min() + 1) for a, b in zip(starts[:-1], starts[1:])]
    return float(np.median(widths)) * cal.dx_mm


def labels_from_defects(labels, defects) -> np.ndarray:
    """Label map that keeps only the cells of the given defects."""
    keep = np.zeros_like(labels)
    for d in defects:
        r0, r1, c0, c1 = d.bbox
        box = labels[r0:r1 + 1, c0:c1 + 1]
        sel = box == int(d.cls)
        keep[r0:r1 + 1, c0:c1 + 1][sel] = int(d.cls)
    return keep


def tape_edges(tape, cal) -> dict:
    rows = np.nonzero(tape.any(axis=1))[0]
    cols = np.nonzero(tape.any(axis=0))[0]
    if len(rows) == 0:
        return {}
    return {
        "add_mm": float(rows[0] * cal.dy_mm),
        "cut_mm": float(rows[-1] * cal.dy_mm),
        "length_mm": float((rows[-1] - rows[0] + 1) * cal.dy_mm),
        "width_mm": float((cols[-1] - cols[0] + 1) * cal.dx_mm),
        "left_mm": float(cols[0] * cal.dx_mm),
        "right_mm": float(cols[-1] * cal.dx_mm),
    }


def detect(hm: HeightMap, cfg: BaselineConfig = BaselineConfig(),
           geometry: TapeGeometry = TapeGeometry(), f: Optional[Features] = None) -> Detection:
    cal = hm.cal
    f = f or features(hm, cfg, geometry)
    labels = np.zeros(f.rel.shape, dtype=np.uint8)
    if not f.tape.any():
        return Detection(labels, f.tape, [], f.step, 0.0, {}, f.noise)

    # hysteresis: regions grown at a low threshold (8-connected) are kept when
    # they contain enough cells above the full threshold
    gap_thr = max(cfg.gap_depth * f.step, cfg.noise_k * f.noise)
    raise_thr = max(cfg.raise_height * f.step, cfg.noise_k * f.noise)
    gap_low = max(cfg.low_fraction * cfg.gap_depth * f.step, cfg.noise_k_low * f.noise)
    raise_low = max(cfg.low_fraction * cfg.raise_height * f.step, cfg.noise_k_low * f.noise)
    usable = f.inner & f.valid
    candidates = {
        "gap": _hysteresis(usable & (f.resid < -gap_low), usable & (f.resid < -gap_thr),
                           cfg.min_strong),
        "raised": _hysteresis(usable & (f.resid > raise_low), usable & (f.resid > raise_thr),
                              cfg.min_strong),
        "nodata": f.tape & ~f.valid,
    }
    for kind, mask in candidates.items():
        rows, cols = np.nonzero(mask)
        if len(rows) < cfg.min_samples:
            continue
        coords = np.column_stack([cols * cal.dx_mm, rows * cal.dy_mm])
        groups = DBSCAN(eps=cfg.eps_mm, min_samples=cfg.min_samples).fit_predict(coords)
        for g in np.unique(groups):
            if g < 0:
                continue
            sel = groups == g
            r, c = rows[sel], cols[sel]
            res = f.resid[r, c] if kind != "nodata" else np.zeros(len(r))
            cls = classify(kind, r, c, res, f.step, cal, cfg)
            if cls is not None:
                labels[r, c] = int(cls)

    defects = defects_from_labels(labels, f, hm, geometry.n_tows, cfg.eps_mm,
                                  cfg.min_samples, cfg.min_area_mm2, cfg.min_gap_width_mm)
    labels = labels_from_defects(labels, defects)
    return Detection(labels, f.tape, defects, f.step,
                     float(f.tape.sum() * hm.cell_area_mm2), tape_edges(f.tape, cal), f.noise)

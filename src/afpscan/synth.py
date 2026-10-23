"""Synthetic laser scans of one tape course with labelled defects.

There are no labelled real scans, so training and evaluation data come from
here. The generator produces raw sensor counts in the same layout as the AIS
export (profiles x 2048 points, 0 = no measurement), and its statistics are
tuned to the real scans: tool waviness along the travel direction, a lateral
crown, a tilted plane, ply step between 0.17 and 0.30 mm, fibre striations,
point noise with outliers and random dropouts.

Defects follow the catalogue: gaps and missing tows, overlaps, twists (a
raised lens on the tow with gaps next to it, like the real one in the scans),
transverse folds, foreign objects and areas without data, sometimes with a
raised rim around them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

from .calibration import DEFAULT, Calibration
from .catalogue import DefectClass


@dataclass
class SynthConfig:
    rows: int = 1528
    cols: int = 2048
    length_mm: float = 200.0
    n_tows: int = 8
    tow_mm: float = 6.35
    ply_mm: Tuple[float, float] = (0.17, 0.30)
    noise_mm: float = 0.07
    speckle_mm: Tuple[float, float] = (0.06, 0.14)  # correlated part of the noise
    speckle_sigma: Tuple[float, float] = (1.0, 3.0)
    outlier_frac: float = 0.02
    outlier_mm: float = 0.3
    striation_mm: float = 0.03
    waviness_mm: float = 0.08
    dropout_frac: float = 0.004
    defects: Tuple[int, int] = (0, 4)
    clean_prob: float = 0.15


@dataclass
class SynthDefect:
    cls: DefectClass
    kind: str
    rows: Tuple[int, int]
    cols: Tuple[int, int]


@dataclass
class SynthScan:
    counts: np.ndarray       # uint16 raw counts
    labels: np.ndarray       # uint8 class per raw point
    defects: List[SynthDefect]
    tape: Tuple[int, int, int, int]  # row0, row1, col0, col1
    ply_mm: float
    cal: Calibration


def _smooth_noise(rng, shape, sigma, scale):
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / (n.std() + 1e-9) * scale


class _Builder:
    def __init__(self, cfg: SynthConfig, cal: Calibration, rng):
        self.cfg, self.cal, self.rng = cfg, cal, rng
        R, C = cfg.rows, cfg.cols
        self.ply = rng.uniform(*cfg.ply_mm)
        self.h = np.zeros((R, C), dtype=np.float32)
        self.labels = np.zeros((R, C), dtype=np.uint8)
        self.invalid = np.zeros((R, C), dtype=bool)
        self.defects: List[SynthDefect] = []

        self.tow_px = cfg.tow_mm / cal.dx_mm
        self.tape_px = int(round(cfg.n_tows * self.tow_px))
        self.len_px = int(round(cfg.length_mm / cal.dy_mm))
        self.r0 = int(rng.integers(150, max(151, R - self.len_px - 80)))
        self.r1 = min(R, self.r0 + self.len_px)
        self.c0 = int(rng.integers(250, max(251, C - self.tape_px - 60)))
        self.c1 = self.c0 + self.tape_px

    def tow_bounds(self, k):
        a = self.c0 + k * self.tow_px
        return int(round(a)), int(round(a + self.tow_px))

    def mm_rows(self, mm):
        return max(1, int(round(mm / self.cal.dy_mm)))

    def mm_cols(self, mm):
        return max(1, int(round(mm / self.cal.dx_mm)))

    # surfaces --------------------------------------------------------------
    def tool(self):
        cfg, rng = self.cfg, self.rng
        R, C = self.h.shape
        rows = np.arange(R)[:, None]
        cols = np.arange(C)[None, :]
        wave = np.zeros((R, 1))
        for _ in range(3):
            period = rng.uniform(150, 600)
            wave += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * rows / period + rng.uniform(0, 6.3))
        wave *= cfg.waviness_mm / max(1e-6, np.abs(wave).max())
        crown = rng.uniform(-0.06, 0.06) * ((cols - C / 2) / (C / 2)) ** 2
        texture = _smooth_noise(rng, (R, C), 2.0, 0.02)
        self.h += (wave + crown + texture).astype(np.float32)

    def tape(self):
        cfg, rng = self.cfg, self.rng
        r0, r1 = self.r0, self.r1
        drift = _smooth_noise(rng, (r1 - r0, 1), 60, 0.03)
        stri = ndi.gaussian_filter1d(rng.standard_normal(self.h.shape[1]), 1.5)
        stri = stri / stri.std() * cfg.striation_mm
        slow = _smooth_noise(rng, (r1 - r0, self.h.shape[1]), (40, 3), 0.015)
        for k in range(cfg.n_tows):
            a, b = self.tow_bounds(k)
            # rounded tow edges and an occasional hairline gap below detection size
            shrink = int(rng.integers(0, 3)) if 0 < k else 0
            a += shrink
            prof = np.ones(b - a)
            ramp = min(4, (b - a) // 4)
            prof[:ramp] = np.linspace(0.3, 1, ramp)
            prof[-ramp:] = np.linspace(1, 0.3, ramp)
            height = self.ply + rng.normal(0, 0.01)
            self.h[r0:r1, a:b] += (height * prof[None, :] + drift + stri[None, a:b]
                                   + slow[:, a:b]).astype(np.float32)

    # defects ---------------------------------------------------------------
    def _span(self, length_mm):
        n = min(self.mm_rows(length_mm), self.r1 - self.r0 - 2)
        start = int(self.rng.integers(self.r0, self.r1 - n))
        return start, start + n

    def _set_label(self, mask, cls):
        self.labels[mask] = int(cls)

    def gap(self):
        rng = self.rng
        k = int(rng.integers(1, self.cfg.n_tows))
        edge = self.tow_bounds(k)[0]
        width = self.mm_cols(rng.uniform(0.4, 2.5))
        ra, rb = self._span(rng.uniform(10, 150))
        a = edge - width // 2
        b = a + width
        region = (slice(ra, rb), slice(a, b))
        self.h[region] -= self.ply * rng.uniform(0.8, 1.0)
        self._set_label(np.s_[ra:rb, a:b], DefectClass.GAP)
        self.defects.append(SynthDefect(DefectClass.GAP, "gap", (ra, rb), (a, b)))

    def missing_tow(self):
        rng = self.rng
        k = int(rng.integers(0, self.cfg.n_tows))
        a, b = self.tow_bounds(k)
        ra, rb = self._span(rng.uniform(15, 80))
        self.h[ra:rb, a:b] -= self.ply
        self._set_label(np.s_[ra:rb, a:b], DefectClass.GAP)
        self.defects.append(SynthDefect(DefectClass.GAP, "missing_tow", (ra, rb), (a, b)))

    def overlap(self):
        rng = self.rng
        k = int(rng.integers(1, self.cfg.n_tows))
        edge = self.tow_bounds(k)[0]
        width = self.mm_cols(rng.uniform(0.6, 3.0))
        ra, rb = self._span(rng.uniform(20, 150))
        a = edge if rng.random() < 0.5 else edge - width
        b = a + width
        self.h[ra:rb, a:b] += self.ply * rng.uniform(0.9, 1.1)
        self._set_label(np.s_[ra:rb, a:b], DefectClass.OVERLAP)
        self.defects.append(SynthDefect(DefectClass.OVERLAP, "overlap", (ra, rb), (a, b)))

    def twist(self):
        rng = self.rng
        k = int(rng.integers(0, self.cfg.n_tows))
        a, b = self.tow_bounds(k)
        ra, rb = self._span(rng.uniform(25, 80))
        amp = self.ply * rng.uniform(1.0, 3.0)
        s = (np.arange(ra, rb) - ra) / max(1, rb - ra - 1)
        half = (b - a) / 2.0
        xc = (a + b) / 2.0
        x = np.arange(a, b)[None, :]
        lens = half * (0.35 + 0.65 * np.sin(np.pi * s))[:, None] * 0.75
        inside = np.abs(x - xc) <= lens
        bump = amp * np.sin(np.pi * s)[:, None] * np.cos(np.pi / 2 * np.clip((x - xc) / np.maximum(lens, 1), -1, 1))
        block = self.h[ra:rb, a:b]
        block += np.where(inside, bump, 0).astype(np.float32)
        # the twisted tow is narrower, the rest of its width is a gap
        block -= np.where(~inside, self.ply, 0).astype(np.float32)
        lab = self.labels[ra:rb, a:b]
        lab[~inside] = int(DefectClass.GAP)
        lab[inside & (bump > 0.4 * self.ply)] = int(DefectClass.OUT_OF_PLANE)
        self.defects.append(SynthDefect(DefectClass.OUT_OF_PLANE, "twist", (ra, rb), (a, b)))

    def fold(self):
        rng = self.rng
        k0 = int(rng.integers(0, self.cfg.n_tows))
        k1 = min(self.cfg.n_tows - 1, k0 + int(rng.integers(0, 3)))
        a, b = self.tow_bounds(k0)[0], self.tow_bounds(k1)[1]
        ra, rb = self._span(rng.uniform(3, 10))
        amp = self.ply * rng.uniform(0.9, 2.5)
        s = (np.arange(ra, rb) - ra) / max(1, rb - ra - 1)
        ridge = (amp * np.sin(np.pi * s) ** 2)[:, None] * np.ones((1, b - a))
        self.h[ra:rb, a:b] += ridge.astype(np.float32)
        lab = self.labels[ra:rb, a:b]
        lab[ridge > 0.4 * self.ply] = int(DefectClass.OUT_OF_PLANE)
        self.defects.append(SynthDefect(DefectClass.OUT_OF_PLANE, "fold", (ra, rb), (a, b)))

    def _ellipse(self, rc, cc, ry, rx):
        R, C = self.h.shape
        r0, r1 = max(0, rc - ry), min(R, rc + ry + 1)
        c0, c1 = max(0, cc - rx), min(C, cc + rx + 1)
        rr, cc_ = np.mgrid[r0:r1, c0:c1]
        d = ((rr - rc) / max(ry, 1)) ** 2 + ((cc_ - cc) / max(rx, 1)) ** 2
        return (slice(r0, r1), slice(c0, c1)), d

    def foreign_object(self):
        rng = self.rng
        size = rng.uniform(0.8, 3.5)
        ry, rx = self.mm_rows(size / 2), self.mm_cols(size / 2 * rng.uniform(0.6, 1.4))
        rc = int(rng.integers(self.r0 + ry, self.r1 - ry))
        cc = int(rng.integers(self.c0 + rx + 10, self.c1 - rx - 10))
        sl, d = self._ellipse(rc, cc, ry, rx)
        amp = self.ply * rng.uniform(0.8, 3.0)
        dome = np.where(d < 1, amp * np.sqrt(np.clip(1 - d, 0, 1)), 0)
        self.h[sl] += dome.astype(np.float32)
        self.labels[sl][dome > 0.3 * self.ply] = int(DefectClass.FOREIGN_OBJECT)
        self.defects.append(SynthDefect(DefectClass.FOREIGN_OBJECT, "fod",
                                        (sl[0].start, sl[0].stop), (sl[1].start, sl[1].stop)))

    def no_data(self):
        rng = self.rng
        ry = self.mm_rows(rng.uniform(1.5, 10) / 2)
        rx = self.mm_cols(rng.uniform(1.5, 10) / 2)
        rc = int(rng.integers(self.r0 + ry, self.r1 - ry))
        cc = int(rng.integers(self.c0 + rx, self.c1 - rx))
        sl, d = self._ellipse(rc, cc, ry, rx)
        if rng.random() < 0.5:   # material piled up around the shadow
            rim = (d >= 1) & (d < 2.2)
            amp = self.ply * rng.uniform(1.0, 2.0)
            self.h[sl] += np.where(rim, amp * (2.2 - d) / 1.2, 0).astype(np.float32)
            self.labels[sl][rim & ((2.2 - d) / 1.2 * amp > 0.4 * self.ply)] = int(DefectClass.OUT_OF_PLANE)
        self.invalid[sl] |= d < 1
        self.labels[sl][d < 1] = int(DefectClass.NO_DATA)
        self.defects.append(SynthDefect(DefectClass.NO_DATA, "no_data",
                                        (sl[0].start, sl[0].stop), (sl[1].start, sl[1].stop)))

    # sensor ----------------------------------------------------------------
    def sensor(self) -> np.ndarray:
        cfg, rng, cal = self.cfg, self.rng, self.cal
        R, C = self.h.shape
        h = self.h + rng.normal(0, cfg.noise_mm, (R, C)).astype(np.float32)
        h += _smooth_noise(rng, (R, C), cfg.speckle_sigma,
                           rng.uniform(*cfg.speckle_mm)).astype(np.float32)
        out = rng.random((R, C)) < cfg.outlier_frac
        h[out] += rng.normal(0, cfg.outlier_mm, out.sum()).astype(np.float32)

        invalid = self.invalid | (rng.random((R, C)) < cfg.dropout_frac)
        for _ in range(int(rng.integers(10, 60))):     # short dropout streaks
            r = int(rng.integers(0, R - 40))
            c = int(rng.integers(0, C))
            invalid[r:r + int(rng.integers(3, 30)), c:c + int(rng.integers(1, 3))] = True

        a, b = rng.uniform(0.35, 0.5), rng.uniform(-0.3, 0.2)
        rows, cols = np.indices((R, C), dtype=np.float32)
        counts = rng.uniform(30000, 30700) + a * cols + b * rows + h / cal.dz_mm
        counts = np.clip(np.round(counts), 1, 65535).astype(np.uint16)
        counts[invalid] = 0
        return counts


GENERATORS = {
    "gap": (lambda b: b.gap(), 3),
    "missing_tow": (lambda b: b.missing_tow(), 1),
    "overlap": (lambda b: b.overlap(), 2),
    "twist": (lambda b: b.twist(), 2),
    "fold": (lambda b: b.fold(), 2),
    "fod": (lambda b: b.foreign_object(), 3),
    "no_data": (lambda b: b.no_data(), 2),
}


def synth_scan(seed: int, cfg: SynthConfig = SynthConfig(), cal: Calibration = DEFAULT) -> SynthScan:
    rng = np.random.default_rng(seed)
    b = _Builder(cfg, cal, rng)
    b.tool()
    b.tape()
    if rng.random() >= cfg.clean_prob:
        names = list(GENERATORS)
        weights = np.array([GENERATORS[n][1] for n in names], dtype=float)
        for _ in range(int(rng.integers(max(1, cfg.defects[0]), cfg.defects[1] + 1))):
            GENERATORS[rng.choice(names, p=weights / weights.sum())][0](b)
    counts = b.sensor()
    labels = b.labels.copy()
    labels[(labels != int(DefectClass.NO_DATA)) & (counts == 0) & (labels != 0)] = 0
    return SynthScan(counts, labels, b.defects, (b.r0, b.r1, b.c0, b.c1), b.ply, cal)


def bin_labels(labels: np.ndarray, factor: int) -> np.ndarray:
    """Label of each block of ``factor`` points: the most frequent defect class
    if it covers at least half the block, else 0."""
    rows, cols = labels.shape
    cols -= cols % factor
    blocks = labels[:, :cols].reshape(rows, cols // factor, factor)
    out = np.zeros((rows, cols // factor), dtype=np.uint8)
    best = np.zeros_like(out, dtype=np.int32)
    for cls in range(1, 6):
        n = (blocks == cls).sum(axis=2)
        take = (n > best) & (n * 2 >= factor)
        out[take] = cls
        best = np.maximum(best, np.where(take, n, best))
    return out

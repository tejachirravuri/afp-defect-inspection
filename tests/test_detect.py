import numpy as np
import pytest

from afpscan.catalogue import DefectClass
from afpscan.detect.baseline import TapeGeometry, detect, features
from afpscan.preprocess import bin_columns, preprocess, ransac_plane
from afpscan.quality import class_summary, quality_factor
from afpscan.synth import SynthConfig, _Builder, bin_labels, synth_scan
from afpscan.calibration import DEFAULT

GEO = TapeGeometry(200.0, 50.8, 8)


def test_bin_columns_handles_invalid():
    counts = np.array([[10, 20, 0, 0, 30, 30, 30, 0]], dtype=np.uint16)
    out = bin_columns(counts, 4)
    assert out[0, 0] == 15.0
    assert out[0, 1] == 30.0
    assert np.isnan(bin_columns(np.array([[1, 0, 0, 0]], dtype=np.uint16), 4)[0, 0])


def test_ransac_recovers_plane_with_outliers():
    rng = np.random.default_rng(0)
    c, r = rng.uniform(0, 2000, 5000), rng.uniform(0, 1500, 5000)
    z = 0.4 * c - 0.2 * r + 30000 + rng.normal(0, 5, 5000)
    z[:1500] += 200          # a raised block, like a tape
    coef, inliers = ransac_plane(c, r, z, threshold=40)
    assert np.allclose(coef, [0.4, -0.2, 30000], atol=[0.01, 0.01, 5])
    assert inliers[1500:].mean() > 0.95


def _scan_with(kind, seed=7):
    rng = np.random.default_rng(seed)
    b = _Builder(SynthConfig(), DEFAULT, rng)
    b.tool()
    b.tape()
    getattr(b, kind)()
    return b.sensor(), b.labels, b


def test_tape_found_at_programmed_size():
    s = synth_scan(11)
    f = features(preprocess(s.counts), geometry=GEO)
    rows = np.nonzero(f.tape.any(axis=1))[0]
    assert abs(rows[0] - s.tape[0]) <= 5
    assert abs(len(rows) * DEFAULT.dy_mm - 200) < 1
    assert abs(f.step - s.ply_mm) < 0.1


@pytest.mark.parametrize("kind,cls", [
    ("missing_tow", DefectClass.GAP),
    ("twist", DefectClass.OUT_OF_PLANE),
    ("no_data", DefectClass.NO_DATA),
])
def test_defect_is_found_where_it_is(kind, cls):
    counts, labels, _ = _scan_with(kind)
    det = detect(preprocess(counts), geometry=GEO)
    gt = bin_labels(labels, 4) == int(cls)
    assert gt.any()
    hit = det.labels[gt] == int(cls)
    assert hit.mean() > 0.2, "only %.0f %% of the %s cells found" % (100 * hit.mean(), kind)


def test_clean_scan_has_high_quality():
    cfg = SynthConfig(clean_prob=1.0)
    for seed in (21, 22):
        s = synth_scan(seed, cfg)
        assert not s.defects
        det = detect(preprocess(s.counts), geometry=GEO)
        assert quality_factor(det.defects, det.tape_area_mm2) > 0.97


def test_quality_factor():
    class D:
        def __init__(self, cls, area):
            self.cls, self.area_mm2 = cls, area
    defects = [D(DefectClass.GAP, 50), D(DefectClass.OUT_OF_PLANE, 25)]
    # weights: gap 1, out of plane 2 -> (50 + 50) / 10000
    assert quality_factor(defects, 10000) == pytest.approx(0.99)
    assert quality_factor([], 10000) == 1.0
    assert quality_factor(defects, 0) == 0.0
    s = class_summary(defects)
    assert s["gap"] == {"count": 1, "area_mm2": 50}
    assert s["foreign_object"]["count"] == 0


def test_bin_labels_majority():
    labels = np.array([[1, 1, 0, 0, 3, 3, 3, 0, 0, 0, 0, 2]], dtype=np.uint8)
    assert bin_labels(labels, 4).tolist() == [[1, 3, 0]]

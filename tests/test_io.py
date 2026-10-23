from pathlib import Path

import numpy as np
import pytest

from afpscan.calibration import DEFAULT
from afpscan.io import parse_name, read_3d, read_ply, to_points, write_3d, write_ply

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_parse_name():
    meta = parse_name("OF_108_P_1_T_3_D_90_C_0_R_1528_000010.3d")
    assert meta == {"program": 108, "ply": 1, "tape": 3, "angle_deg": 90, "course": 0,
                    "profiles": 1528, "sequence": 10}


def test_write_read_roundtrip(tmp_path):
    counts = np.random.default_rng(0).integers(0, 40000, (50, 64), dtype=np.uint16)
    path = tmp_path / "OF_1_P_1_T_2_D_0_C_0_R_50_000001.3d"
    write_3d(path, counts, buffer_rows=200)
    scan = read_3d(path)
    assert scan.shape == (50, 64)
    assert np.array_equal(scan.counts, counts)
    assert scan.sensor == "C5_2040_GigE"
    assert scan.meta["tape"] == 2


def test_profiles_override_and_bounds(tmp_path):
    path = tmp_path / "x.3d"
    write_3d(path, np.ones((10, 16), dtype=np.uint16), buffer_rows=20)
    assert read_3d(path).shape == (20, 16)
    assert read_3d(path, profiles=10).shape == (10, 16)
    with pytest.raises(ValueError):
        read_3d(path, profiles=30)


def test_truncated_file(tmp_path):
    path = tmp_path / "bad.3d"
    write_3d(path, np.ones((10, 16), dtype=np.uint16))
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(ValueError, match="truncated"):
        read_3d(path)


def test_ply_roundtrip(tmp_path):
    pts = np.random.default_rng(1).random((100, 3))
    cls = np.arange(100) % 6
    write_ply(tmp_path / "a.ply", pts, cls)
    back = read_ply(tmp_path / "a.ply")
    assert back.shape == (100, 4)
    assert np.allclose(back[:, :3], pts, atol=1e-6)
    assert np.array_equal(back[:, 3], cls)


def test_points_skip_invalid(tmp_path):
    path = tmp_path / "x.3d"
    counts = np.full((4, 8), 1000, dtype=np.uint16)
    counts[0, 0] = 0
    write_3d(path, counts)
    pts = to_points(read_3d(path), DEFAULT)
    assert len(pts) == 31
    assert np.allclose(pts[:, 2], 1000 * DEFAULT.dz_mm)


@pytest.mark.skipif(not DATA.is_dir(), reason="raw scans not present")
def test_real_program():
    xml = next(DATA.rglob("*R_1528_000002.xml"))
    from afpscan.io import read_program
    tape = read_program(xml).tape(2)
    assert len(tape.fibers) == 8
    assert all(abs(f.width_mm - 6.35) < 0.1 for f in tape.fibers)
    assert all(f.length_mm == 200 for f in tape.fibers)

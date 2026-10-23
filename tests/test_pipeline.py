import json

import numpy as np
import pytest

from afpscan import database
from afpscan.cli import main
from afpscan.io import write_3d
from afpscan.metrics import confusion, object_counts, pixel_scores
from afpscan.pipeline import analyse
from afpscan.synth import synth_scan


@pytest.fixture(scope="module")
def scan_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("scans")
    s = synth_scan(5)
    path = d / ("OF_1_P_1_T_1_D_90_C_0_R_%d_000001.3d" % s.counts.shape[0])
    write_3d(path, s.counts, buffer_rows=6000)
    return path


def test_analyse_and_database(scan_file, tmp_path):
    result, hm, det = analyse(scan_file)
    assert 0.0 <= result.q <= 1.0
    assert result.meta["profiles"] == hm.h.shape[0]
    con = database.connect(tmp_path / "r.db")
    sid = database.insert(con, result, material="CF/PA6")
    assert database.known(con, result.sha1, "baseline")
    # inserting again replaces the row instead of duplicating it
    database.insert(con, result)
    assert con.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 1
    n = con.execute("SELECT COUNT(*) FROM defects").fetchone()[0]
    assert n == len(result.defects)
    assert len(database.summary(con)) == 5


def test_cli_info_and_analyse(scan_file, tmp_path, capsys):
    assert main(["info", str(scan_file)]) == 0
    assert "profiles" in capsys.readouterr().out
    out = tmp_path / "out"
    assert main(["analyse", str(scan_file), "--db", str(tmp_path / "a.db"),
                 "--out", str(out)]) == 0
    report = json.loads(next(out.glob("*.json")).read_text())
    assert report["method"] == "baseline"
    assert set(report["summary"]) == {"gap", "overlap", "out_of_plane", "foreign_object", "no_data"}


def test_cli_export_ply(scan_file, tmp_path):
    assert main(["export-ply", str(scan_file), str(tmp_path / "p.ply"), "--step", "8"]) == 0
    assert (tmp_path / "p.ply").stat().st_size > 1000


def test_watch_once(scan_file, tmp_path):
    from afpscan.watch import watch
    watch(scan_file.parent, tmp_path / "w.db", tmp_path / "w", once=True)
    watch(scan_file.parent, tmp_path / "w.db", tmp_path / "w", once=True)  # nothing new
    con = database.connect(tmp_path / "w.db")
    assert con.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 1


def test_metrics():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    gt[10:14, 10:14] = 3
    pred = np.zeros_like(gt)
    pred[2:6, 3:7] = 1          # overlaps the gap
    pred[15:19, 0:4] = 4        # false foreign object
    mask = np.ones_like(gt, dtype=bool)
    scores = pixel_scores(confusion(gt, pred, mask))
    assert scores["gap"]["iou"] == pytest.approx(12 / 20)
    obj = object_counts(gt, pred, mask, min_cells=4)
    assert obj["gap"] == {"gt": 1, "found": 1, "pred": 1, "correct": 1}
    assert obj["out_of_plane"]["found"] == 0
    assert obj["foreign_object"] == {"gt": 0, "found": 0, "pred": 1, "correct": 0}
    assert obj["any_defect"]["gt"] == 2 and obj["any_defect"]["found"] == 1


def test_unet_shapes():
    torch = pytest.importorskip("torch")
    from afpscan.detect.unet import N_CLASSES, UNet, predict_array
    model = UNet(base=4).eval()
    x = np.random.default_rng(0).random((4, 50, 37)).astype(np.float32)
    x[3] = 1
    labels = predict_array(model, x)
    assert labels.shape == (50, 37)
    assert labels.max() < N_CLASSES
    with torch.no_grad():
        assert model(torch.zeros(1, 4, 64, 64)).shape == (1, N_CLASSES, 64, 64)

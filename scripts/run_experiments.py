#!/usr/bin/env python3
"""Reproduces everything under results/.

    python scripts/run_experiments.py --data <synthetic cache> [--model models/unet_synthetic.pt]

1. baseline and U-Net on the synthetic test split -> results/synthetic_eval.json
2. both methods on every real scan in data/raw     -> results/real/, results/results.db,
                                                       results/real_scans.csv
3. figures                                          -> results/figures/
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from afpscan import database  # noqa: E402
from afpscan.calibration import DEFAULT  # noqa: E402
from afpscan.catalogue import CATALOGUE, DEFECTS, NAMES  # noqa: E402
from afpscan.evaluate import evaluate_cached  # noqa: E402
from afpscan.pipeline import analyse  # noqa: E402
from afpscan.preprocess import HeightMap  # noqa: E402
from afpscan.viz import legend, overlay  # noqa: E402


def synthetic(data: Path, model_path: Path, out: Path):
    files = sorted((data / "test").glob("*.npz"))
    report = evaluate_cached(files, str(model_path))
    (out / "synthetic_eval.json").write_text(json.dumps(report, indent=1))

    methods = [m for m in ("baseline", "cross_section", "unet") if m in report]
    lines = ["| class | " + " | ".join("%s IoU | %s recall | %s precision" % (m, m, m) for m in methods) + " |",
             "|---|" + "---|---|---|" * len(methods)]
    for name in [NAMES[c] for c in DEFECTS] + ["any_defect"]:
        cells = []
        for m in methods:
            px, ob = report[m]["pixel"][name], report[m]["objects"][name]
            cells += ["%.2f" % px["iou"], "%.2f (%d/%d)" % (ob["recall"], ob["found"], ob["gt"]),
                      "%.2f" % ob["precision"]]
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    lines.append("")
    lines.append("Mean absolute error of Q: " + ", ".join(
        "%s %.4f" % (m, report[m]["q_mean_abs_error"]) for m in methods))
    (out / "synthetic_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if "unet" in report:
        from afpscan.detect import unet
        model = unet.load(model_path)
        cal = DEFAULT.binned(4)
        picks = files[:4]
        from afpscan.detect.baseline import BaselineConfig, TapeGeometry, detect
        fig, axes = plt.subplots(len(picks), 4, figsize=(24, 3.2 * len(picks)))
        for row, f in zip(axes, picks):
            z = np.load(f)
            h = z["h"].astype(np.float32)
            h[h < -8] = np.nan
            hm = HeightMap(h, cal, (0, 0, 0))
            tape = z["x"][3] > 0
            geo = TapeGeometry(200.0, 50.8, 8)
            xs = detect(hm, BaselineConfig(surface_method="cross_section"), geo).labels
            base = detect(hm, BaselineConfig(), geo).labels
            from afpscan.detect.baseline import features
            net = unet.detect(model, hm, features(hm, BaselineConfig(), geo)).labels
            for ax, lab, title in [(row[0], z["y"], "ground truth"),
                                   (row[1], base, "baseline (median surface)"),
                                   (row[2], xs, "baseline (cross section)"),
                                   (row[3], net, "U-Net")]:
                overlay(ax, hm, lab, tape, "%s - %s" % (f.stem, title))
        legend(axes[0][0])
        fig.tight_layout()
        fig.savefig(out / "figures" / "synthetic_examples.png", dpi=80)
        plt.close(fig)

        hist_path = model_path.with_suffix(".history.json")
        if hist_path.is_file():
            hist = json.loads(hist_path.read_text())
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot([h["iteration"] for h in hist], [h["train_loss"] for h in hist], label="train")
            ax.plot([h["iteration"] for h in hist], [h["val_loss"] for h in hist], label="validation")
            ax.set_xlabel("iteration")
            ax.set_ylabel("cross entropy + dice")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "figures" / "training_curve.png", dpi=100)
            plt.close(fig)
    return report


def real(raw: Path, model_path: Path, out: Path):
    model = None
    if model_path.is_file():
        from afpscan.detect import unet
        model = unet.load(model_path)
    (out / "real").mkdir(exist_ok=True)
    db = out / "results.db"
    if db.exists():
        db.unlink()
    con = database.connect(db)
    rows = []
    scans = [p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
             for p in sorted(raw.resolve().rglob("*.3d"))]
    methods = ["baseline", "cross_section"] + (["unet"] if model else [])
    fig, axes = plt.subplots(len(scans), len(methods), figsize=(10 * len(methods), 3 * len(scans)),
                             squeeze=False)
    for ax_row, path in zip(axes, scans):
        for col, method in enumerate(methods):
            result, hm, det = analyse(path, method, model)
            database.insert(con, result)
            (out / "real" / ("%s_%s.json" % (path.stem, method))).write_text(
                json.dumps(result.to_dict(), indent=1))
            rows.append([path.stem, method, round(result.q, 4), len(result.defects)]
                        + [result.summary[NAMES[c]]["count"] for c in DEFECTS]
                        + [round(result.summary[NAMES[c]]["area_mm2"], 1) for c in DEFECTS]
                        + [round(result.step_mm, 3), round(result.noise_mm, 3),
                           round(result.edges.get("length_mm", 0), 1),
                           round(result.edges.get("width_mm", 0), 1)])
            overlay(ax_row[col], hm, det.labels, det.tape,
                    "%s  %s  Q=%.3f" % (path.stem, method, result.q))
            print("%-40s %-8s Q=%.3f defects=%d" % (path.stem, method, result.q, len(result.defects)))
    legend(axes[0][0])
    fig.tight_layout()
    fig.savefig(out / "figures" / "real_scans.png", dpi=60)
    plt.close(fig)

    header = (["scan", "method", "q", "defects"] + ["n_" + NAMES[c] for c in DEFECTS]
              + ["area_" + NAMES[c] + "_mm2" for c in DEFECTS]
              + ["step_mm", "noise_mm", "tape_length_mm", "tape_width_mm"])
    with (out / "real_scans.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    con.close()


def single_figures(raw: Path, model_path: Path, out: Path):
    """Larger figures of the scans that show each kind of finding."""
    picks = {"twist": "R_1528_000004", "gap": "R_1528_000009", "no_data": "R_1528_000007"}
    model = None
    if model_path.is_file():
        from afpscan.detect import unet
        model = unet.load(model_path)
    for name, key in picks.items():
        path = next(raw.rglob("*%s.3d" % key))
        methods = ["baseline", "cross_section"] + (["unet"] if model else [])
        fig, axes = plt.subplots(len(methods), 1, figsize=(12, 3.4 * len(methods)), squeeze=False)
        for ax, method in zip(axes[:, 0], methods):
            result, hm, det = analyse(path, method, model)
            overlay(ax, hm, det.labels, det.tape, "%s  %s  Q=%.3f" % (path.stem, method, result.q))
            ax.set_xlim(det.edges["add_mm"] - 15, det.edges["cut_mm"] + 15)
        legend(axes[0, 0])
        fig.tight_layout()
        fig.savefig(out / "figures" / ("real_%s.png" % name), dpi=110)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, help="synthetic cache built with 'afpscan build-dataset'")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "unet_synthetic.pt")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    if args.data:
        synthetic(args.data, args.model, args.out)
    real(args.raw, args.model, args.out)
    single_figures(args.raw, args.model, args.out)


if __name__ == "__main__":
    main()

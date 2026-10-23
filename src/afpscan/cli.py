"""Command line interface.

    afpscan info SCAN.3d
    afpscan export-ply SCAN.3d OUT.ply [--step 2] [--with-labels]
    afpscan analyse SCAN.3d ... [--method baseline|cross_section|unet] [--db results.db] [--out DIR]
    afpscan watch FOLDER --db results.db --out DIR
    afpscan synth OUT_DIR --seeds 0-9
    afpscan build-dataset DIR
    afpscan train DIR --model models/unet.pt
    afpscan evaluate DIR/test --model models/unet.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("afpscan")


def _seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _model(args):
    if args.method != "unet":
        return None
    from .detect import unet
    return unet.load(args.model)


def _cfg(args):
    from .detect.baseline import BaselineConfig
    return BaselineConfig(tape_mode=getattr(args, "tape_mode", "auto"))


def cmd_info(args):
    from .io import read_3d
    s = read_3d(args.scan)
    valid = s.valid
    print("%s\n  sensor %s, %d profiles x %d points, %.1f %% valid" % (
        s.path.name, s.sensor or "?", s.shape[0], s.shape[1], 100 * valid.mean()))
    print("  from file name: %s" % ", ".join("%s=%s" % kv for kv in s.meta.items()))
    if s.program:
        t = s.program.tape(s.meta.get("tape", -1))
        print("  program: %d tapes" % len(s.program.tapes))
        if t:
            print("  tape %d: length %.1f mm, max speed %g, %d fibres" % (
                t.number, t.length_mm, t.max_speed, len(t.fibers)))
            for f in t.fibers:
                print("    fibre %d: %.0f mm long, y %.2f .. %.2f mm (%.2f mm wide)" % (
                    f.number, f.length_mm, f.y_min_mm, f.y_max_mm, f.width_mm))
    return 0


def cmd_export_ply(args):
    from .calibration import DEFAULT
    from .io import read_3d, to_points, write_ply
    scan = read_3d(args.scan)
    if not args.with_labels:
        pts = to_points(scan, DEFAULT, args.step)
        write_ply(args.out, pts)
        print("wrote %d points to %s" % (len(pts), args.out))
        return 0
    from .pipeline import BIN, analyse
    _, hm, det = analyse(args.scan, cfg=_cfg(args), scan=scan)
    rows, cols = np.nonzero(scan.counts[::args.step, ::args.step])
    rows, cols = rows * args.step, cols * args.step
    pts = np.column_stack([cols * DEFAULT.dx_mm, rows * DEFAULT.dy_mm,
                           scan.counts[rows, cols] * DEFAULT.dz_mm])
    lab = det.labels[rows, np.minimum(cols // BIN, det.labels.shape[1] - 1)]
    write_ply(args.out, pts, lab)
    print("wrote %d points with defect classes to %s" % (len(pts), args.out))
    return 0


def cmd_analyse(args):
    from . import database
    from .pipeline import analyse
    model = _model(args)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    con = database.connect(args.db) if args.db else None
    rows = []
    for path in args.scans:
        result, hm, det = analyse(path, args.method, model, _cfg(args))
        rows.append(result)
        print("%-45s Q=%.3f  %s" % (Path(path).name, result.q, "  ".join(
            "%s %d" % (k, v["count"]) for k, v in result.summary.items() if v["count"])))
        if con:
            database.insert(con, result, args.material)
        if out:
            (out / (Path(path).stem + "_%s.json" % args.method)).write_text(
                json.dumps(result.to_dict(), indent=1))
            if args.plot:
                from .viz import save_overlay
                save_overlay(out / (Path(path).stem + "_%s.png" % args.method), hm, det,
                             "%s  %s  Q=%.3f" % (Path(path).stem, args.method, result.q))
    return 0


def cmd_watch(args):
    from .watch import watch
    watch(args.folder, args.db, args.out, args.method, _model(args), args.interval,
          args.once, args.material)
    return 0


def cmd_synth(args):
    from .io import write_3d
    from .synth import synth_scan
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in _seeds(args.seeds):
        s = synth_scan(seed)
        name = "SYN_%05d_T_1_R_%d" % (seed, s.counts.shape[0])
        write_3d(out / (name + ".3d"), s.counts, buffer_rows=6000)
        np.savez_compressed(out / (name + "_labels.npz"), labels=s.labels)
        print(name, ", ".join(d.kind for d in s.defects) or "clean")
    return 0


def cmd_build_dataset(args):
    from .dataset import SPLITS, build
    for split, seeds in SPLITS.items():
        files = build(seeds, Path(args.dir) / split, args.workers)
        print(split, len(files))
    return 0


def cmd_train(args):
    from .detect import unet
    data = Path(args.dir)
    cfg = unet.TrainConfig(iterations=args.iterations, batch=args.batch, base=args.base)
    history = unet.train(sorted((data / "train").glob("*.npz")),
                         sorted((data / "val").glob("*.npz")), Path(args.model), cfg)
    Path(args.model).with_suffix(".history.json").write_text(json.dumps(history, indent=1))
    return 0


def cmd_evaluate(args):
    from .evaluate import evaluate_cached
    report = evaluate_cached(sorted(Path(args.dir).glob("*.npz")), args.model)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="afpscan", description="AFP laser scan defect inspection")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def method_args(sp):
        sp.add_argument("--method", choices=["baseline", "cross_section", "unet"], default="baseline")
        sp.add_argument("--model", default="models/unet_synthetic.pt")
        sp.add_argument("--tape-mode", choices=["auto", "full"], default="auto",
                        help="'full' for scans where the whole view is laid material")

    sp = sub.add_parser("info", help="show header, file name fields and layup program")
    sp.add_argument("scan")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("export-ply", help="point cloud for CloudCompare / Open3D")
    sp.add_argument("scan")
    sp.add_argument("out")
    sp.add_argument("--step", type=int, default=2, help="take every n-th point")
    sp.add_argument("--with-labels", action="store_true", help="add defect class as scalar field")
    sp.add_argument("--tape-mode", choices=["auto", "full"], default="auto")
    sp.set_defaults(func=cmd_export_ply)

    sp = sub.add_parser("analyse", help="inspect one or more scans")
    sp.add_argument("scans", nargs="+")
    method_args(sp)
    sp.add_argument("--db", help="SQLite file to add results to")
    sp.add_argument("--out", help="folder for JSON reports (and plots)")
    sp.add_argument("--plot", action="store_true")
    sp.add_argument("--material")
    sp.set_defaults(func=cmd_analyse)

    sp = sub.add_parser("watch", help="inspect new scans appearing in a folder")
    sp.add_argument("folder")
    sp.add_argument("--db", required=True)
    sp.add_argument("--out", required=True)
    method_args(sp)
    sp.add_argument("--interval", type=float, default=2.0)
    sp.add_argument("--once", action="store_true", help="one pass, then exit")
    sp.add_argument("--material")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("synth", help="write synthetic scans as .3d files with labels")
    sp.add_argument("out")
    sp.add_argument("--seeds", default="0-4")
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("build-dataset", help="cache synthetic train/val/test features")
    sp.add_argument("dir")
    sp.add_argument("--workers", type=int, default=4)
    sp.set_defaults(func=cmd_build_dataset)

    sp = sub.add_parser("train", help="train the U-Net on a cached dataset")
    sp.add_argument("dir")
    sp.add_argument("--model", default="models/unet_synthetic.pt")
    sp.add_argument("--iterations", type=int, default=3000)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--base", type=int, default=16)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="baseline and U-Net against synthetic ground truth")
    sp.add_argument("dir")
    sp.add_argument("--model", default="models/unet_synthetic.pt")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_evaluate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

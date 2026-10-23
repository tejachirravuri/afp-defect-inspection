"""Cached synthetic datasets: features and labels on the analysis grid."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .detect.baseline import BaselineConfig, TapeGeometry, features
from .detect.baseline import detect as baseline_detect
from .detect.unet import make_input
from .preprocess import preprocess
from .synth import SynthConfig, bin_labels, synth_scan

BIN = 4


def build_one(seed: int, out_dir: str) -> str:
    path = Path(out_dir) / ("synth_%05d.npz" % seed)
    if path.is_file():
        return str(path)
    scan = synth_scan(seed, SynthConfig())
    hm = preprocess(scan.counts, scan.cal, BIN)
    geo = TapeGeometry(200.0, 50.8, 8)
    f = features(hm, BaselineConfig(), geo)
    base = baseline_detect(hm, BaselineConfig(), geo, f)
    y = bin_labels(scan.labels, BIN)
    np.savez_compressed(path, x=make_input(f).astype(np.float16), y=y,
                        baseline=base.labels, step=f.step, noise=f.noise,
                        resid=f.resid.astype(np.float16),
                        h=np.nan_to_num(hm.h, nan=-9).astype(np.float16))
    meta = {"seed": seed, "ply_mm": scan.ply_mm, "tape": scan.tape,
            "defects": [{"class": int(d.cls), "kind": d.kind, "rows": d.rows, "cols": d.cols}
                        for d in scan.defects]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return str(path)


def build(seeds: Iterable[int], out_dir, workers: int = 4) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    if workers <= 1:
        return [Path(build_one(s, str(out_dir))) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [Path(p) for p in pool.map(build_one, seeds, [str(out_dir)] * len(seeds))]


SPLITS = {"train": range(1000, 1120), "val": range(2000, 2020), "test": range(3000, 3040)}

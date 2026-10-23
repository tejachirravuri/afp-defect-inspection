"""One call from a .3d file to defects, quality factor and a result record."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .calibration import DEFAULT, Calibration
from .catalogue import NAMES, DefectClass
from .detect import baseline
from .detect.baseline import BaselineConfig, TapeGeometry
from .io import Scan, read_3d
from .preprocess import preprocess
from .quality import class_summary, quality_factor

BIN = 4


@dataclass
class ScanResult:
    file: str
    sha1: str
    method: str
    meta: dict
    sensor: str
    recorded: Optional[str]
    programmed_length_mm: Optional[float]
    programmed_width_mm: Optional[float]
    max_speed: Optional[float]
    edges: dict
    step_mm: float
    noise_mm: float
    tape_area_mm2: float
    q: float
    summary: dict
    defects: List[dict] = field(default_factory=list)
    processing_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry(scan: Scan) -> TapeGeometry:
    tape = scan.program.tape(scan.meta.get("tape", -1)) if scan.program else None
    if tape is None or not tape.fibers:
        return TapeGeometry()
    fibers = [f for f in tape.fibers if f.y_max_mm == f.y_max_mm]  # skip NaN
    if not fibers:
        return TapeGeometry(None, None, len(tape.fibers))
    return TapeGeometry(max(f.length_mm for f in fibers),
                        max(f.y_max_mm for f in fibers) - min(f.y_min_mm for f in fibers),
                        len(fibers))


def _recorded(path: Path) -> Optional[str]:
    """Scan date from a YYYY-MM-DD folder, else the file time."""
    for part in reversed(path.parts):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            return part
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return None


def analyse(path, method: str = "baseline", model=None, cfg: Optional[BaselineConfig] = None,
            cal: Calibration = DEFAULT, scan: Optional[Scan] = None):
    """Returns (ScanResult, HeightMap, Detection)."""
    start = time.perf_counter()
    path = Path(path)
    scan = scan or read_3d(path)
    cfg = cfg or BaselineConfig()
    geo = geometry(scan)
    hm = preprocess(scan, cal, BIN)
    scan_length = scan.shape[0] * cal.dy_mm
    scan_width = scan.shape[1] * cal.dx_mm
    tape = scan.program.tape(scan.meta.get("tape", -1)) if scan.program else None
    # fibre length, not the tape path length (which includes run-in and run-out)
    longer = geo.length_mm is not None and geo.length_mm >= 0.95 * scan_length
    wider = geo.width_mm is not None and geo.width_mm >= scan_width
    if cfg.tape_mode == "auto" and (longer or wider):
        # the course fills the whole view: everything measured is laid material
        cfg = replace(cfg, tape_mode="full")
    if method == "cross_section":
        cfg = replace(cfg, surface_method="cross_section")
    elif method == "unet":
        # the model was trained on residuals to the median surface
        cfg = replace(cfg, surface_method="median")
    f = baseline.features(hm, cfg, geo)
    if method in ("baseline", "cross_section"):
        det = baseline.detect(hm, cfg, geo, f)
    elif method == "unet":
        from .detect import unet
        if model is None:
            raise ValueError("method 'unet' needs a model")
        det = unet.detect(model, hm, f, geo.n_tows)
    else:
        raise ValueError("unknown method %r" % method)

    result = ScanResult(
        file=path.as_posix(), sha1=sha1(path), method=method, meta=dict(scan.meta),
        sensor=scan.sensor, recorded=_recorded(path),
        programmed_length_mm=geo.length_mm, programmed_width_mm=geo.width_mm,
        max_speed=tape.max_speed if tape else None,
        edges=det.edges, step_mm=det.step_mm, noise_mm=det.noise_mm,
        tape_area_mm2=det.tape_area_mm2,
        q=quality_factor(det.defects, det.tape_area_mm2),
        summary=class_summary(det.defects),
        defects=[{"class_id": int(d.cls), "class": NAMES[DefectClass(d.cls)],
                  "area_mm2": round(d.area_mm2, 3), "length_mm": round(d.length_mm, 2),
                  "width_mm": round(d.width_mm, 2), "max_height_mm": round(d.max_height_mm, 3),
                  "x_mm": round(d.x_mm, 2), "y_mm": round(d.y_mm, 2), "tow": d.tow,
                  "bbox": list(d.bbox)} for d in det.defects],
    )
    result.processing_s = round(time.perf_counter() - start, 2)
    return result, hm, det

"""Reading AIS laser scans (.3d), the layup program (.xml) and writing point clouds.

The .3d export of the inspection system holds the raw output of an
Automation Technology C5 laser triangulation camera:

    uint32[4] header: (16, points per profile, profile buffer size, ?)
    uint16[buffer, points] laser line positions, 0 = no measurement
    trailer: camera name, then one chunk record per profile

Only the first R profiles of the buffer are measured, R is part of the file
name (``..._R_1528_...``). Rows after that are left-over buffer content.
"""
from __future__ import annotations

import base64
import gzip
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .calibration import Calibration

HEADER = struct.Struct("<4I")
NAME_FIELDS = {"OF": "program", "P": "ply", "T": "tape", "D": "angle_deg",
               "C": "course", "R": "profiles"}


@dataclass
class Scan:
    path: Path
    counts: np.ndarray          # (profiles, points) uint16, 0 = invalid
    meta: Dict[str, int]
    sensor: str = ""
    program: Optional["LayupProgram"] = None

    @property
    def valid(self) -> np.ndarray:
        return self.counts > 0

    @property
    def shape(self):
        return self.counts.shape


@dataclass
class Fiber:
    number: int
    length_mm: float
    y_min_mm: float
    y_max_mm: float

    @property
    def width_mm(self) -> float:
        return self.y_max_mm - self.y_min_mm


@dataclass
class Tape:
    number: int
    length_mm: float
    max_speed: float
    fibers: List[Fiber] = field(default_factory=list)


@dataclass
class LayupProgram:
    tapes: List[Tape]

    def tape(self, number: int) -> Optional[Tape]:
        return next((t for t in self.tapes if t.number == number), None)


def parse_name(path) -> Dict[str, int]:
    """OF_108_P_1_T_3_D_90_C_0_R_1528_000010 -> {program: 108, ply: 1, ...}."""
    stem = Path(path).stem
    meta = {}
    for key, value in re.findall(r"([A-Z]+)_(-?\d+)", stem):
        if key in NAME_FIELDS:
            meta[NAME_FIELDS[key]] = int(value)
    seq = re.search(r"_(\d{6})$", stem)
    if seq:
        meta["sequence"] = int(seq.group(1))
    return meta


def read_3d(path, profiles: Optional[int] = None, with_program: bool = True) -> Scan:
    path = Path(path)
    data = path.read_bytes()
    hsize, points, buffer_rows, _ = HEADER.unpack_from(data)
    if hsize != 16:
        raise ValueError("%s: unexpected header size %d" % (path, hsize))
    body = hsize + points * buffer_rows * 2
    if len(data) < body:
        raise ValueError("%s: file is truncated" % path)
    counts = np.frombuffer(data, dtype="<u2", count=points * buffer_rows,
                           offset=hsize).reshape(buffer_rows, points)

    meta = parse_name(path)
    rows = profiles or meta.get("profiles") or buffer_rows
    if not 0 < rows <= buffer_rows:
        raise ValueError("%s: %d profiles requested, buffer has %d" % (path, rows, buffer_rows))
    counts = counts[:rows].copy()

    sensor = ""
    trailer = data[body:body + 64]
    m = re.search(rb"(C\d_[\w]+)", trailer)
    if m:
        sensor = m.group(1).decode("ascii", "replace")

    program = None
    xml_path = path.with_suffix(".xml")
    if with_program and xml_path.is_file():
        program = read_program(xml_path)
    return Scan(path, counts, meta, sensor, program)


def write_3d(path, counts: np.ndarray, buffer_rows: Optional[int] = None,
             sensor: str = "C5_2040_GigE") -> None:
    """Writes counts in the same layout as the AIS export (used for synthetic
    scans). Unused buffer rows are zero, the trailer only names the camera."""
    counts = np.asarray(counts, dtype="<u2")
    rows, points = counts.shape
    buffer_rows = max(buffer_rows or rows, rows)
    body = np.zeros((buffer_rows, points), dtype="<u2")
    body[:rows] = counts
    name = sensor.encode("ascii")
    trailer = b"Automation Technology" + struct.pack("<I", len(name)) + name
    with open(path, "wb") as f:
        f.write(HEADER.pack(16, points, buffer_rows, 0))
        f.write(body.tobytes())
        f.write(trailer)


def _decode_contour(text: str) -> np.ndarray:
    """Contours are gzip + base64: 1 byte flag, uint32 byte count, then
    float32 records of (x, y, z, nx, ny, nz) in metres."""
    raw = gzip.decompress(base64.b64decode(text))
    nbytes = struct.unpack_from("<I", raw, 1)[0]
    values = np.frombuffer(raw, dtype="<f4", count=nbytes // 4, offset=5)
    return values.reshape(-1, 6)


def read_program(path) -> LayupProgram:
    root = ET.parse(path).getroot()
    tapes = []
    for tape_el in root.iter("CReducedTape"):
        tape = Tape(int(tape_el.get("NumTape")), float(tape_el.get("Length")),
                    float(tape_el.get("MaxSpeed", "nan")))
        for fiber_el in tape_el.iter("CReducedFiber"):
            contour_el = fiber_el.find("DataContour")
            y_min = y_max = float("nan")
            if contour_el is not None and contour_el.text:
                pts = _decode_contour(contour_el.text.strip())
                y_min, y_max = float(pts[:, 1].min()) * 1000.0, float(pts[:, 1].max()) * 1000.0
            tape.fibers.append(Fiber(int(fiber_el.get("NumFiber")),
                                     float(fiber_el.get("Length")), y_min, y_max))
        tapes.append(tape)
    return LayupProgram(tapes)


def to_points(scan: Scan, cal: Calibration, step: int = 1) -> np.ndarray:
    """Valid measurements as an (N, 3) array in millimetres."""
    counts = scan.counts[::step, ::step]
    rows, cols = np.nonzero(counts)
    z = counts[rows, cols].astype(np.float64) * cal.dz_mm
    return np.column_stack([cols * step * cal.dx_mm, rows * step * cal.dy_mm, z])


def write_ply(path, points: np.ndarray, values: Optional[np.ndarray] = None) -> None:
    """Binary little endian PLY, readable by CloudCompare and Open3D.
    ``values`` (e.g. defect class per point) is stored as a scalar field."""
    points = np.asarray(points, dtype="<f4")
    header = ["ply", "format binary_little_endian 1.0",
              "element vertex %d" % len(points),
              "property float x", "property float y", "property float z"]
    if values is not None:
        header.append("property float scalar_class")
        points = np.column_stack([points, np.asarray(values, dtype="<f4")])
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(points, dtype="<f4").tobytes())


def read_ply(path) -> np.ndarray:
    """Reads binary or ascii PLY files with float x, y, z vertices (e.g. an
    ATOS export). Returns (N, k) with all vertex properties."""
    with open(path, "rb") as f:
        header, props, count, fmt = [], [], 0, "ascii"
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                count = int(line.split()[2])
            elif line.startswith("property") and count:
                props.append(line.split()[1])
            elif line == "end_header":
                break
        if fmt == "ascii":
            return np.loadtxt(f, max_rows=count, ndmin=2)
        types = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1",
                 "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([("p%d" % i, endian + types[t]) for i, t in enumerate(props)])
        arr = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
        return np.column_stack([arr[name].astype(np.float64) for name in dtype.names])

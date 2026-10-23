"""SQLite result database: one row per inspected tape course, one per defect.

Columns follow the parameter list of the work plan: time, material, speed,
orientation, and from the inspection the number, type and area of defects
and the quality factor Q.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .catalogue import CATALOGUE

SCHEMA = """
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    weight REAL
);
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file TEXT NOT NULL,
    sha1 TEXT NOT NULL,
    method TEXT NOT NULL,
    program INTEGER, ply INTEGER, tape INTEGER, angle_deg INTEGER,
    course INTEGER, profiles INTEGER, sequence INTEGER,
    recorded TEXT,
    material TEXT,
    max_speed REAL,
    sensor TEXT,
    programmed_length_mm REAL, programmed_width_mm REAL,
    length_mm REAL, width_mm REAL,
    step_mm REAL, noise_mm REAL, tape_area_mm2 REAL,
    q REAL, n_defects INTEGER,
    processed_at TEXT, processing_s REAL,
    UNIQUE (sha1, method)
);
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    area_mm2 REAL, length_mm REAL, width_mm REAL, max_height_mm REAL,
    x_mm REAL, y_mm REAL, tow INTEGER
);
CREATE INDEX IF NOT EXISTS defects_scan ON defects(scan_id);
"""


def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    con.executemany("INSERT OR REPLACE INTO classes VALUES (?, ?, ?, ?)",
                    [(int(c), i.name, i.description, i.weight) for c, i in CATALOGUE.items()])
    con.commit()
    return con


def known(con, sha1: str, method: str) -> bool:
    return con.execute("SELECT 1 FROM scans WHERE sha1 = ? AND method = ?",
                       (sha1, method)).fetchone() is not None


def insert(con, result, material: Optional[str] = None) -> int:
    """Insert or replace the result of one scan, returns the scan id."""
    m = result.meta
    con.execute("DELETE FROM scans WHERE sha1 = ? AND method = ?", (result.sha1, result.method))
    cur = con.execute(
        """INSERT INTO scans (file, sha1, method, program, ply, tape, angle_deg, course,
               profiles, sequence, recorded, material, max_speed, sensor,
               programmed_length_mm, programmed_width_mm, length_mm, width_mm,
               step_mm, noise_mm, tape_area_mm2, q, n_defects, processed_at, processing_s)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (result.file, result.sha1, result.method, m.get("program"), m.get("ply"), m.get("tape"),
         m.get("angle_deg"), m.get("course"), m.get("profiles"), m.get("sequence"),
         result.recorded, material, result.max_speed, result.sensor,
         result.programmed_length_mm, result.programmed_width_mm,
         result.edges.get("length_mm"), result.edges.get("width_mm"),
         result.step_mm, result.noise_mm, result.tape_area_mm2, result.q, len(result.defects),
         datetime.now().isoformat(timespec="seconds"), result.processing_s))
    scan_id = cur.lastrowid
    con.executemany(
        """INSERT INTO defects (scan_id, class_id, area_mm2, length_mm, width_mm,
               max_height_mm, x_mm, y_mm, tow) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(scan_id, d["class_id"], d["area_mm2"], d["length_mm"], d["width_mm"],
          d["max_height_mm"], d["x_mm"], d["y_mm"], d["tow"]) for d in result.defects])
    con.commit()
    return scan_id


SUMMARY = """
SELECT s.method, c.name, COUNT(d.id) AS n, ROUND(COALESCE(SUM(d.area_mm2), 0), 1) AS area_mm2
FROM scans s CROSS JOIN classes c
LEFT JOIN defects d ON d.scan_id = s.id AND d.class_id = c.id
WHERE c.id > 0
GROUP BY s.method, c.id ORDER BY s.method, c.id
"""


def summary(con):
    return con.execute(SUMMARY).fetchall()

"""Watch a folder for new scans and inspect them as they arrive.

This stands in for the online integration with the inspection system: the
AIS exports each tape course as a .3d/.xml pair into a folder, this loop
picks it up once the file has stopped growing, runs the inspection, writes a
JSON report next to a results folder and adds the result to the database.
A real cell controller would read Q and the defect list from there (or get
them pushed over OPC UA, which is not part of this code).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from . import database
from .pipeline import analyse, sha1

log = logging.getLogger(__name__)


def _stable(path: Path, sizes: dict) -> bool:
    size = path.stat().st_size
    stable = sizes.get(path) == size
    sizes[path] = size
    return stable


def watch(folder, db_path, out_dir, method="baseline", model=None, interval=2.0,
          once=False, material=None, q_alarm=0.95):
    folder, out_dir = Path(folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = database.connect(db_path)
    sizes = {}
    log.info("watching %s (method %s)", folder, method)
    while True:
        for path in sorted(folder.rglob("*.3d")):
            try:
                if not once and not _stable(path, sizes):
                    continue
                digest = sha1(path)
                if database.known(con, digest, method):
                    continue
                result, _, _ = analyse(path, method, model)
                database.insert(con, result, material)
                (out_dir / (path.stem + "_%s.json" % method)).write_text(
                    json.dumps(result.to_dict(), indent=1))
                level = logging.WARNING if result.q < q_alarm else logging.INFO
                log.log(level, "%s  Q=%.3f  defects=%d  (%.1fs)", path.name, result.q,
                        len(result.defects), result.processing_s)
            except (OSError, ValueError) as e:
                log.error("%s: %s", path.name, e)
        if once:
            break
        time.sleep(interval)
    con.close()

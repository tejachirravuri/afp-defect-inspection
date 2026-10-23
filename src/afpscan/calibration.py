"""Scale of the scan grid.

The .3d files carry no calibration. The defaults come from the data itself:

- dx: the layup program says a tape is 8 tows of 6.35 mm (50.8 mm); in the
  scans that tape covers about 1566 of the 2048 points per profile.
- dy: the programmed tow length is 200 mm and covers about 795 profiles.
- dz: the camera outputs the laser line position with 6 sub-pixel bits. The
  tape stands about 120 counts above the tool; with an assumed ply thickness
  of 0.18 mm that gives 0.0015 mm per count. This one is an assumption, not a
  measurement. Replace it once the sensor calibration is known.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    dx_mm: float = 0.0324   # along the laser line
    dy_mm: float = 0.2516   # between profiles (travel direction)
    dz_mm: float = 0.0015   # per height count (assumed)

    def binned(self, factor: int) -> "Calibration":
        return Calibration(self.dx_mm * factor, self.dy_mm, self.dz_mm)


DEFAULT = Calibration()

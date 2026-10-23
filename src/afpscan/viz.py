"""Figures: height map with the tape outline and defect overlay."""
from __future__ import annotations

import numpy as np

from .catalogue import CATALOGUE, DEFECTS, DefectClass


def _cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap([(0, 0, 0, 0)] + [CATALOGUE[c].color for c in DEFECTS])


def overlay(ax, hm, labels=None, tape=None, title="", vmin=None, vmax=None, smooth=3):
    """Draws the scan with travel direction left to right, as on the machine.
    The background is the height map after a small median filter."""
    from scipy import ndimage as ndi
    cal = hm.cal
    extent = (0, hm.h.shape[0] * cal.dy_mm, hm.h.shape[1] * cal.dx_mm, 0)
    h = hm.h
    if smooth:
        fill = np.nanmedian(h)
        h = ndi.median_filter(np.nan_to_num(h, nan=fill), size=smooth)
        h[~hm.valid] = np.nan
    lo, hi = np.nanpercentile(h, [1, 99])
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    ax.imshow(np.nan_to_num(h, nan=vmin).T, cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto", extent=extent, interpolation="nearest")
    if tape is not None and tape.any():
        xs = (np.arange(tape.shape[0]) + 0.5) * cal.dy_mm
        ys = (np.arange(tape.shape[1]) + 0.5) * cal.dx_mm
        ax.contour(xs, ys, tape.T.astype(float), [0.5], colors="#00bcd4", linewidths=0.8)
    if labels is not None:
        ax.imshow(np.ma.masked_equal(labels, 0).T, cmap=_cmap(), vmin=0, vmax=len(DefectClass) - 1,
                  aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel("travel direction [mm]")
    ax.set_ylabel("across tape [mm]")
    if title:
        ax.set_title(title, fontsize=9)


def legend(ax):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CATALOGUE[c].color, label="%d %s" % (c, CATALOGUE[c].name))
                       for c in DEFECTS], loc="upper right", fontsize=7, ncol=5, framealpha=0.8)


def save_overlay(path, hm, det, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3.6))
    overlay(ax, hm, det.labels, det.tape, title)
    legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

"""Small U-Net that labels every cell of the analysis grid with a defect class.

Input channels (all from ``baseline.features``, heights in units of the tape
step so the unknown height calibration drops out):

    0  residual to the local tape surface / step, clipped to [-4, 6]
    1  height above the tool / step, clipped to [-2, 6]
    2  valid measurement (0/1)
    3  tape area (0/1)

Needs torch (``pip install afpscan[dl]``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # the rule based detector works without torch
    torch = None
    nn = None

from ..catalogue import DefectClass
from .baseline import Detection, Features, defects_from_labels, labels_from_defects, tape_edges

N_CLASSES = len(DefectClass)
IN_CHANNELS = 4


def make_input(f: Features) -> np.ndarray:
    step = max(f.step, 1e-3)
    valid = f.valid.astype(np.float32)
    resid = np.clip(np.nan_to_num(f.resid) / step, -4, 6) * valid
    rel = np.clip(np.nan_to_num(f.rel) / step, -2, 6) * valid
    return np.stack([resid, rel, valid, f.tape.astype(np.float32)]).astype(np.float32)


def _require_torch():
    if torch is None:
        raise ImportError("the U-Net needs torch: pip install torch")


if nn is not None:
    class _Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.body(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=IN_CHANNELS, n_classes=N_CLASSES, base=16, depth=3):
            super().__init__()
            chans = [base * 2 ** i for i in range(depth + 1)]
            self.down = nn.ModuleList()
            cin = in_channels
            for c in chans[:-1]:
                self.down.append(_Block(cin, c))
                cin = c
            self.bottom = _Block(chans[-2], chans[-1])
            self.up = nn.ModuleList()
            self.dec = nn.ModuleList()
            for c_hi, c_lo in zip(chans[::-1][:-1], chans[::-1][1:]):
                self.up.append(nn.ConvTranspose2d(c_hi, c_lo, 2, stride=2))
                self.dec.append(_Block(2 * c_lo, c_lo))
            self.head = nn.Conv2d(chans[0], n_classes, 1)
            self.depth = depth

        def forward(self, x):
            skips = []
            for block in self.down:
                x = block(x)
                skips.append(x)
                x = F.max_pool2d(x, 2)
            x = self.bottom(x)
            for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
                x = dec(torch.cat([up(x), skip], dim=1))
            return self.head(x)


# --- data ------------------------------------------------------------------

class PatchSampler:
    """Random patches from cached scans, centred on defects most of the time."""

    def __init__(self, files: List[Path], size=(128, 128), defect_prob=0.7, seed=0):
        self.items = []
        for f in files:
            d = np.load(f)
            self.items.append((d["x"].astype(np.float32), d["y"].astype(np.int64)))
        self.size = size
        self.defect_prob = defect_prob
        self.rng = np.random.default_rng(seed)
        self.defect_cells = [np.argwhere(y > 0) for _, y in self.items]
        self.tape_cells = [np.argwhere(x[3] > 0) for x, _ in self.items]

    def patch(self):
        i = int(self.rng.integers(len(self.items)))
        x, y = self.items[i]
        ph, pw = self.size
        cells = self.defect_cells[i] if (len(self.defect_cells[i]) and
                                         self.rng.random() < self.defect_prob) else self.tape_cells[i]
        r, c = cells[int(self.rng.integers(len(cells)))]
        r0 = int(np.clip(r - self.rng.integers(ph), 0, x.shape[1] - ph))
        c0 = int(np.clip(c - self.rng.integers(pw), 0, x.shape[2] - pw))
        xp, yp = x[:, r0:r0 + ph, c0:c0 + pw].copy(), y[r0:r0 + ph, c0:c0 + pw].copy()
        if self.rng.random() < 0.5:                      # flip across the tape
            xp, yp = xp[:, :, ::-1], yp[:, ::-1]
        if self.rng.random() < 0.5:                      # flip along travel
            xp, yp = xp[:, ::-1], yp[::-1]
        return np.ascontiguousarray(xp), np.ascontiguousarray(yp)

    def batch(self, n):
        xs, ys = zip(*(self.patch() for _ in range(n)))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def class_weights(files, cap=20.0):
    counts = np.zeros(N_CLASSES)
    for f in files:
        y = np.load(f)["y"]
        counts += np.bincount(y.ravel(), minlength=N_CLASSES)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    w = np.minimum(w / w[0], cap)
    return w.astype(np.float32)


def _dice_loss(logits, target, n_classes):
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + 1) / (union + 1)
    return 1 - dice[1:].mean()


@dataclass
class TrainConfig:
    iterations: int = 3000
    batch: int = 16
    lr: float = 2e-3
    patch: tuple = (128, 128)
    base: int = 16
    seed: int = 0
    log_every: int = 100


def train(train_files, val_files, out: Path, cfg: TrainConfig = TrainConfig(), log=print):
    _require_torch()
    torch.manual_seed(cfg.seed)
    sampler = PatchSampler(list(train_files), cfg.patch, seed=cfg.seed)
    val_sampler = PatchSampler(list(val_files), cfg.patch, seed=cfg.seed + 1)
    val_batches = [val_sampler.batch(cfg.batch) for _ in range(8)]
    weights = torch.from_numpy(class_weights(train_files))
    log("class weights: " + ", ".join("%s %.1f" % (c.name.lower(), w)
                                     for c, w in zip(DefectClass, weights)))

    model = UNet(base=cfg.base)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.iterations)
    history = []
    best = float("inf")
    start = time.time()
    for it in range(1, cfg.iterations + 1):
        model.train()
        x, y = sampler.batch(cfg.batch)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=weights) + _dice_loss(logits, y, N_CLASSES)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % cfg.log_every == 0 or it == cfg.iterations:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([(F.cross_entropy(model(vx), vy, weight=weights)
                                     + _dice_loss(model(vx), vy, N_CLASSES)).item()
                                    for vx, vy in val_batches]))
            history.append({"iteration": it, "train_loss": float(loss.item()), "val_loss": vl})
            log("iter %5d  train %.3f  val %.3f  (%.0fs)" % (it, loss.item(), vl, time.time() - start))
            if vl < best:
                best = vl
                save(model, out, cfg)
    return history


def save(model, path: Path, cfg: Optional[TrainConfig] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "base": getattr(cfg, "base", 16)}, path)


def load(path: Path):
    _require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(base=ckpt.get("base", 16))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def predict_labels(model, f: Features) -> np.ndarray:
    """Same inspection area as the rule based detector: material defects only
    inside the tape minus the edge band, missing data anywhere in the tape."""
    labels = predict_array(model, make_input(f))
    labels[~f.tape] = 0
    labels[~f.inner & (labels != int(DefectClass.NO_DATA))] = 0
    return labels


def predict_array(model, x: np.ndarray) -> np.ndarray:
    """Class per cell for an input array (channels, rows, cols)."""
    _require_torch()
    x = np.asarray(x, dtype=np.float32)
    _, rows, cols = x.shape
    m = 2 ** model.depth
    pr, pc = (-rows) % m, (-cols) % m
    xt = torch.from_numpy(np.pad(x, ((0, 0), (0, pr), (0, pc)), mode="edge"))[None]
    with torch.no_grad():
        logits = model(xt)[0, :, :rows, :cols]
    labels = logits.argmax(dim=0).numpy().astype(np.uint8)
    labels[x[3] <= 0] = 0
    return labels


def detect(model, hm, f: Features, n_tows=8, min_area_mm2=1.0) -> Detection:
    labels = predict_labels(model, f)
    defects = defects_from_labels(labels, f, hm, n_tows, min_area_mm2=min_area_mm2)
    return Detection(labels_from_defects(labels, defects), f.tape, defects, f.step,
                     float(f.tape.sum() * hm.cell_area_mm2), tape_edges(f.tape, hm.cal), f.noise)

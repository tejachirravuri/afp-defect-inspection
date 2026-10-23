# Method

## 1. Reading and gridding

`read_3d` keeps the first R profiles (R from the file name). Points with value
0 are invalid. Four neighbouring points along the laser line are averaged
into one cell, which gives an analysis grid of about 0.13 mm across by
0.25 mm along the travel direction and halves the point noise.

## 2. Height above the tool

A plane is fitted to the raw counts with RANSAC (`preprocess.tool_plane`). If
the tape covers most of the view, the largest plane is the tape top; this is
caught by checking for many points clearly below it and fitting again on
those.

The tool is not flat. Along the travel direction it undulates by about
±0.15 mm, across it has a slight crown, so a plane alone leaves the tool
partly above half a ply. The detector therefore fits a second model on top:
a per profile offset plus a per column offset, estimated with medians on the
points classified as tool (Otsu threshold), three iterations. The first
estimate uses the 10th percentile of each profile, which next to the tape is
the tool level.

## 3. Tape area and step

With the layup program the tape size is known: 200 mm long, 8 x 6.35 mm
wide (or 4 tows for half tapes).

- Along the travel direction the window of programmed length with the most
  material is chosen, then the add and cut ends are refined to where the
  height jumps are largest.
- Across, the window of programmed width with the most material, extended
  one tow at a time over material laid right next to it (in two scans a
  4-tow course was laid next to an earlier one), then shifted by up to 2 mm
  to where the height jumps at both tape edges are largest.
- Without a program: the largest region that is mostly material.

The tape step (height of one ply) is the median tape height minus the median
tool height right next to the tape.

If the programmed fibres are at least as long as the scan, or the programmed
course is wider than the field of view (the 700 mm multi course export), the
whole valid area is treated as laid material.

A band of 1 mm along both tape edges and 2 mm at the add and cut ends is
not inspected for material defects. The tow edges are rounded over about
that width and the ends are only located to within a few profiles, so every
method would read bare tool there as a gap. The edge and end positions
themselves are measured and stored with the result.

## 4. Rule based detection (baseline)

- Local tape surface: median over about 10 mm along and 15 mm across, so a
  missing tow or a twist (both about one tow wide) cannot pull it along.
  Computed on a 4 x decimated grid and interpolated back.
- Alternative surface (`--method cross_section`): the first step of the
  cross-sectional line processing of Tang et al. (2021), which I had started
  implementing on exported point clouds in 2024. Every laser profile is split
  into n_tows / 2 + 1 groups, only heights between the 25 % and 75 %
  quantiles of each group are kept, a quadratic is fitted to them and fitted
  again without points more than 2.5 sigma off. Everything after that is the
  same as for the median surface.
- Residual = height - surface. Noise = robust standard deviation (MAD) of
  the residual inside the tape.
- Thresholds: a gap needs the residual below `-max(0.5 step, 3.5 noise)`,
  raised material above `max(0.75 step, 3.5 noise)`. Hysteresis: regions are
  grown at 60 % of these (or 2.5 noise) and kept if they contain at least 12
  cells above the full threshold. This keeps whole defects together without
  picking up noise.
- No data: invalid cells inside the tape.
- The kept cells are grouped with DBSCAN (0.8 mm) and each group classified
  by size and shape (see the catalogue).

## 5. U-Net

Input channels on the analysis grid: residual / step, height / step,
validity and tape mask. Dividing by the step makes the input independent of
the unknown height calibration.

- 3 levels, 16 to 128 channels, about 0.48 M parameters.
- Trained on 128 x 128 patches of 120 synthetic scans, 70 % of patches
  centred on a defect, random flips, cross entropy with class weights plus
  dice loss, AdamW with one-cycle schedule, CPU only.
- Prediction on the full grid, cells outside the tape set to 0, then the
  same DBSCAN grouping and minimum area as the baseline.

## 6. Synthetic scans

`afpscan.synth` builds raw counts in the AIS layout. Tool waviness, crown,
plane tilt, ply height 0.17 to 0.30 mm, fibre striations, point noise,
correlated speckle (tuned so the measured noise matches the real scans,
0.05 to 0.08 mm), outliers, random dropouts and dropout streaks. Defects:
gaps (0.4 to 2.5 mm wide), missing tows, overlaps, twists modelled after the
real one (a raised lens on the tow with gaps on both sides), transverse
folds, foreign objects, no data areas with or without a raised rim.

Ground truth is the drawn defect mask binned like the analysis grid (a cell
gets a class if at least half its points have it).

## 7. Quality factor

    Q = 1 - sum_c (w_c * A_c) / A_tape

with the class weights from the catalogue. Q = 1 is a clean course. The
weights are a proposal; they should eventually come from mechanical tests.

## 8. Metrics

- Pixel level, inside the tape: IoU, precision and recall per class.
- Object level: a ground truth defect (8-connected component of at least
  1 mm²) is found if a prediction of the same class touches it within two
  cells; a predicted object is correct if it touches ground truth of its
  class. `any_defect` ignores the class.
- Mean absolute error of Q against Q computed from the ground truth.

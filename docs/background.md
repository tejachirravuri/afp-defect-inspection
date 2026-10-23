# Background

From autumn 2023 I worked as a student research assistant (HiWi) at
Fraunhofer IGCV in Augsburg, in the group working on automated fiber
placement (AFP) for aerospace parts. In AFP a robot head lays several narrow
carbon fibre tows side by side, course after course, onto a tool. Every
course is scanned right after layup by a laser line sensor (the automated
inspection system, AIS), and defects such as gaps, overlaps, twisted tows or
debris have to be found before the next ply goes on top.

My working title was *Integrating AI-based defect detection and quality
assessment with laser and ATOS sensors for AFP in aerospace applications*.
The idea was a standard inspection routine with as little manual work as
possible. The plan had six tasks:

1. **Standard defect classes 0 to 5.** Earlier work in the group used
   different class lists, so labelled data could not be shared between
   methods.
2. **Laser line sensor.** Understand the sensor and its point clouds, set up
   the earlier segmentation approaches (RANSAC to separate tows from the
   table and DBSCAN to group defect points; normal vector and histogram
   thresholds, which was the benchmark), review deep learning methods for
   3D point clouds and build a learning method that is compared against the
   benchmark.
3. **ATOS (GOM) scanner.** Check whether the same labelling and model work on
   ATOS point clouds, possibly fuse both sensors.
4. **Defect area, number of defects and a quality factor Q.**
5. **Deployment** as a quality check software next to the AIS.
6. **Final database** with process parameters (time, material, speed,
   orientation) and the inspection results.

## What this repository covers

| Task | Status here |
|---|---|
| 1 defect classes | done: [defect_catalogue.md](defect_catalogue.md), `afpscan.catalogue` |
| 2 laser sensor | reader for the raw AIS export and the layup program, point cloud export for CloudCompare, rule based detector built on the earlier ideas (RANSAC plane, DBSCAN grouping), U-Net trained on synthetic scans, comparison on synthetic ground truth and on the real scans |
| 3 ATOS | not done, no ATOS data available. `afpscan.io.read_ply` reads ATOS .ply exports, but the detector works on the gridded laser data and would need a resampling step first |
| 4 area, count, Q | done: per defect area, length, width, height and tow; Q per course |
| 5 deployment | partly: `afpscan watch` picks up new scans from an export folder, inspects them and writes JSON and database rows. The communication with the cell controller (e.g. OPC UA) is not part of this |
| 6 database | done: SQLite schema with scans, defects and the class table |

## Why synthetic training data

The scans come without labels, and twelve courses are far too few to train
anything. Labelling 3D laser data by hand is slow and was one of the open
points of the plan. So the U-Net is trained on synthetic scans that copy the
sensor format and the noise and surface statistics of the real ones, with
the defects drawn in at known positions. The same synthetic data gives a
ground truth for comparing the two methods. Whether the model transfers to
real scans can only be judged by eye on the twelve real courses, see
[results.md](results.md).

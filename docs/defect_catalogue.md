# Defect catalogue (classes 0 to 5)

The first task in my work plan was a common set of defect classes. Before,
each method in the group had its own: one used three classes, another four
(wrinkles, overlaps, bridges, ...), without shared labels, so data labelled
for one method could not be used for another.

The catalogue below maps the terms of the AIS manual and of the literature
onto six ids. All labels, detector outputs and database rows use them
(`afpscan.catalogue`).

| Id | Name | What it is | AIS terms | Literature terms | Weight in Q |
|---|---|---|---|---|---|
| 0 | ok | tape or tool surface without findings | | | 0 |
| 1 | gap | material missing between or instead of tows | GAPS | gap, narrow / wide gap, missing tow, lap gap | 1 |
| 2 | overlap | a tow lying partly on its neighbour (one extra ply along a tow edge) | | overlap, lap | 1 |
| 3 | out_of_plane | tow lifted out of the layup plane | TWISTS, FOLDINGS, Z-defects | twist, fold, wrinkle, pucker, bridging, blister, pull-up | 2 |
| 4 | foreign_object | small raised spot not belonging to a tow | CAULIFLOWERS | FOD, fuzzball, resin ball | 2 |
| 5 | no_data | no valid measurement inside the tape area | loss of information | occlusion, missing data | 0.5 |

Notes:

- Class 5 is not a material defect. It is there because an area without
  data cannot be released. In the scans it often sits right next to a
  pile-up of material that shadows the laser.
- Ply edge positions (ADD, CUT, LEFT, RIGHT in the AIS) are not a class.
  They are measured for every course and stored with the result, so they can
  be compared with the program.
- How a defect is told apart by the rule based detector:

| Found as | Becomes |
|---|---|
| dip below the local tape surface, at least 3 mm long and 0.4 mm wide | gap |
| raised, at least 15 mm long, at most 4 mm wide, about one ply high | overlap |
| raised, at most 4 mm in both directions | foreign_object |
| any other raised region | out_of_plane |
| no measurement inside the tape | no_data |

- Old data labelled with other class lists can be mapped onto these ids with
  a small table (e.g. "wrinkle" -> 3, "bridge" -> 3, "overlap" -> 2), which
  is how earlier data sets can be made usable again.

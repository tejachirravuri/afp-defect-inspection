# Scan data

Laser line scans of single AFP tape courses, recorded on 2023-04-05 at
Fraunhofer IGCV (Augsburg) with the automated inspection system (AIS) of the
fiber placement cell: an EDIXIA AF-Inspect head with an Automation Technology
C5-2040 GigE laser triangulation camera. The data belongs to Fraunhofer IGCV.
It is kept in this private repository for my own work and is not licensed for
redistribution.

| Folder | Content |
|---|---|
| `raw/test025_2023-04-05/` | 11 scans of 200 mm long courses (test series 025), each with its layup program (`.xml`) |
| `raw/export/` | one 700 mm curved course laid over earlier plies, with program and a VRML export of points picked in the AIS software |

File names: `OF_108_P_1_T_3_D_90_C_0_R_1528_000010`

| Field | Meaning (as far as I could work it out) |
|---|---|
| `OF_108` | layup program 108 |
| `P_1` | ply 1 |
| `T_3` | tape (course) number in the program |
| `D_90` | ply orientation in degrees |
| `C_0` | always 0 in this data, probably a course or camera index |
| `R_1528` | number of recorded laser profiles |
| `000010` | running number of the export |

What can be seen in the scans:

| Scan | Findings |
|---|---|
| 000002, 000003, 000004, 000005 | twisted tow on tow 6 (counted from the upper edge of the scan), with gaps on both sides of it |
| 000007, 000008, 000009 | long gap between tows 4 and 5 over the full course |
| 000001, 000006, 000007 | material pile-up at the start of the course with no measurement in its shadow |
| 000010, 000011 | strong data loss all over the tape (different surface, or sensor exposure) |
| 000001 | the first 140 mm of the scan have no data at all |
| 2787_000011 | multi-course layup, the course runs through the whole scan |

The `.3d` format is described in [../docs/data_format.md](../docs/data_format.md).

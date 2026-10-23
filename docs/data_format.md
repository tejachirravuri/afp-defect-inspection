# AIS export formats

There was no documentation of the export files, so this is reverse
engineered from the files themselves and the C5 camera manual.

## `.3d` scan

Little endian binary:

| Offset | Type | Content |
|---|---|---|
| 0 | uint32 | header size, always 16 |
| 4 | uint32 | points per profile (2048, the sensor width) |
| 8 | uint32 | profile buffer size (6000) |
| 12 | uint32 | unknown |
| 16 | uint16[buffer][points] | laser line position per point, 0 = no measurement |
| after | bytes | trailer: `Automation Technology`, camera name (`C5_2040_GigE`), then one chunk record per profile (frame counter, encoder position, ...) |

Only the first `R` profiles of the buffer are measured, `R` is in the file
name. The rows after that are left-over buffer content and must be ignored.
`afpscan.io.read_3d` does this.

The values are the position of the laser line on the sensor with sub-pixel
bits (the camera outputs 16 bit in 3D mode). They are proportional to height
but the scale depends on the triangulation geometry, which is not in the file.

## Scale

| Axis | Value | Source |
|---|---|---|
| across the tape (x) | 0.0324 mm per point | programmed 8 x 6.35 mm tape covers about 1566 points |
| along travel (y) | 0.2516 mm per profile | programmed 200 mm tow covers about 795 profiles; the 700 mm course covers 2787 |
| height (z) | 0.0015 mm per count | **assumed**: tape step of about 120 to 200 counts with a ply of about 0.18 to 0.3 mm |

Because the height scale is not known, every detection threshold is defined
relative to the tape step measured in the scan itself, and the reported
heights in mm should be read as approximate.

## `.xml` layup program

```
CReducedProgram / Plys / CReducedPly / Tapes / CReducedTape (NumTape, Length, MaxSpeed)
    Fibers / CReducedFiber (NumFiber, Length)
        DataContour   gzip + base64
        Segments / CReducedFiberSegment (AddPathPosition, CutPathPosition)
    Targets / CTarget (X, Y, Z, Rx, Ry, Rz)   robot path points
    DataContour, CenterLine                    gzip + base64
```

Decoded contours: one flag byte, a uint32 byte count, then float32 records of
`(x, y, z, nx, ny, nz)` in metres. For the 200 mm courses each fibre contour
is a rectangle of 200 x 6.35 mm, which gives tow width and tape length.
`afpscan.io.read_program` returns them per tape and fibre.

## `.wrl`

VRML 2.0 with small cylinders at picked 3D points, exported by the AIS
software. Not used here.

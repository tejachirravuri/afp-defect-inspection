| class | baseline IoU | baseline recall | baseline precision | cross_section IoU | cross_section recall | cross_section precision | unet IoU | unet recall | unet precision |
|---|---|---|---|---|---|---|---|---|---|
| gap | 0.55 | 0.90 (35/39) | 0.98 | 0.65 | 0.95 (37/39) | 0.88 | 0.70 | 0.95 (37/39) | 0.34 |
| overlap | 0.24 | 0.56 (5/9) | 0.75 | 0.38 | 0.67 (6/9) | 0.89 | 0.82 | 1.00 (9/9) | 0.31 |
| out_of_plane | 0.39 | 0.82 (23/28) | 0.35 | 0.41 | 0.68 (19/28) | 0.67 | 0.74 | 0.96 (27/28) | 0.53 |
| foreign_object | 0.16 | 0.73 (8/11) | 0.10 | 0.29 | 0.73 (8/11) | 0.21 | 0.67 | 0.73 (8/11) | 1.00 |
| no_data | 0.98 | 1.00 (8/8) | 1.00 | 0.98 | 1.00 (8/8) | 1.00 | 0.98 | 1.00 (8/8) | 1.00 |
| any_defect | 0.56 | 0.93 (82/88) | 0.62 | 0.64 | 0.90 (79/88) | 0.85 | 0.74 | 0.95 (84/88) | 0.40 |

Mean absolute error of Q: baseline 0.0093, cross_section 0.0089, unet 0.0060

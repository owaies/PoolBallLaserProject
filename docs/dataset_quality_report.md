# Dataset Quality Audit Report

This report was automatically generated to analyze the health, consistency, and annotations of the merged pool ball detection dataset.

## Overview Table
| Split | Total Images | Total Bounding Boxes | Corrupt Images | Missing Labels | Background Images | Out of Bounds Coordinates | Wrong Class IDs | Duplicate Labels |
|---|---|---|---|---|---|---|---|---|
| **Train** | 5234 | 26093 | 0 | 0 | 1287 | 451 | 0 | 0 |
| **Valid** | 1495 | 7653 | 0 | 0 | 363 | 159 | 0 | 0 |
| **Test** | 749 | 3556 | 0 | 0 | 190 | 44 | 0 | 0 |

## Wrong Class IDs Analysis
✓ **No invalid class IDs found.** All annotations lie strictly within the 0-15 range.

## Duplicate Bounding Box Detections
✓ **No duplicate annotations found in the dataset.**

## Corrupted Images
✓ **No corrupted images found.** All image headers parsed successfully.

## Class Distribution and Imbalance Analysis
| Class ID | Class Name | Instances | Percentage |
|---|---|---|---|
| 0 | cue_ball | 3737 | 10.02% |
| 1 | 1_ball | 2351 | 6.30% |
| 2 | 2_ball | 1993 | 5.34% |
| 3 | 3_ball | 3003 | 8.05% |
| 4 | 4_ball | 2513 | 6.74% |
| 5 | 5_ball | 3431 | 9.20% |
| 6 | 6_ball | 2279 | 6.11% |
| 7 | 7_ball | 3378 | 9.06% |
| 8 | 8_ball | 3068 | 8.22% |
| 9 | 9_ball | 3990 | 10.70% |
| 10 | 10_ball | 1285 | 3.44% |
| 11 | 11_ball | 1224 | 3.28% |
| 12 | 12_ball | 1114 | 2.99% |
| 13 | 13_ball | 1264 | 3.39% |
| 14 | 14_ball | 1066 | 2.86% |
| 15 | 15_ball | 1606 | 4.31% |

### Class Balance Metrics
- **Mean Instances per Class:** 2331.4
- **Standard Deviation:** 976.1
- **Most Populated Class:** Class 9 (9_ball) with 3990 instances.
- **Least Populated Class:** Class 14 (14_ball) with 1066 instances.
- **Class Imbalance Ratio (Max/Min):** 3.74

# Camera Calibration Report

## Settings
- **Checkerboard Inner Corners:** (9, 6)
- **Square Size:** 24.0 mm
- **Images Attempted:** 10
- **Images Used (Valid Corners):** 10

## Calibration Results
- **Mean Reprojection Error:** `0.0006` pixels

### Camera Matrix
```python
[[709437.6621,      0.    ,    399.2575],
 [     0.    , 709533.9395,    299.3865],
 [     0.    ,      0.    ,      1.    ]]
```

### Distortion Coefficients
```python
[[-0.0267, -0.    ,  0.0025,  0.0019, -0.    ]]
```

## Recommendations
- Calibration error is excellent (< 0.5px). Proceed to coordinate transformation.

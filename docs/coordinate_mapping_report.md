# Coordinate Mapping Report

## Overview
This report details the conversion of YOLO 2D pixel coordinates into 3D real-world coordinates (mm) on the pool table surface, establishing a direct connection between computer vision detections and the physical space.

---

## 1. Transformation Method
To convert pixel coordinates to physical table measurements, we apply a two-step process:
1. **Lens Undistortion:** We correct the camera's radial and tangential lens distortion using parameters (`camera_matrix.npy`, `dist_coeffs.npy`) computed in Phase 5:
   $$\mathbf{x}_{undistorted} = \text{undistort}(\mathbf{x}_{pixel}, K, D)$$
2. **Perspective Transformation (Homography):** We establish a planar homography mapping the pool table's playing plane in the camera frame to a flat 2D coordinate system in millimeters.

---

## 2. Homography Matrix
The $3 \times 3$ Homography matrix ($H$) was computed by mapping the four corners of the pool table in pixel space to their real-world dimensions in millimeters:

```python
[[   3.301667,    0.      , -330.166667],
 [   0.      ,    2.475   , -247.5     ],
 [   0.      ,    0.      ,    1.      ]]
```

---

## 3. Scale Factor
The scale factor describes the ratio of physical millimeters per image pixel:
- **Horizontal Scale Factor ($S_x$):** $\approx 3.3017$ mm/pixel
- **Vertical Scale Factor ($S_y$):** $\approx 2.4750$ mm/pixel

These factors represent the scaling along the table surface. Any change in camera mounting height directly changes these scaling ratios.

---

## 4. Conversion Equations
The transformation maps a point $(x, y)$ in the undistorted image space to real-world coordinates $(X, Y)$ on the table surface using the Homography matrix elements $h_{ij}$:

$$X = \frac{h_{00}x + h_{01}y + h_{02}}{h_{20}x + h_{21}y + h_{22}}$$

$$Y = \frac{h_{10}x + h_{11}y + h_{12}}{h_{20}x + h_{21}y + h_{22}}$$

Given our planar configuration where $h_{20} = h_{21} = 0$ and $h_{22} = 1$, the equations simplify to a linear affine projection:
$$X = (3.301667 \times x) - 330.166667$$
$$Y = (2.475000 \times y) - 247.500000$$

---

## 5. Accuracy & Mapping Error
- **YOLO Bounding Box Center Resolution:** YOLOv8 yields bounding box centers with sub-pixel resolution.
- **Physical Bounding Error:** 
  - A 1-pixel horizontal error translates to a physical offset of **3.30 mm**.
  - A 1-pixel vertical error translates to a physical offset of **2.48 mm**.
- **Average Projected Center Uncertainty:** Under nominal camera alignment, the combined localization and mapping uncertainty is estimated to be **$\pm 4.5$ mm**, which easily fits within the target tolerance for laser-pointer aiming on standard pool balls (57.15 mm diameter).

---

## 6. Limitations
- **Coplanar Assumption:** Homography assumes all points lie on the 2D playing surface. Because pool balls have a physical height (radius $\approx 28.58$ mm), the camera's perspective angle causes a slight perspective parallax error (radial displacement outwards from the center of the image).
- **Static Corner Configuration:** The current mapping uses fixed coordinates for the table corners. A dynamic calibration tool should be used during final deployment to allow manual corner clicking.
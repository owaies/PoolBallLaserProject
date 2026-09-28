# Hardware Bridge

The software project currently reaches the coordinate-mapping stage while the ESP32, pan-tilt servos, and laser hardware integration remain future work.

## Current software boundary

Camera input is processed by the detection pipeline. Detected ball centers are converted into physical table coordinates. Configuration files hold game-specific table dimensions, while the hardware layer has not yet been marked as implemented.

## Future integration contract

The eventual bridge should accept validated millimetre coordinates, apply the calibrated pan and tilt transform, and drive the servo mechanism without changing the detection or homography code.

Keep hardware communication isolated from model inference so a camera or model regression can be tested without connected hardware.

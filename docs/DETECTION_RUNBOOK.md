# Detection Runbook

## Standard flow

1. Confirm the intended game configuration.
2. Confirm the calibrated table coordinate system.
3. Place input media in the supported detection input location.
4. Run the detection script.
5. Review the generated CSV and JSON outputs and annotated results.
6. Compare changed results with the existing detection tests before adjusting calibration values.

## Reproducibility

The detection script processes supported file inputs in deterministic filename order. This makes repeated runs easier to compare and reduces noise in result ordering.

## Troubleshooting

When output changes unexpectedly, check the model weights, game configuration, calibration values, input images, and generated result files separately. Do not compensate for a detection problem by changing homography constants without evidence.

# Detection Runbook

## Standard flow

1. Confirm the intended game configuration.
2. Confirm the calibrated table coordinate system.
3. Place input media under the repository's `datasets/` directory (for example, `datasets/raw`). The batch API resolves relative paths from the project root and rejects paths outside this directory, including symlinks that escape it.
4. Run the detection script.
5. Review the generated CSV and JSON outputs and annotated results.
6. Compare changed results with the existing detection tests before adjusting calibration values.

## Reproducibility

The detection script processes supported file inputs in deterministic filename order. This makes repeated runs easier to compare and reduces noise in result ordering.

## Troubleshooting

When output changes unexpectedly, check the model weights, game configuration, calibration values, input images, and generated result files separately. Do not compensate for a detection problem by changing homography constants without evidence.


## Batch API path boundary

The `POST /api/detect/folder` endpoint only processes existing directories
inside `datasets/`. Absolute paths are accepted only when their resolved
location remains inside that root. Missing directories return a generic 404;
invalid or out-of-root paths return a generic 400. Internal filesystem paths
and inference exceptions are not returned to API clients.

Run the focused path-containment checks from the repository root:

```bash
pytest backend/tests/test_detect_folder_security.py
```

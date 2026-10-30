# Image detection upload safety

The `POST /api/detect/image` endpoint reads no more than the configured
`MAX_UPLOAD_SIZE` plus one byte before rejecting an oversized upload. The
client-provided filename is used only to validate its extension; annotated
outputs receive a generated identifier so path components and repeated names
cannot control output locations or overwrite another request's artifacts.

Invalid image content returns HTTP 400. Unexpected model or inference failures
are logged by the backend and return a generic HTTP 503 response without
including internal exception details in the API response.

## Regression checks

Run the focused tests from the repository root:

```bash
pytest backend/tests/test_detect_upload_security.py
```

Verify that:

- an upload one byte over the configured limit is rejected before inference;
- a filename containing path components produces a generated output name;
- non-image content with an allowed extension returns HTTP 400;
- inference failures return a generic HTTP 503 response.

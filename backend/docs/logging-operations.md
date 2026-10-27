# Backend logging operations

The backend writes application and Uvicorn messages to `backend/logs/backend.log`. Keep log output bounded in production so repeated errors do not consume the host's writable disk allocation. When rotating logs are enabled, inspect the active log first, then the numbered backups for older incidents. Treat logs as operational data: do not add passwords, access tokens, or other secrets to log messages, and remove retained files according to the deployment's retention policy.

## Log API behavior

The `/api/logs` endpoint returns at most the requested number of trailing lines, with a default of 100 and an allowed range of 1–1000. If the log file does not exist yet, the endpoint returns a short informational message. Filesystem and text-decoding failures return a generic server error without exposing local paths.

Regression coverage for tail selection and the missing-file case lives in `backend/tests/test_logs_api.py`.

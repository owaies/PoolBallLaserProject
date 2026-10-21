# Backend logging operations

The backend writes application and Uvicorn messages to `backend/logs/backend.log`. Keep log output bounded in production so repeated errors do not consume the host's writable disk allocation. When rotating logs are enabled, inspect the active log first, then the numbered backups for older incidents. Treat logs as operational data: do not add passwords, access tokens, or other secrets to log messages, and remove retained files according to the deployment's retention policy.

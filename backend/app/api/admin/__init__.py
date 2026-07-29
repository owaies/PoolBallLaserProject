"""Admin API package."""
from fastapi import APIRouter

from .auth import router as auth_router
from .datasets import router as datasets_router
from .upload import router as upload_router
from .validate import router as validate_router
from .merge import router as merge_router
from .duplicates import router as duplicates_router
from .clean import router as clean_router
from .statistics import router as statistics_router
from .preview import router as preview_router
from .split import router as split_router
from .backup import router as backup_router
from .reports import router as reports_router
from .logs import router as logs_router

admin_router = APIRouter()
admin_router.include_router(auth_router, tags=["Admin Auth"])
admin_router.include_router(datasets_router, tags=["Admin Datasets"])
admin_router.include_router(upload_router, tags=["Admin Upload"])
admin_router.include_router(validate_router, tags=["Admin Validate"])
admin_router.include_router(merge_router, tags=["Admin Merge"])
admin_router.include_router(duplicates_router, tags=["Admin Duplicates"])
admin_router.include_router(clean_router, tags=["Admin Clean"])
admin_router.include_router(statistics_router, tags=["Admin Statistics"])
admin_router.include_router(preview_router, tags=["Admin Preview"])
admin_router.include_router(split_router, tags=["Admin Split"])
admin_router.include_router(backup_router, tags=["Admin Backup"])
admin_router.include_router(reports_router, tags=["Admin Reports"])
admin_router.include_router(logs_router, tags=["Admin Logs"])

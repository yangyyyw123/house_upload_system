from .bootstrap import ensure_database_tables, ensure_sqlite_columns
from .db import db
from .models import (
    CrackDetectionResults,
    DetectionTask,
    HouseInfo,
    ImageSegmentationRecord,
    QrTarget,
)
from .queries import (
    bind_qr_target_house,
    extract_primary_target,
    get_bundle_records,
    get_house_history_records,
    get_previous_house_record,
    get_recent_house_records,
    resolve_house_from_request,
    resolve_house_from_target,
    resolve_qr_target_house,
)

__all__ = [
    "CrackDetectionResults",
    "DetectionTask",
    "HouseInfo",
    "ImageSegmentationRecord",
    "QrTarget",
    "bind_qr_target_house",
    "db",
    "ensure_database_tables",
    "ensure_sqlite_columns",
    "extract_primary_target",
    "get_bundle_records",
    "get_house_history_records",
    "get_previous_house_record",
    "get_recent_house_records",
    "resolve_house_from_request",
    "resolve_house_from_target",
    "resolve_qr_target_house",
]

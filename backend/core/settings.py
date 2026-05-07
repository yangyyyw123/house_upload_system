from __future__ import annotations

import os
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def normalize_frontend_entry_path(value: str | None) -> str:
    candidate = (value or "/platform").strip() or "/platform"
    if not candidate.startswith("/"):
        candidate = f"/{candidate}"
    return candidate.rstrip("/") or "/"


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_ROOT = BASE_DIR / "uploads"
SEGMENTATION_ROOT = UPLOAD_ROOT / "segmentation"
TARGET_ROOT = UPLOAD_ROOT / "targets"
PREFERRED_PYTHON = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
APP_TIMEZONE_NAME = os.environ.get("APP_TIMEZONE", "Asia/Shanghai")
try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except ZoneInfoNotFoundError:
    APP_TIMEZONE = timezone(timedelta(hours=8)) if APP_TIMEZONE_NAME == "Asia/Shanghai" else timezone.utc

FRONTEND_ENTRY_PATH = normalize_frontend_entry_path(os.environ.get("HOUSE_UPLOAD_ENTRY_PATH"))

QUALITY_GUIDANCE = {
    "good": "图像质量通过，可继续自动分析。",
    "warning": "图像可以分析，但建议根据提示优化拍摄质量。",
    "reject": "图像质量不足，建议按提示补拍；如必须保留本次记录，可手动强制上传。",
}

RISK_TEXT = {
    "无风险": "当前裂缝迹象较轻，建议纳入常规观察。",
    "低风险": "当前更像表层裂缝，建议记录并定期复拍。",
    "中风险": "裂缝已有持续发展风险，建议安排专业人员复核。",
    "高风险": "裂缝风险较高，建议尽快停用相关区域并联系专业机构检测。",
}

RISK_ACTIONS = {
    "无风险": "保留本次检测档案，1-3 个月内复拍一次。",
    "低风险": "建议定点复拍并安排表层修补，如位于非承重部位可先观察。",
    "中风险": "建议尽快安排专业人员现场复核，并做好临时警示。",
    "高风险": "建议立即停止使用相关区域，联系结构安全检测与加固单位。",
}

RISK_LEVEL_THRESHOLDS = [
    ("无风险", 0, 1),
    ("低风险", 2, 4),
    ("中风险", 5, 7),
    ("高风险", 8, None),
]

REPORT_PAGE_SIZE = (1240, 1754)
REPORT_MARGIN = 72
REPORT_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]
BUNDLE_CAPTURE_SCALES = [
    ("long_view", "长景"),
    ("medium_view", "中景"),
    ("close_view", "近景"),
]
CAPTURE_SCALE_LABELS = {key: label for key, label in BUNDLE_CAPTURE_SCALES}
MARKER_MODE_LABELS = {
    "qr_anchor_rectified": "二维码 + 四角校正",
    "qr_decode": "二维码解码",
    "legacy_rect": "旧版靶标识别",
}

BUNDLE_PROJECTION_COLORS = {
    "中景": (44, 163, 242),
    "近景": (255, 170, 40),
}

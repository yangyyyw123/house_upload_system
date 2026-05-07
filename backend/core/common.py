from __future__ import annotations

import json
from uuid import uuid4


def clean_text(value: str | None, fallback: str = "") -> str:
    return (value or fallback).strip()


def encode_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def decode_json(value: str | None, fallback: object) -> object:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def generate_detection_code(house_number: str | None) -> str:
    prefix = clean_text(house_number, "HOUSE").upper().replace(" ", "")[:12] or "HOUSE"
    return f"{prefix}-{uuid4().hex[:8].upper()}"


def generate_bundle_code(house_number: str | None) -> str:
    prefix = clean_text(house_number, "BUNDLE").upper().replace(" ", "")[:12] or "BUNDLE"
    return f"{prefix}-BATCH-{uuid4().hex[:6].upper()}"


def generate_task_code(house_number: str | None) -> str:
    prefix = clean_text(house_number, "TASK").upper().replace(" ", "")[:12] or "TASK"
    return f"{prefix}-TASK-{uuid4().hex[:8].upper()}"


def collect_upload_metadata(form_data) -> dict[str, str]:
    return {
        "capture_scale": clean_text(form_data.get("capture_scale"), "未标注"),
        "component_type": clean_text(form_data.get("component_type"), "未标注"),
        "scenario_type": clean_text(form_data.get("scenario_type"), "常规巡检"),
    }

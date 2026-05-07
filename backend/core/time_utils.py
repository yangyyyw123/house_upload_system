from __future__ import annotations

from datetime import datetime, timezone

from .common import clean_text
from .settings import APP_TIMEZONE


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def to_local_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(APP_TIMEZONE)


def to_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def serialize_datetime_value(value: datetime | None) -> str | None:
    utc_value = to_utc_datetime(value)
    if utc_value is None:
        return None
    return utc_value.isoformat(timespec="seconds").replace("+00:00", "Z")


def format_datetime_value(value: datetime | None, include_seconds: bool = True) -> str:
    local_value = to_local_datetime(value)
    if local_value is None:
        return "-"
    return local_value.strftime("%Y-%m-%d %H:%M:%S" if include_seconds else "%Y-%m-%d %H:%M")


def parse_local_datetime_input(value: str | None) -> datetime | None:
    text = clean_text(value)
    if not text:
        return None

    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            local_value = datetime.strptime(text, fmt).replace(tzinfo=APP_TIMEZONE)
        except ValueError:
            continue
        return local_value.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
    return None


def format_datetime_local_input(value: datetime | None) -> str:
    local_value = to_local_datetime(value)
    if local_value is None:
        return ""
    return local_value.strftime("%Y-%m-%dT%H:%M")


def get_record_uploaded_at(record) -> datetime | None:
    return record.uploaded_at or record.created_at


def get_record_generated_at(record) -> datetime | None:
    return record.created_at


def get_bundle_uploaded_at(records) -> datetime | None:
    timestamps = [get_record_uploaded_at(record) for record in records if get_record_uploaded_at(record)]
    return min(timestamps) if timestamps else None


def get_bundle_generated_at(records) -> datetime | None:
    timestamps = [get_record_generated_at(record) for record in records if get_record_generated_at(record)]
    return max(timestamps) if timestamps else None


def format_report_timestamp(record, include_seconds: bool = False) -> str:
    if record.created_at:
        return format_datetime_value(record.created_at, include_seconds=include_seconds)
    return f"记录 {record.id}"


def format_uploaded_timestamp(record, include_seconds: bool = False) -> str:
    uploaded_at = get_record_uploaded_at(record)
    if uploaded_at:
        return format_datetime_value(uploaded_at, include_seconds=include_seconds)
    return f"记录 {record.id}"

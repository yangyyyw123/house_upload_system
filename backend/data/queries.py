from __future__ import annotations

from sqlalchemy import and_

try:
    from ..core.common import clean_text
    from ..core.settings import BUNDLE_CAPTURE_SCALES
    from .db import db
    from .models import HouseInfo, ImageSegmentationRecord, QrTarget
except ImportError:
    from core.common import clean_text
    from core.settings import BUNDLE_CAPTURE_SCALES
    from data.db import db
    from data.models import HouseInfo, ImageSegmentationRecord, QrTarget


def get_previous_house_record(record: ImageSegmentationRecord) -> ImageSegmentationRecord | None:
    if not record.house_id:
        return None

    query = ImageSegmentationRecord.query.filter_by(house_id=record.house_id, analysis_status="completed").filter(
        ImageSegmentationRecord.id != record.id
    )

    if record.created_at is not None:
        query = query.filter(
            and_(
                ImageSegmentationRecord.created_at.isnot(None),
                (
                    (ImageSegmentationRecord.created_at < record.created_at)
                    | (
                        and_(
                            ImageSegmentationRecord.created_at == record.created_at,
                            ImageSegmentationRecord.id < record.id,
                        )
                    )
                ),
            )
        )
    else:
        query = query.filter(ImageSegmentationRecord.id < record.id)

    return query.order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc()).first()


def get_house_history_records(house_id: int | None, limit: int = 5) -> list[ImageSegmentationRecord]:
    if not house_id:
        return []
    return (
        ImageSegmentationRecord.query.filter_by(house_id=house_id)
        .order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .limit(limit)
        .all()
    )


def get_bundle_records(bundle_code: str) -> list[ImageSegmentationRecord]:
    records = (
        ImageSegmentationRecord.query.filter_by(upload_bundle_code=bundle_code)
        .order_by(ImageSegmentationRecord.created_at.asc(), ImageSegmentationRecord.id.asc())
        .all()
    )
    scale_order = {label: index for index, (_, label) in enumerate(BUNDLE_CAPTURE_SCALES)}
    return sorted(records, key=lambda item: scale_order.get(item.capture_scale or "", 99))


def resolve_house_from_request(house_id_value: str | None) -> tuple[HouseInfo | None, tuple[dict, int] | None]:
    if not house_id_value:
        return None, None

    try:
        house_id = int(house_id_value)
    except ValueError:
        return None, ({"message": "house_id 无效。"}, 400)

    house = db.session.get(HouseInfo, house_id)
    if house is None:
        return None, ({"message": "未找到对应的房屋档案。"}, 404)

    return house, None


def resolve_qr_target_house(house_id_value: str | None, house_number: str) -> HouseInfo | None:
    normalized_house_number = clean_text(house_number)
    if normalized_house_number:
        matched_by_number = HouseInfo.query.filter_by(house_number=normalized_house_number).first()
        if matched_by_number is not None:
            return matched_by_number

    if not house_id_value:
        return None

    try:
        house_id = int(house_id_value)
    except ValueError:
        return None

    house = db.session.get(HouseInfo, house_id)
    if house is None:
        return None
    if normalized_house_number and clean_text(house.house_number) != normalized_house_number:
        return None
    return house


def extract_primary_target(marker_detection: dict[str, object] | None) -> dict[str, str]:
    targets = (marker_detection or {}).get("targets") or []
    if not isinstance(targets, list):
        return {}

    for target in targets:
        if not isinstance(target, dict):
            continue
        target_id = clean_text(target.get("target_id"))
        house_number = clean_text(target.get("house_number"))
        scene_type = clean_text(target.get("scene_type"))
        if target_id or house_number:
            return {
                "target_id": target_id,
                "house_number": house_number,
                "scene_type": scene_type,
            }
    return {}


def resolve_house_from_target(target_info: dict[str, str]) -> HouseInfo | None:
    target_id = clean_text(target_info.get("target_id"))
    if target_id:
        qr_target = QrTarget.query.filter_by(target_id=target_id).first()
        if qr_target and qr_target.house_id:
            return db.session.get(HouseInfo, qr_target.house_id)

    house_number = clean_text(target_info.get("house_number"))
    if house_number:
        return HouseInfo.query.filter_by(house_number=house_number).first()
    return None


def bind_qr_target_house(target_id: str | None, house: HouseInfo | None) -> None:
    if not target_id or house is None:
        return

    qr_target = QrTarget.query.filter_by(target_id=target_id).first()
    if qr_target is None:
        return

    if qr_target.house_id is None:
        qr_target.house_id = house.id
    if not clean_text(qr_target.house_number):
        qr_target.house_number = house.house_number


def get_recent_house_records(house_id: int | None, limit: int = 5) -> list[ImageSegmentationRecord]:
    if not house_id:
        return []
    return (
        ImageSegmentationRecord.query.filter_by(house_id=house_id, analysis_status="completed")
        .order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .limit(limit)
        .all()
    )

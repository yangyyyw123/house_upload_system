import csv
import os
import sys
import textwrap
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat, UnidentifiedImageError
from sqlalchemy import and_, or_, text
from werkzeug.utils import secure_filename

try:
    from .crack_segmentation_service import (
        CrackSegmentationService,
        InferenceDependencyError,
        ModelConfigurationError,
    )
except ImportError:
    from crack_segmentation_service import (
        CrackSegmentationService,
        InferenceDependencyError,
        ModelConfigurationError,
    )

try:
    from .crack_quantification_service import CrackQuantificationService, QuantificationError
except ImportError:
    from crack_quantification_service import CrackQuantificationService, QuantificationError

try:
    from .target_generator_service import TARGET_SPECS, TargetGenerationError, TargetGeneratorService
except ImportError:
    from target_generator_service import TARGET_SPECS, TargetGenerationError, TargetGeneratorService

try:
    from .data import (
        CrackDetectionResults,
        DetectionTask,
        HouseInfo,
        ImageSegmentationRecord,
        QrTarget,
        bind_qr_target_house,
        db,
        ensure_database_tables as _ensure_database_tables,
        ensure_sqlite_columns as _ensure_sqlite_columns,
        extract_primary_target,
        get_bundle_records,
        get_house_history_records,
        get_previous_house_record,
        get_recent_house_records,
        resolve_house_from_request,
        resolve_house_from_target,
        resolve_qr_target_house,
    )
except ImportError:
    from data import (
        CrackDetectionResults,
        DetectionTask,
        HouseInfo,
        ImageSegmentationRecord,
        QrTarget,
        bind_qr_target_house,
        db,
        ensure_database_tables as _ensure_database_tables,
        ensure_sqlite_columns as _ensure_sqlite_columns,
        extract_primary_target,
        get_bundle_records,
        get_house_history_records,
        get_previous_house_record,
        get_recent_house_records,
        resolve_house_from_request,
        resolve_house_from_target,
        resolve_qr_target_house,
    )

try:
    from .core import (
        BUNDLE_CAPTURE_SCALES,
        BUNDLE_PROJECTION_COLORS,
        FRONTEND_ENTRY_PATH,
        PREFERRED_PYTHON,
        QUALITY_GUIDANCE,
        REPORT_FONT_CANDIDATES,
        REPORT_MARGIN,
        REPORT_PAGE_SIZE,
        RISK_ACTIONS,
        RISK_LEVEL_THRESHOLDS,
        RISK_TEXT,
        SEGMENTATION_ROOT,
        TARGET_ROOT,
        UPLOAD_ROOT,
        allowed_file,
        build_file_url,
        build_target_public_url,
        clean_text,
        collect_upload_metadata,
        decode_json,
        encode_json,
        ensure_runtime_directories,
        format_capture_scale_label,
        format_datetime_value,
        format_marker_mode_label,
        format_report_timestamp,
        format_uploaded_timestamp,
        generate_bundle_code,
        generate_detection_code,
        generate_task_code,
        get_bundle_generated_at,
        get_bundle_uploaded_at,
        get_record_generated_at,
        get_record_uploaded_at,
        get_runtime_dependency_status,
        parse_local_datetime_input,
        serialize_datetime_value,
        utc_now,
    )
except ImportError:
    from core import (
        BUNDLE_CAPTURE_SCALES,
        BUNDLE_PROJECTION_COLORS,
        FRONTEND_ENTRY_PATH,
        PREFERRED_PYTHON,
        QUALITY_GUIDANCE,
        REPORT_FONT_CANDIDATES,
        REPORT_MARGIN,
        REPORT_PAGE_SIZE,
        RISK_ACTIONS,
        RISK_LEVEL_THRESHOLDS,
        RISK_TEXT,
        SEGMENTATION_ROOT,
        TARGET_ROOT,
        UPLOAD_ROOT,
        allowed_file,
        build_file_url,
        build_target_public_url,
        clean_text,
        collect_upload_metadata,
        decode_json,
        encode_json,
        ensure_runtime_directories,
        format_capture_scale_label,
        format_datetime_value,
        format_marker_mode_label,
        format_report_timestamp,
        format_uploaded_timestamp,
        generate_bundle_code,
        generate_detection_code,
        generate_task_code,
        get_bundle_generated_at,
        get_bundle_uploaded_at,
        get_record_generated_at,
        get_record_uploaded_at,
        get_runtime_dependency_status,
        parse_local_datetime_input,
        serialize_datetime_value,
        utc_now,
    )


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.static_folder = str((BASE_DIR.parent / "frontend").resolve())
app.static_url_path = "/static"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///house.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = str(UPLOAD_ROOT)
app.config["SEGMENTATION_FOLDER"] = str(SEGMENTATION_ROOT)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

db.init_app(app)
segmentation_service = CrackSegmentationService()
quantification_service = CrackQuantificationService()
target_generator_service = TargetGeneratorService(TARGET_ROOT)
task_executor = ThreadPoolExecutor(max_workers=1)


try:
    from .core.runtime import format_inference_error as _format_inference_error
except ImportError:
    from core.runtime import format_inference_error as _format_inference_error


def format_inference_error(exc: Exception) -> str:
    return _format_inference_error(exc, segmentation_service=segmentation_service)


def ensure_database_tables() -> None:
    _ensure_database_tables(app)


def ensure_sqlite_columns() -> None:
    _ensure_sqlite_columns(app)


def analyze_image_quality(image_path: Path) -> dict[str, object]:
    warnings: list[str] = []
    score = 100.0
    hard_fail = False

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            grayscale = image.convert("L")
            brightness = round(ImageStat.Stat(grayscale).mean[0], 2)
            edge_strength = round(ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES)).mean[0], 2)
    except (UnidentifiedImageError, OSError) as exc:
        return {
            "status": "reject",
            "score": 0.0,
            "summary": f"无法读取图像内容：{exc}",
            "warnings": ["文件可能已损坏或不是有效图像，请重新拍摄并上传。"],
            "metrics": {},
            "guidance": QUALITY_GUIDANCE["reject"],
        }

    file_size_kb = round(image_path.stat().st_size / 1024, 2)

    if min(width, height) < 900:
        score -= 28
        warnings.append("图像分辨率偏低，建议使用原图上传，并保证裂缝区域清晰。")
    elif min(width, height) < 1400:
        score -= 12
        warnings.append("图像分辨率一般，细微裂缝的测量稳定性可能下降。")

    if min(width, height) < 400:
        hard_fail = True
        warnings.append("图像尺寸过小，无法满足裂缝测量要求。")

    if file_size_kb < 280:
        score -= 12
        warnings.append("文件体积偏小，可能经过压缩，建议关闭聊天软件压缩后重新上传。")

    if brightness < 45:
        score -= 18
        warnings.append("画面偏暗，建议补光后重拍，避免裂缝细节丢失。")
    elif brightness > 220:
        score -= 10
        warnings.append("画面过亮，可能存在过曝区域，建议降低曝光或调整角度。")

    if edge_strength < 5:
        score -= 24
        warnings.append("图像锐度偏低，可能存在抖动或失焦，建议稳定手机并重新对焦。")
    elif edge_strength < 8:
        score -= 10
        warnings.append("图像清晰度一般，建议补拍一张更稳定的近景照片。")

    score = max(0.0, min(100.0, round(score, 1)))
    if hard_fail or score < 45:
        status = "reject"
        summary = "图像质量不足，建议先补拍再提交分析。"
    elif warnings:
        status = "warning"
        summary = "图像可以进入分析，但存在影响结果稳定性的风险。"
    else:
        status = "good"
        summary = "图像质量良好，满足自动分析要求。"

    return {
        "status": status,
        "score": score,
        "summary": summary,
        "warnings": warnings,
        "metrics": {
            "width": width,
            "height": height,
            "file_size_kb": file_size_kb,
            "brightness": brightness,
            "edge_strength": edge_strength,
        },
        "guidance": QUALITY_GUIDANCE[status],
    }


def assess_detection_risk(
    segmentation_result: dict[str, object],
    quantification: dict[str, object],
    quality_report: dict[str, object],
    metadata: dict[str, str],
    history_records: list["ImageSegmentationRecord"] | None = None,
) -> dict[str, object]:
    area_ratio = float(segmentation_result.get("crack_area_ratio") or 0.0)
    component_type = clean_text(metadata.get("component_type"), "未标注")
    scenario_type = clean_text(metadata.get("scenario_type"), "未标注")
    max_width_mm = quantification.get("max_width_mm")
    crack_length_mm = quantification.get("crack_length_mm")
    crack_angle_deg = quantification.get("crack_angle_deg")
    is_structural = component_type in {"承重墙", "梁", "柱"}
    width_value = float(max_width_mm or 0.0)
    length_value = float(crack_length_mm or 0.0)

    score = 0
    score_breakdown: list[dict[str, object]] = []
    explanation_notes: list[str] = []

    def add_score_item(points: int, title: str, detail: str) -> None:
        nonlocal score
        score += points
        score_breakdown.append(
            {
                "title": title,
                "score": points,
                "detail": detail,
            }
        )

    if max_width_mm is not None:
        if width_value >= 1.0:
            add_score_item(5, "最大宽度", f"最大宽度 {width_value:.3f} mm，已达到高风险阈值。")
        elif width_value >= 0.6:
            add_score_item(4, "最大宽度", f"最大宽度 {width_value:.3f} mm，已明显超过常规观察阈值。")
        elif width_value >= 0.3:
            add_score_item(2, "最大宽度", f"最大宽度 {width_value:.3f} mm，建议重点跟踪。")
    else:
        if area_ratio >= 0.008:
            add_score_item(4, "面积占比", "当前未建立毫米尺度，裂缝面积占比较高。")
        elif area_ratio >= 0.003:
            add_score_item(2, "面积占比", "当前未建立毫米尺度，裂缝面积占比已有增长迹象。")
        elif area_ratio >= 0.001:
            add_score_item(1, "面积占比", "当前未建立毫米尺度，裂缝面积占比已进入观察区间。")

    if crack_length_mm is not None:
        if length_value >= 800:
            add_score_item(3, "裂缝长度", f"裂缝长度 {length_value:.1f} mm，属于长裂缝。")
        elif length_value >= 300:
            add_score_item(2, "裂缝长度", f"裂缝长度 {length_value:.1f} mm，已超过常规巡检关注阈值。")
        elif length_value >= 120:
            add_score_item(1, "裂缝长度", f"裂缝长度 {length_value:.1f} mm，建议继续跟踪。")
    elif area_ratio >= 0.005:
        add_score_item(1, "面积占比", "当前未建立长度量测，裂缝面积占比较高。")

    if is_structural:
        add_score_item(2, "构件类型", "裂缝位于承重或主要受力构件。")

    if scenario_type == "灾后评估":
        add_score_item(1, "场景类型", "当前场景为灾后评估，风险基线提高一级。")

    if crack_angle_deg is not None and is_structural:
        angle_value = abs(float(crack_angle_deg))
        normalized_angle = min(angle_value % 180.0, 180.0 - (angle_value % 180.0))
        if 35.0 <= normalized_angle <= 65.0:
            add_score_item(1, "裂缝方向", f"裂缝方向 {angle_value:.1f} deg，属于结构受力敏感方向。")

    latest_previous = history_records[0] if history_records else None
    if latest_previous is not None:
        previous_width = latest_previous.max_width_mm
        previous_length = latest_previous.crack_length_mm
        if previous_width is not None and max_width_mm is not None and previous_width > 0:
            width_growth = (width_value - float(previous_width)) / float(previous_width)
            if width_growth >= 0.5:
                add_score_item(2, "历史变化", f"最大宽度较上次增长 {width_growth * 100:.1f}%。")
            elif width_growth >= 0.2:
                add_score_item(1, "历史变化", f"最大宽度较上次增长 {width_growth * 100:.1f}%。")
        elif previous_length is not None and crack_length_mm is not None and previous_length > 0:
            length_growth = (length_value - float(previous_length)) / float(previous_length)
            if length_growth >= 0.5:
                add_score_item(1, "历史变化", f"裂缝长度较上次增长 {length_growth * 100:.1f}%。")
        elif latest_previous.crack_area_ratio and area_ratio > latest_previous.crack_area_ratio:
            area_growth = (area_ratio - float(latest_previous.crack_area_ratio)) / max(float(latest_previous.crack_area_ratio), 1e-6)
            if area_growth >= 0.5:
                add_score_item(1, "历史变化", f"裂缝面积占比较上次增长 {area_growth * 100:.1f}%。")

    level = "无风险"
    for risk_level, min_score, _max_score in reversed(RISK_LEVEL_THRESHOLDS):
        if score >= min_score:
            level = risk_level
            break

    summary = RISK_TEXT[level]
    summary_details = [item["detail"] for item in score_breakdown[:3]]
    if summary_details:
        summary += " " + " ".join(summary_details)
    recommendation = RISK_ACTIONS[level]
    if max_width_mm is not None and width_value >= 1.0 and is_structural:
        recommendation = "建议尽快安排专业结构安全检测，并限制相关承重区域继续使用。"
    elif max_width_mm is not None and width_value >= 0.3:
        recommendation = "建议在 1-2 周内复拍复核，并结合现场巡查确认是否继续扩展。"

    if not score_breakdown:
        explanation_notes.append("当前未触发宽度、长度、构件类型或历史增长等加分项，系统按常规观察范围处理。")

    quality_status = quality_report.get("status")
    if quality_status == "warning":
        quality_note = "当前图像质量一般，建议结合补拍结果综合判断。"
        explanation_notes.append(quality_note)
        summary += " " + quality_note
    elif quality_status == "reject":
        quality_note = "当前图像质量较差，本次结果应优先视为补拍前的参考。"
        explanation_notes.append(quality_note)
        summary += " " + quality_note

    return {
        "risk_level": level,
        "score": score,
        "risk_summary": summary,
        "recommendation": recommendation,
        "score_breakdown": score_breakdown,
        "explanation_notes": explanation_notes,
        "level_thresholds": build_risk_level_thresholds(),
    }


def build_record_segmentation_snapshot(record: "ImageSegmentationRecord") -> dict[str, object]:
    return {
        "crack_area_ratio": record.crack_area_ratio,
    }


def build_record_quantification_snapshot(record: "ImageSegmentationRecord") -> dict[str, object]:
    return {
        "max_width_mm": record.max_width_mm,
        "crack_length_mm": record.crack_length_mm,
        "crack_angle_deg": record.crack_angle_deg,
    }


def build_record_quality_snapshot(record: "ImageSegmentationRecord") -> dict[str, object]:
    return {
        "status": record.quality_status,
        "score": record.quality_score,
        "warnings": decode_json(record.quality_warnings_json, []),
    }


def build_record_metadata(record: "ImageSegmentationRecord") -> dict[str, str]:
    return {
        "component_type": clean_text(record.component_type, "未标注"),
        "scenario_type": clean_text(record.scenario_type, "未标注"),
    }


def build_previous_record_lookup(
    records: list["ImageSegmentationRecord"],
) -> dict[int, "ImageSegmentationRecord | None"]:
    previous_by_id: dict[int, ImageSegmentationRecord | None] = {}
    completed_records = sorted(
        [record for record in records if record.analysis_status == "completed"],
        key=lambda item: (item.created_at or datetime.min, item.id or 0),
    )

    previous_record: ImageSegmentationRecord | None = None
    for record in completed_records:
        previous_by_id[record.id] = previous_record
        previous_record = record

    for record in records:
        previous_by_id.setdefault(record.id, None)
    return previous_by_id


def build_risk_assessment_for_record(
    record: "ImageSegmentationRecord",
    previous_record: "ImageSegmentationRecord | None" = None,
) -> dict[str, object]:
    if previous_record is None:
        previous_record = get_previous_house_record(record)

    return assess_detection_risk(
        segmentation_result=build_record_segmentation_snapshot(record),
        quantification=build_record_quantification_snapshot(record),
        quality_report=build_record_quality_snapshot(record),
        metadata=build_record_metadata(record),
        history_records=[previous_record] if previous_record else [],
    )


def get_record_trend_metric(record: "ImageSegmentationRecord") -> tuple[str, float]:
    if record.max_width_mm is not None:
        return "max_width_mm", float(record.max_width_mm)
    if record.crack_length_mm is not None:
        return "crack_length_mm", float(record.crack_length_mm)
    return "crack_area_ratio", float(record.crack_area_ratio or 0.0)


def get_trend_metric_label(metric_name: str) -> tuple[str, str]:
    mapping = {
        "max_width_mm": ("最大宽度趋势曲线", "mm"),
        "crack_length_mm": ("裂缝长度趋势曲线", "mm"),
        "crack_area_ratio": ("裂缝占比趋势曲线", "ratio"),
    }
    return mapping.get(metric_name, ("趋势曲线", "value"))


def serialize_quality_report(report: dict[str, object]) -> dict[str, object]:
    return {
        "status": report["status"],
        "score": report["score"],
        "summary": report["summary"],
        "warnings": report["warnings"],
        "metrics": report["metrics"],
        "guidance": report["guidance"],
    }


def build_analysis_stages(
    marker_detection: dict[str, object] | None,
    quantification: dict[str, object] | None,
) -> dict[str, dict[str, str]]:
    marker_status = (marker_detection or {}).get("status")
    marker_count = int((marker_detection or {}).get("marker_count") or 0)
    physical_scale = (marker_detection or {}).get("physical_scale_mm_per_pixel")
    quant_status = (quantification or {}).get("status")

    if marker_status in {"completed", "anchor_rectified"}:
        recognition_summary = f"识别阶段已完成，检测到 {marker_count} 个靶标，并可生成 mm/pixel 比例。"
    elif marker_status == "qr_detected":
        recognition_summary = f"二维码已识别，但四角锚点不足，未完成透视矫正；当前比例仅基于二维码边长估计。"
    elif marker_status == "fallback_detected":
        recognition_summary = f"二维码未识别成功，已回退到矩形靶标估计模式，检测到 {marker_count} 个候选靶标。"
    else:
        recognition_summary = "识别阶段已完成裂缝分割，但未识别到可用靶标。"

    plane_mode = (quantification or {}).get("plane_mode")
    if quant_status == "completed":
        quant_summary = (
            "量化阶段已完成，已在透视矫正后的平面上输出毫米级宽度、长度和角度指标。"
            if plane_mode == "rectified_plane"
            else "量化阶段已完成，已输出毫米级宽度、长度和角度指标。"
        )
    elif quant_status == "pixel_only":
        quant_summary = "量化阶段已完成，但当前仅能输出像素级指标，未换算为毫米。"
    else:
        quant_summary = "量化阶段暂未生成有效结果。"

    return {
        "recognition": {
            "status": "completed",
            "summary": recognition_summary,
            "physical_scale": f"{physical_scale:.6f} mm/pixel" if physical_scale else "未建立",
        },
        "quantification": {
            "status": str(quant_status or "unavailable"),
            "summary": quant_summary,
        },
    }


def build_empty_quantification_result(message: str, status: str = "failed") -> dict[str, object]:
    return {
        "marker_detection": {
            "status": "not_found",
            "message": "marker not available",
            "marker_count": 0,
            "physical_scale_mm_per_pixel": None,
            "annotated_image_path": None,
            "detection_method": None,
            "anchor_count": 0,
            "perspective_corrected": False,
            "targets": [],
        },
        "quantification": {
            "status": status,
            "message": message,
            "plane_mode": "image_plane",
            "component_count": 0,
            "sample_count": 0,
            "max_width_px": None,
            "avg_width_px": None,
            "median_width_px": None,
            "crack_length_px": None,
            "crack_angle_deg": None,
            "max_width_mm": None,
            "avg_width_mm": None,
            "median_width_mm": None,
            "crack_length_mm": None,
            "quant_overlay_path": None,
            "width_chart_path": None,
        },
    }


def run_quantification_analysis(
    source_image_path: Path,
    segmentation_result: dict[str, object],
) -> dict[str, object]:
    mask_path = segmentation_result.get("mask_path")
    if not mask_path:
        return build_empty_quantification_result("缺少裂缝掩码，无法进入量化阶段。", status="unavailable")

    try:
        return quantification_service.analyze(
            source_image_path=source_image_path,
            mask_path=Path(str(mask_path)),
            output_dir=Path(str(mask_path)).parent,
        )
    except (FileNotFoundError, QuantificationError) as exc:
        return build_empty_quantification_result(str(exc))
    except Exception as exc:  # pragma: no cover
        return build_empty_quantification_result(f"量化阶段异常：{exc}")


def build_bundle_projection_unavailable(message: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "message": message,
        "overview_image_url": None,
        "projected_items": [],
        "component_bbox": None,
    }


def collect_bundle_projection_inputs_from_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for result in results:
        source_path = result.get("file_path")
        segmentation = result.get("segmentation") or {}
        mask_path = segmentation.get("mask_path")
        items.append(
            {
                "capture_scale": clean_text(result.get("capture_scale"), "未标注"),
                "detection_code": clean_text(result.get("detection_code"), "-"),
                "source_image_path": Path(str(source_path)) if source_path else None,
                "mask_path": Path(str(mask_path)) if mask_path else None,
            }
        )
    return items


def collect_bundle_projection_inputs_from_records(records: list["ImageSegmentationRecord"]) -> list[dict[str, object]]:
    return [
        {
            "capture_scale": clean_text(record.capture_scale, "未标注"),
            "detection_code": clean_text(record.detection_code, f"record-{record.id}"),
            "source_image_path": Path(record.source_image_path) if record.source_image_path else None,
            "mask_path": Path(record.mask_path) if record.mask_path else None,
        }
        for record in records
    ]


def load_projection_mask_contours(mask_path: Path | None) -> list[np.ndarray]:
    if mask_path is None or not mask_path.exists():
        return []

    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        return []

    binary_mask = ((mask_image > 127).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < 24:
            continue
        kept.append(contour.reshape(-1, 2).astype(np.float32))
    kept.sort(key=lambda item: cv2.contourArea(item.reshape(-1, 1, 2)), reverse=True)
    return kept[:24]


def build_bundle_projection_context(item: dict[str, object]) -> dict[str, object] | None:
    source_image_path = item.get("source_image_path")
    if not isinstance(source_image_path, Path) or not source_image_path.exists():
        return None

    source_image = cv2.imread(str(source_image_path))
    if source_image is None:
        return None

    marker_detection = quantification_service.inspect_target_image(source_image, include_geometry=True)
    targets = marker_detection.get("targets") or []
    projection_target = next(
        (
            target
            for target in targets
            if target.get("image_to_target_matrix") is not None and target.get("target_to_image_matrix") is not None
        ),
        None,
    )
    target_key = clean_text(
        (projection_target or {}).get("target_id") or (projection_target or {}).get("house_number"),
        "",
    )

    return {
        "capture_scale": item.get("capture_scale"),
        "detection_code": item.get("detection_code"),
        "source_image_path": source_image_path,
        "mask_path": item.get("mask_path"),
        "image": source_image,
        "marker_detection": marker_detection,
        "projection_target": projection_target,
        "target_key": target_key,
        "mask_contours": load_projection_mask_contours(item.get("mask_path") if isinstance(item.get("mask_path"), Path) else None),
    }


def project_contours_to_long_view(
    contours: list[np.ndarray],
    source_target: dict[str, object],
    long_target: dict[str, object],
    long_image_shape: tuple[int, ...],
) -> list[np.ndarray]:
    if not contours:
        return []

    source_to_target = np.asarray(source_target.get("image_to_target_matrix"), dtype=np.float32)
    target_to_long = np.asarray(long_target.get("target_to_image_matrix"), dtype=np.float32)
    if source_to_target.shape != (3, 3) or target_to_long.shape != (3, 3):
        return []

    long_height, long_width = long_image_shape[:2]
    projected: list[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        source_points = contour.reshape(1, -1, 2).astype(np.float32)
        target_points = cv2.perspectiveTransform(source_points, source_to_target)
        long_points = cv2.perspectiveTransform(target_points, target_to_long).reshape(-1, 2)
        if not np.isfinite(long_points).all():
            continue

        long_points[:, 0] = np.clip(long_points[:, 0], -long_width, long_width * 2.0)
        long_points[:, 1] = np.clip(long_points[:, 1], -long_height, long_height * 2.0)
        min_x = float(long_points[:, 0].min())
        min_y = float(long_points[:, 1].min())
        max_x = float(long_points[:, 0].max())
        max_y = float(long_points[:, 1].max())
        if max_x < -48 or max_y < -48 or min_x > long_width + 48 or min_y > long_height + 48:
            continue
        projected.append(long_points.astype(np.float32))
    return projected


def compute_projection_bbox(contours: list[np.ndarray], image_shape: tuple[int, ...]) -> dict[str, int] | None:
    if not contours:
        return None

    all_points = np.concatenate(contours, axis=0)
    image_height, image_width = image_shape[:2]
    min_x = float(all_points[:, 0].min())
    min_y = float(all_points[:, 1].min())
    max_x = float(all_points[:, 0].max())
    max_y = float(all_points[:, 1].max())

    pad_x = max(36, int(round(max(max_x - min_x, 1.0) * 0.18)))
    pad_y = max(36, int(round(max(max_y - min_y, 1.0) * 0.18)))
    left = max(0, int(math.floor(min_x - pad_x)))
    top = max(0, int(math.floor(min_y - pad_y)))
    right = min(image_width - 1, int(math.ceil(max_x + pad_x)))
    bottom = min(image_height - 1, int(math.ceil(max_y + pad_y)))
    if right <= left or bottom <= top:
        return None
    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
    }


def draw_bundle_projection_overlay(
    long_context: dict[str, object],
    projected_items: list[dict[str, object]],
    component_bbox: dict[str, int] | None,
    output_path: Path,
) -> None:
    base_image = np.asarray(long_context["image"]).copy()
    tint_layer = base_image.copy()

    long_target = long_context.get("projection_target") or {}
    long_contour = np.asarray(long_target.get("contour") or [], dtype=np.float32).reshape(-1, 2)
    if long_contour.shape[0] >= 4:
        long_target_contour = np.round(long_contour).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(base_image, [long_target_contour], True, (92, 214, 178), 2, cv2.LINE_AA)
        anchor_x, anchor_y = tuple(long_target_contour[0][0])
        cv2.putText(
            base_image,
            "TARGET",
            (int(anchor_x), max(28, int(anchor_y) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (92, 214, 178),
            2,
            cv2.LINE_AA,
        )

    scale_labels = {"中景": "MID", "近景": "CLOSE"}
    for item in projected_items:
        color = BUNDLE_PROJECTION_COLORS.get(str(item["capture_scale"]), (44, 163, 242))
        contour_arrays = item.get("projected_contours") or []
        for contour in contour_arrays:
            contour_array = np.round(np.asarray(contour, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            if contour_array.shape[0] < 3:
                continue
            cv2.fillPoly(tint_layer, [contour_array], color)
            cv2.polylines(base_image, [contour_array], True, color, 2, cv2.LINE_AA)

        centroid = item.get("centroid")
        if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            cv2.putText(
                base_image,
                scale_labels.get(str(item["capture_scale"]), "VIEW"),
                (int(centroid[0]) + 8, max(24, int(centroid[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    overlay = cv2.addWeighted(tint_layer, 0.2, base_image, 0.8, 0.0)
    if component_bbox:
        x = int(component_bbox["x"])
        y = int(component_bbox["y"])
        w = int(component_bbox["width"])
        h = int(component_bbox["height"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (56, 128, 78), 3, cv2.LINE_AA)
        cv2.putText(
            overlay,
            "FOCUS BOX",
            (x + 8, max(30, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (56, 128, 78),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def build_bundle_projection_from_inputs(items: list[dict[str, object]], bundle_code: str) -> dict[str, object]:
    if not items:
        return build_bundle_projection_unavailable("当前批次暂无可用于构件回投的图像。")

    contexts = [context for item in items if (context := build_bundle_projection_context(item)) is not None]
    if not contexts:
        return build_bundle_projection_unavailable("当前批次图像无法读取，未生成长景回投结果。")

    context_by_scale = {str(context["capture_scale"]): context for context in contexts}
    long_context = context_by_scale.get("长景")
    if long_context is None:
        return build_bundle_projection_unavailable("缺少长景图像，无法生成构件框选结果。")

    long_target = long_context.get("projection_target")
    if not long_target or not long_context.get("target_key"):
        return build_bundle_projection_unavailable("长景图像未识别到可用于定位的二维码靶标，无法回投中近景结果。")

    long_target_key = str(long_context["target_key"])
    projected_items: list[dict[str, object]] = []
    all_projected_contours: list[np.ndarray] = []
    for scale in ("中景", "近景"):
        source_context = context_by_scale.get(scale)
        if source_context is None:
            continue
        if not source_context.get("projection_target") or source_context.get("target_key") != long_target_key:
            continue

        projected_contours = project_contours_to_long_view(
            contours=source_context.get("mask_contours") or [],
            source_target=source_context["projection_target"],
            long_target=long_target,
            long_image_shape=np.asarray(long_context["image"]).shape,
        )
        if not projected_contours:
            continue

        projected_points = np.concatenate(projected_contours, axis=0)
        contour_bbox = compute_projection_bbox(projected_contours, np.asarray(long_context["image"]).shape)
        centroid = [
            round(float(projected_points[:, 0].mean()), 1),
            round(float(projected_points[:, 1].mean()), 1),
        ]
        projected_items.append(
            {
                "capture_scale": scale,
                "detection_code": source_context.get("detection_code"),
                "target_id": source_context["projection_target"].get("target_id"),
                "projected_contours": projected_contours,
                "projected_contour_count": len(projected_contours),
                "centroid": centroid,
                "bbox": contour_bbox,
            }
        )
        all_projected_contours.extend(projected_contours)

    if not projected_items:
        return build_bundle_projection_unavailable("未找到与长景同一靶标的中景/近景有效结果，暂无法完成回投。")

    component_bbox = compute_projection_bbox(all_projected_contours, np.asarray(long_context["image"]).shape)
    output_path = SEGMENTATION_ROOT / bundle_code / f"{bundle_code}_long_projection.png"
    draw_bundle_projection_overlay(long_context, projected_items, component_bbox, output_path)

    matched_scales = "、".join(item["capture_scale"] for item in projected_items)
    return {
        "status": "completed",
        "message": f"已将{matched_scales}裂缝结果回投到长景图像，并生成构件关注框。",
        "overview_image_url": build_file_url(output_path),
        "projected_items": [
            {
                "capture_scale": item["capture_scale"],
                "detection_code": item["detection_code"],
                "target_id": item["target_id"],
                "projected_contour_count": item["projected_contour_count"],
                "bbox": item["bbox"],
            }
            for item in projected_items
        ],
        "component_bbox": component_bbox,
    }


def build_bundle_projection_from_results(results: list[dict[str, object]], bundle_code: str) -> dict[str, object]:
    return build_bundle_projection_from_inputs(collect_bundle_projection_inputs_from_results(results), bundle_code)


def build_bundle_projection_from_records(
    records: list["ImageSegmentationRecord"],
    bundle_code: str,
) -> dict[str, object]:
    return build_bundle_projection_from_inputs(collect_bundle_projection_inputs_from_records(records), bundle_code)


def build_trend_summary(records: list["ImageSegmentationRecord"]) -> dict[str, object]:
    usable_records = [
        record
        for record in records
        if record.analysis_status == "completed"
        and (record.max_width_mm is not None or record.crack_length_mm is not None or record.crack_area_ratio is not None)
    ]

    if len(usable_records) < 2:
        return {
            "status": "insufficient",
            "message": "至少完成两次同一房屋的检测后，系统才会给出发展趋势。",
        }

    latest = usable_records[0]
    previous = usable_records[1]
    metric_name, latest_value = get_record_trend_metric(latest)
    _previous_metric_name, previous_value = get_record_trend_metric(previous)
    delta_ratio = round(latest_value - previous_value, 6)
    previous_ratio = previous_value or 0.0
    delta_percent = None
    if previous_ratio > 0:
        delta_percent = round(delta_ratio / previous_ratio * 100, 2)

    if delta_percent is not None and delta_percent >= 80:
        status = "rapid"
        message = f"最近一次检测显示关键指标 {metric_name} 增长较快，建议尽快复核。"
    elif delta_percent is not None and delta_percent >= 20:
        status = "growing"
        message = f"关键指标 {metric_name} 较上次有所增加，建议缩短复拍周期。"
    elif delta_percent is not None and delta_percent <= -15:
        status = "improving"
        message = f"关键指标 {metric_name} 较上次下降，裂缝表现暂时稳定。"
    else:
        status = "stable"
        message = f"近两次检测的关键指标 {metric_name} 变化不大，当前趋势整体平稳。"

    return {
        "status": status,
        "message": message,
        "latest_detection_code": latest.detection_code,
        "previous_detection_code": previous.detection_code,
        "metric_name": metric_name,
        "latest_value": latest_value,
        "previous_value": previous_value,
        "delta_area_ratio": delta_ratio,
        "delta_percent": delta_percent,
    }


def load_report_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in REPORT_FONT_CANDIDATES:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size, index=0 if not bold else 0)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_spacing: int = 10,
) -> int:
    x, y = position
    chars_per_line = max(8, max_width // max(font.size if hasattr(font, "size") else 12, 12))
    wrapped_lines: list[str] = []
    for paragraph in str(text).splitlines() or [str(text)]:
        wrapped_lines.extend(textwrap.wrap(paragraph, width=chars_per_line) or [""])
    for line in wrapped_lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += font.size + line_spacing if hasattr(font, "size") else 22
    return y


def build_risk_level_thresholds() -> list[dict[str, object]]:
    return [
        {
            "risk_level": risk_level,
            "min_score": min_score,
            "max_score": max_score,
        }
        for risk_level, min_score, max_score in RISK_LEVEL_THRESHOLDS
    ]


def add_report_header(
    draw: ImageDraw.ImageDraw,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    page_index: int,
) -> int:
    draw.rounded_rectangle((REPORT_MARGIN, 40, REPORT_PAGE_SIZE[0] - REPORT_MARGIN, 160), radius=26, fill="#EAF4F1")
    draw.text((REPORT_MARGIN + 20, 58), "房屋裂缝智能检测报告", fill="#123B3A", font=title_font)
    draw.text(
        (REPORT_MARGIN + 22, 112),
        f"第 {page_index} 页  |  本报告由系统自动生成，供复核与归档使用",
        fill="#4E6472",
        font=body_font,
    )
    return 190


def add_key_value_block(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, str]],
    start_y: int,
    columns: int = 2,
) -> int:
    label_font = load_report_font(24)
    value_font = load_report_font(30, bold=True)
    block_width = (REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 20 * (columns - 1)) // columns
    row_height = 118

    for index, (label, value) in enumerate(items):
        row = index // columns
        col = index % columns
        left = REPORT_MARGIN + col * (block_width + 20)
        top = start_y + row * (row_height + 14)
        draw.rounded_rectangle((left, top, left + block_width, top + row_height), radius=18, fill="#F8FBFC", outline="#D8E6E6")
        draw.text((left + 20, top + 18), label, fill="#5D7284", font=label_font)
        draw.text((left + 20, top + 52), value, fill="#183042", font=value_font)

    rows = (len(items) + columns - 1) // columns
    return start_y + rows * (row_height + 14)


def add_section_title(draw: ImageDraw.ImageDraw, title: str, start_y: int) -> int:
    section_font = load_report_font(34, bold=True)
    draw.text((REPORT_MARGIN, start_y), title, fill="#183042", font=section_font)
    draw.line((REPORT_MARGIN, start_y + 48, REPORT_PAGE_SIZE[0] - REPORT_MARGIN, start_y + 48), fill="#D8E6E6", width=3)
    return start_y + 68


def add_paragraph_card(
    draw: ImageDraw.ImageDraw,
    title: str,
    body: str,
    start_y: int,
    height: int = 152,
) -> int:
    title_font = load_report_font(26, bold=True)
    body_font = load_report_font(24)
    draw.rounded_rectangle(
        (REPORT_MARGIN, start_y, REPORT_PAGE_SIZE[0] - REPORT_MARGIN, start_y + height),
        radius=18,
        fill="#F8FBFC",
        outline="#D8E6E6",
    )
    draw.text((REPORT_MARGIN + 20, start_y + 18), title, fill="#183042", font=title_font)
    end_y = draw_wrapped_text(
        draw,
        body,
        (REPORT_MARGIN + 20, start_y + 58),
        body_font,
        "#425A6A",
        REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 40,
    )
    return max(start_y + height, end_y + 18)


def add_image_triptych(record: "ImageSegmentationRecord") -> Image.Image:
    canvas = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    caption_font = load_report_font(24, bold=True)
    current_y = add_report_header(draw, title_font, body_font, 2)
    current_y = add_section_title(draw, "结果图像与历史趋势", current_y)

    image_labels = [
        ("原图", Path(record.source_image_path)),
        ("裂缝掩码", Path(record.mask_path) if record.mask_path else None),
        ("叠加结果", Path(record.overlay_path) if record.overlay_path else None),
    ]

    card_width = (REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 32) // 3
    card_height = 430

    for index, (label, image_path) in enumerate(image_labels):
        left = REPORT_MARGIN + index * (card_width + 16)
        top = current_y
        draw.rounded_rectangle((left, top, left + card_width, top + card_height), radius=18, fill="#F8FBFC", outline="#D8E6E6")
        draw.text((left + 18, top + 18), label, fill="#183042", font=caption_font)
        image_box = (left + 18, top + 62, left + card_width - 18, top + card_height - 18)
        if image_path and image_path.exists():
            with Image.open(image_path) as source_image:
                thumbnail = ImageOps.exif_transpose(source_image).convert("RGB")
                thumbnail.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
                paste_x = image_box[0] + (image_box[2] - image_box[0] - thumbnail.width) // 2
                paste_y = image_box[1] + (image_box[3] - image_box[1] - thumbnail.height) // 2
                canvas.paste(thumbnail, (paste_x, paste_y))
        else:
            draw.rounded_rectangle(image_box, radius=12, outline="#D8E6E6", fill="#FFFFFF")
            draw.text((image_box[0] + 28, image_box[1] + 140), "当前无可用图像", fill="#7A8F9C", font=body_font)

    return canvas


def add_recognition_visual_page(record: "ImageSegmentationRecord") -> Image.Image:
    canvas = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    caption_font = load_report_font(24, bold=True)
    current_y = add_report_header(draw, title_font, body_font, 2)
    current_y = add_section_title(draw, "三、识别阶段图像摘要", current_y)

    image_labels = [
        ("原图", Path(record.source_image_path)),
        ("裂缝掩码", Path(record.mask_path) if record.mask_path else None),
        ("识别叠加图", Path(record.overlay_path) if record.overlay_path else None),
        ("靶标识别图", Path(record.target_overlay_path) if record.target_overlay_path else None),
    ]
    card_width = (REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 20) // 2
    card_height = 340
    for index, (label, image_path) in enumerate(image_labels):
        row = index // 2
        col = index % 2
        left = REPORT_MARGIN + col * (card_width + 20)
        top = current_y + row * (card_height + 18)
        draw.rounded_rectangle((left, top, left + card_width, top + card_height), radius=18, fill="#F8FBFC", outline="#D8E6E6")
        draw.text((left + 18, top + 18), label, fill="#183042", font=caption_font)
        image_box = (left + 18, top + 62, left + card_width - 18, top + card_height - 18)
        if image_path and image_path.exists():
            with Image.open(image_path) as source_image:
                thumbnail = ImageOps.exif_transpose(source_image).convert("RGB")
                thumbnail.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
                paste_x = image_box[0] + (image_box[2] - image_box[0] - thumbnail.width) // 2
                paste_y = image_box[1] + (image_box[3] - image_box[1] - thumbnail.height) // 2
                canvas.paste(thumbnail, (paste_x, paste_y))
        else:
            draw.rounded_rectangle(image_box, radius=12, outline="#D8E6E6", fill="#FFFFFF")
            draw.text((image_box[0] + 28, image_box[1] + 120), "当前无可用图像", fill="#7A8F9C", font=body_font)

    current_y += card_height * 2 + 36
    quality_warnings = decode_json(record.quality_warnings_json, [])
    warning_text = "；".join(quality_warnings) if quality_warnings else "当前图像未发现明显质量问题。"
    current_y = add_paragraph_card(draw, "质量提示", warning_text, current_y, height=132)
    current_y += 16
    stage_summary = build_analysis_stages(
        {
            "status": record.marker_status,
            "marker_count": record.marker_count,
            "physical_scale_mm_per_pixel": record.physical_scale_mm_per_pixel,
        },
        {
            "status": record.quantification_status,
        },
    )
    add_paragraph_card(draw, "阶段说明", stage_summary["recognition"]["summary"], current_y, height=132)
    return canvas


def add_trend_chart(
    draw: ImageDraw.ImageDraw,
    records: list["ImageSegmentationRecord"],
    start_y: int,
) -> int:
    title_font = load_report_font(26, bold=True)
    body_font = load_report_font(22)
    small_font = load_report_font(18)
    chart_top = start_y + 54
    chart_height = 300
    chart_left = REPORT_MARGIN + 40
    chart_right = REPORT_PAGE_SIZE[0] - REPORT_MARGIN - 40
    chart_bottom = chart_top + chart_height

    draw.rounded_rectangle(
        (REPORT_MARGIN, start_y, REPORT_PAGE_SIZE[0] - REPORT_MARGIN, chart_bottom + 84),
        radius=18,
        fill="#F8FBFC",
        outline="#D8E6E6",
    )
    metric_name, _initial_value = get_record_trend_metric(records[0]) if records else ("crack_area_ratio", 0.0)
    chart_title, metric_unit = get_trend_metric_label(metric_name)
    draw.text((REPORT_MARGIN + 20, start_y + 18), chart_title, fill="#183042", font=title_font)

    if len(records) < 2:
        draw.text((REPORT_MARGIN + 20, chart_top + 120), "当前历史数据不足，至少两次检测后才会生成趋势曲线。", fill="#7A8F9C", font=body_font)
        return chart_bottom + 84

    ordered_records = list(reversed(records))
    values = [get_record_trend_metric(record)[1] for record in ordered_records]
    max_value = max(values)
    min_value = min(values)
    minimum_span = 0.0005 if metric_unit == "ratio" else 0.05
    span = max(max_value - min_value, max_value * 0.2, minimum_span)
    y_min = max(0.0, min_value - span * 0.15) if metric_unit != "ratio" else max(0.0, min_value - span * 0.15)
    y_max = max_value + span * 0.15

    for index in range(5):
        ratio = index / 4
        y = chart_bottom - int(chart_height * ratio)
        value = y_min + (y_max - y_min) * ratio
        draw.line((chart_left, y, chart_right, y), fill="#E2EAEE", width=2)
        draw.text(
            (REPORT_MARGIN + 4, y - 10),
            f"{value:.4f}" if metric_unit == "ratio" else f"{value:.2f}",
            fill="#6A7F8E",
            font=small_font,
        )

    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="#90A4AE", width=3)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="#90A4AE", width=3)

    step = (chart_right - chart_left) / max(1, len(ordered_records) - 1)
    points: list[tuple[int, int]] = []
    for index, record in enumerate(ordered_records):
        value = get_record_trend_metric(record)[1]
        x = int(chart_left + step * index)
        normalized = (value - y_min) / max(y_max - y_min, 1e-9)
        y = int(chart_bottom - normalized * chart_height)
        points.append((x, y))
        draw.text((x - 26, chart_bottom + 16), format_report_timestamp(record)[5:], fill="#6A7F8E", font=small_font)

    if len(points) >= 2:
        draw.line(points, fill="#2A7F62", width=6, joint="curve")

    for point, record, value in zip(points, ordered_records, values):
        risk_color = {
            "无风险": "#4C9A2A",
            "低风险": "#C9941A",
            "中风险": "#D66A1F",
            "高风险": "#C43D2C",
        }.get(record.risk_level or "", "#2A7F62")
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=risk_color, outline="white", width=3)
        draw.text(
            (point[0] - 18, point[1] - 34),
            f"{value:.4f}" if metric_unit == "ratio" else f"{value:.2f}",
            fill="#183042",
            font=small_font,
        )

    draw.text(
        (REPORT_MARGIN + 20, chart_bottom + 48),
        "说明：曲线优先展示毫米级量测值变化；若缺少物理尺度，则回退展示裂缝占比。点位颜色对应当次风险等级。",
        fill="#6A7F8E",
        font=body_font,
    )
    return chart_bottom + 84


def add_history_comparison_grid(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    records: list["ImageSegmentationRecord"],
    start_y: int,
) -> int:
    title_font = load_report_font(26, bold=True)
    body_font = load_report_font(20)
    small_font = load_report_font(18)
    draw.rounded_rectangle(
        (REPORT_MARGIN, start_y, REPORT_PAGE_SIZE[0] - REPORT_MARGIN, start_y + 420),
        radius=18,
        fill="#F8FBFC",
        outline="#D8E6E6",
    )
    draw.text((REPORT_MARGIN + 20, start_y + 18), "多次检测对比图", fill="#183042", font=title_font)

    if not records:
        draw.text((REPORT_MARGIN + 20, start_y + 120), "暂无可用于对比的历史图像。", fill="#7A8F9C", font=body_font)
        return start_y + 420

    compare_records = records[:3]
    card_width = (REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 32) // 3
    card_top = start_y + 64
    card_height = 320

    for index, record in enumerate(compare_records):
        left = REPORT_MARGIN + index * (card_width + 16)
        draw.rounded_rectangle((left, card_top, left + card_width, card_top + card_height), radius=16, fill="#FFFFFF", outline="#D8E6E6")
        draw.text((left + 14, card_top + 12), record.detection_code or f"记录 {record.id}", fill="#183042", font=body_font)
        draw.text((left + 14, card_top + 40), format_report_timestamp(record), fill="#6A7F8E", font=small_font)
        image_path = Path(record.overlay_path) if record.overlay_path and Path(record.overlay_path).exists() else Path(record.source_image_path)
        image_box = (left + 14, card_top + 72, left + card_width - 14, card_top + 218)
        if image_path.exists():
            with Image.open(image_path) as comparison_image:
                thumbnail = ImageOps.exif_transpose(comparison_image).convert("RGB")
                thumbnail.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
                paste_x = image_box[0] + (image_box[2] - image_box[0] - thumbnail.width) // 2
                paste_y = image_box[1] + (image_box[3] - image_box[1] - thumbnail.height) // 2
                canvas.paste(thumbnail, (paste_x, paste_y))
        else:
            draw.rounded_rectangle(image_box, radius=10, fill="#F8FBFC", outline="#D8E6E6")
            draw.text((image_box[0] + 22, image_box[1] + 54), "无图像", fill="#7A8F9C", font=body_font)

        metric_y = card_top + 236
        draw.text((left + 14, metric_y), f"风险：{record.risk_level or '-'}", fill="#425A6A", font=small_font)
        draw.text((left + 14, metric_y + 28), f"占比：{float(record.crack_area_ratio or 0):.6f}", fill="#425A6A", font=small_font)
        draw.text((left + 14, metric_y + 56), f"质量：{record.quality_status or '-'} / {record.quality_score or '-'}", fill="#425A6A", font=small_font)

    return start_y + 420


def format_metric_value(mm_value: float | None, px_value: float | None, suffix: str = "") -> str:
    if mm_value is not None:
        return f"{mm_value:.3f} mm{suffix}"
    if px_value is not None:
        return f"{px_value:.3f} px{suffix}"
    return "-"


def format_risk_threshold_text(risk_assessment: dict[str, object]) -> str:
    thresholds = risk_assessment.get("level_thresholds") or []
    parts = []
    for item in thresholds:
        min_score = item.get("min_score")
        max_score = item.get("max_score")
        range_text = f"{min_score}+" if max_score is None else f"{min_score}-{max_score}"
        parts.append(f"{item.get('risk_level', '-')} {range_text} 分")
    return " / ".join(parts)


def build_report_risk_detail_text(risk_assessment: dict[str, object]) -> str:
    lines = [f"摘要：{risk_assessment.get('risk_summary') or '暂无风险说明。'}"]
    score_breakdown = risk_assessment.get("score_breakdown") or []
    if score_breakdown:
        for item in score_breakdown[:5]:
            lines.append(f"+{item.get('score', 0)} 分｜{item.get('detail', '-')}")
    else:
        lines.append("评分依据：本次未触发宽度、长度、构件类型或历史增长等加分项。")

    for note in (risk_assessment.get("explanation_notes") or [])[:2]:
        lines.append(f"附注：{note}")

    threshold_text = format_risk_threshold_text(risk_assessment)
    if threshold_text:
        lines.append(f"分级阈值：{threshold_text}")
    return "\n".join(lines)


def add_quantification_visual_page(record: "ImageSegmentationRecord") -> Image.Image:
    canvas = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    caption_font = load_report_font(24, bold=True)
    current_y = add_report_header(draw, title_font, body_font, 4)
    current_y = add_section_title(draw, "五、靶标识别与量化分析", current_y)

    current_y = add_key_value_block(
        draw,
        [
            ("靶标识别", record.marker_status or "-"),
            ("靶标数量", str(record.marker_count or 0)),
            ("物理比例", f"{record.physical_scale_mm_per_pixel:.6f} mm/pixel" if record.physical_scale_mm_per_pixel else "未建立"),
            ("量化状态", record.quantification_status or "-"),
            ("最大宽度", format_metric_value(record.max_width_mm, record.max_width_px)),
            ("平均宽度", format_metric_value(record.avg_width_mm, record.avg_width_px)),
            ("裂缝长度", format_metric_value(record.crack_length_mm, record.crack_length_px)),
            ("裂缝角度", f"{record.crack_angle_deg:.2f} deg" if record.crack_angle_deg is not None else "-"),
        ],
        current_y,
        columns=2,
    )

    current_y += 18
    current_y = add_paragraph_card(
        draw,
        "量化说明",
        record.quantification_message or "当前记录暂无量化结论。",
        current_y,
        height=122,
    )
    current_y += 18

    image_labels = [
        ("靶标识别图", Path(record.target_overlay_path) if record.target_overlay_path else None),
        ("量化叠加图", Path(record.quant_overlay_path) if record.quant_overlay_path else None),
        ("宽度统计图", Path(record.width_chart_path) if record.width_chart_path else None),
    ]
    card_width = (REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 32) // 3
    card_height = 500

    for index, (label, image_path) in enumerate(image_labels):
        left = REPORT_MARGIN + index * (card_width + 16)
        top = current_y
        draw.rounded_rectangle((left, top, left + card_width, top + card_height), radius=18, fill="#F8FBFC", outline="#D8E6E6")
        draw.text((left + 18, top + 18), label, fill="#183042", font=caption_font)
        image_box = (left + 18, top + 62, left + card_width - 18, top + card_height - 18)
        if image_path and image_path.exists():
            with Image.open(image_path) as image:
                thumbnail = ImageOps.exif_transpose(image).convert("RGB")
                thumbnail.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
                paste_x = image_box[0] + (image_box[2] - image_box[0] - thumbnail.width) // 2
                paste_y = image_box[1] + (image_box[3] - image_box[1] - thumbnail.height) // 2
                canvas.paste(thumbnail, (paste_x, paste_y))
        else:
            draw.rounded_rectangle(image_box, radius=12, outline="#D8E6E6", fill="#FFFFFF")
            draw.text((image_box[0] + 24, image_box[1] + 160), "当前无可用图像", fill="#7A8F9C", font=body_font)

    return canvas


def build_report_pages(
    record: "ImageSegmentationRecord",
    house: "HouseInfo | None",
    trend_summary: dict[str, object],
    generated_at: datetime | None,
) -> list[Image.Image]:
    history_records = get_house_history_records(record.house_id, limit=5) if record.house_id else []
    risk_assessment = build_risk_assessment_for_record(record)

    page_one = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page_one)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    current_y = add_report_header(draw, title_font, body_font, 1)

    current_y = add_section_title(draw, "一、检测概览", current_y)
    current_y = add_key_value_block(
        draw,
        [
            ("检测编号", record.detection_code or "-"),
            ("房屋编号", house.house_number if house else "未绑定"),
            ("上传时间", format_uploaded_timestamp(record, include_seconds=True)),
            ("生成时间", format_datetime_value(generated_at)),
            ("房屋类型", house.house_type if house and house.house_type else "未标注"),
            ("裂缝位置", house.crack_location if house and house.crack_location else "未标注"),
            ("检测任务", house.detection_type if house and house.detection_type else "未标注"),
            ("拍摄尺度", record.capture_scale or "未标注"),
            ("构件类型", record.component_type or "未标注"),
            ("场景类型", record.scenario_type or "未标注"),
        ],
        current_y,
        columns=2,
    )

    current_y += 18
    current_y = add_section_title(draw, "二、当前风险结论", current_y)
    current_y = add_key_value_block(
        draw,
        [
            ("质量状态", f"{record.quality_status or '-'} / {record.quality_score or '-'}"),
            ("风险等级", risk_assessment["risk_level"] or "待人工复核"),
            ("风险评分", f"{risk_assessment['score']} 分"),
            ("最大宽度", format_metric_value(record.max_width_mm, record.max_width_px)),
            ("裂缝长度", format_metric_value(record.crack_length_mm, record.crack_length_px)),
        ],
        current_y,
        columns=3,
    )

    current_y += 18
    current_y = add_paragraph_card(
        draw,
        "风险说明与评分依据",
        build_report_risk_detail_text(risk_assessment),
        current_y,
        height=242,
    )
    current_y += 14
    current_y = add_paragraph_card(
        draw,
        "处置建议",
        risk_assessment["recommendation"] or "暂无处置建议。",
        current_y,
        height=122,
    )

    page_two = add_recognition_visual_page(record)
    draw_two = ImageDraw.Draw(page_two)
    current_y = 700
    current_y = add_section_title(draw_two, "三、历史趋势与归档摘要", current_y)
    current_y = add_paragraph_card(
        draw_two,
        "趋势结论",
        trend_summary.get("message") or "当前检测次数不足，暂无法生成趋势判断。",
        current_y,
        height=136,
    )
    current_y += 16
    history_title_font = load_report_font(26, bold=True)
    history_body_font = load_report_font(22)
    draw_two.rounded_rectangle(
        (REPORT_MARGIN, current_y, REPORT_PAGE_SIZE[0] - REPORT_MARGIN, current_y + 420),
        radius=18,
        fill="#F8FBFC",
        outline="#D8E6E6",
    )
    draw_two.text((REPORT_MARGIN + 20, current_y + 18), "最近 5 次检测记录", fill="#183042", font=history_title_font)
    table_y = current_y + 64
    headers = ["编号", "时间", "风险", "占比", "质量"]
    widths = [280, 260, 150, 160, 180]
    start_x = REPORT_MARGIN + 20
    x = start_x
    for header, width in zip(headers, widths):
        draw_two.rounded_rectangle((x, table_y, x + width, table_y + 42), radius=10, fill="#EAF4F1")
        draw_two.text((x + 12, table_y + 9), header, fill="#183042", font=history_body_font)
        x += width + 10

    row_y = table_y + 58
    for row in history_records:
        values = [
            row.detection_code or "-",
            format_report_timestamp(row, include_seconds=True),
            row.risk_level or "-",
            f"{row.crack_area_ratio or 0:.6f}",
            f"{row.quality_status or '-'} / {row.quality_score or '-'}",
        ]
        x = start_x
        for value, width in zip(values, widths):
            draw_two.rounded_rectangle((x, row_y, x + width, row_y + 42), radius=10, fill="#FFFFFF", outline="#D8E6E6")
            draw_two.text((x + 10, row_y + 9), value[:24], fill="#425A6A", font=history_body_font)
            x += width + 10
        row_y += 54

    current_y += 440
    draw_two.text(
        (REPORT_MARGIN, current_y),
        "备注：本报告为系统自动分析结果，适用于筛查与档案管理；如结果为中高风险，请尽快安排专业人员现场复核。",
        fill="#7A8F9C",
        font=history_body_font,
    )
    page_three = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw_three = ImageDraw.Draw(page_three)
    current_y = add_report_header(draw_three, title_font, body_font, 3)
    current_y = add_section_title(draw_three, "四、趋势曲线与多次检测对比", current_y)
    current_y = add_trend_chart(draw_three, history_records, current_y)
    current_y += 18
    add_history_comparison_grid(page_three, draw_three, history_records, current_y)
    page_four = add_quantification_visual_page(record)

    return [page_one, page_two, page_three, page_four]


def build_report_pdf(record: "ImageSegmentationRecord", house: "HouseInfo | None", trend_summary: dict[str, object]) -> BytesIO:
    pages = build_report_pages(record, house, trend_summary, get_record_generated_at(record))
    buffer = BytesIO()
    rgb_pages = [page.convert("RGB") for page in pages]
    rgb_pages[0].save(buffer, format="PDF", save_all=True, append_images=rgb_pages[1:])
    buffer.seek(0)
    return buffer


def _legacy_build_bundle_summary_page(
    records: list["ImageSegmentationRecord"],
    house: "HouseInfo | None",
    bundle_code: str,
    generated_at: datetime | None,
) -> Image.Image:
    page = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    current_y = add_report_header(draw, title_font, body_font, 1)
    current_y = add_section_title(draw, "一、三景总报告概览", current_y)
    current_y = add_key_value_block(
        draw,
        [
            ("批次编号", bundle_code),
            ("房屋编号", house.house_number if house else "未绑定"),
            ("上传时间", format_datetime_value(get_bundle_uploaded_at(records))),
            ("生成时间", format_datetime_value(generated_at)),
            ("记录数量", str(len(records))),
            ("检测任务", house.detection_type if house and house.detection_type else "未标注"),
            ("房屋类型", house.house_type if house and house.house_type else "未标注"),
        ],
        current_y,
        columns=2,
    )
    current_y += 20
    current_y = add_section_title(draw, "二、长景 / 中景 / 近景摘要", current_y)

    header_font = load_report_font(22, bold=True)
    cell_font = load_report_font(20)
    table_y = current_y + 10
    headers = ["尺度", "检测编号", "生成时间", "风险", "最大宽度", "裂缝长度"]
    widths = [110, 250, 240, 120, 180, 180]
    start_x = REPORT_MARGIN + 12
    x = start_x
    for header, width in zip(headers, widths):
        draw.rounded_rectangle((x, table_y, x + width, table_y + 44), radius=10, fill="#EAF4F1")
        draw.text((x + 10, table_y + 10), header, fill="#183042", font=header_font)
        x += width + 10

    row_y = table_y + 60
    for record in records:
        values = [
            record.capture_scale or "-",
            record.detection_code or "-",
            format_report_timestamp(record, include_seconds=True),
            record.risk_level or "-",
            format_metric_value(record.max_width_mm, record.max_width_px),
            format_metric_value(record.crack_length_mm, record.crack_length_px),
        ]
        x = start_x
        for value, width in zip(values, widths):
            draw.rounded_rectangle((x, row_y, x + width, row_y + 48), radius=10, fill="#FFFFFF", outline="#D8E6E6")
            draw.text((x + 10, row_y + 12), value[:24], fill="#425A6A", font=cell_font)
            x += width + 10
        row_y += 60

    current_y = row_y + 18
    summary_lines = "；".join(
        f"{record.capture_scale or '未标注'}：{record.risk_level or '-'}，最大宽度 {format_metric_value(record.max_width_mm, record.max_width_px)}"
        for record in records
    )
    add_paragraph_card(
        draw,
        "总览说明",
        summary_lines or "当前批次暂无可汇总的结果。",
        current_y,
        height=150,
    )
    return page


def _legacy_build_bundle_record_page(
    record: "ImageSegmentationRecord",
    page_index: int,
    generated_at: datetime | None,
) -> Image.Image:
    page = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    caption_font = load_report_font(24, bold=True)
    current_y = add_report_header(draw, title_font, body_font, page_index)
    current_y = add_section_title(draw, f"{record.capture_scale or '未标注'}检测摘要", current_y)
    current_y = add_key_value_block(
        draw,
        [
            ("检测编号", record.detection_code or "-"),
            ("上传时间", format_uploaded_timestamp(record, include_seconds=True)),
            ("生成时间", format_report_timestamp(record, include_seconds=True)),
            ("风险等级", record.risk_level or "-"),
            ("最大宽度", format_metric_value(record.max_width_mm, record.max_width_px)),
            ("裂缝长度", format_metric_value(record.crack_length_mm, record.crack_length_px)),
            ("靶标状态", record.marker_status or "-"),
            ("物理比例", f"{record.physical_scale_mm_per_pixel:.6f} mm/pixel" if record.physical_scale_mm_per_pixel else "未建立"),
        ],
        current_y,
        columns=2,
    )
    current_y += 18
    current_y = add_paragraph_card(draw, "结论摘要", record.risk_summary or "暂无风险说明。", current_y, height=122)
    current_y += 18

    image_labels = [
        ("识别图", Path(record.overlay_path) if record.overlay_path else None),
        ("靶标图", Path(record.target_overlay_path) if record.target_overlay_path else None),
        ("量化图", Path(record.quant_overlay_path) if record.quant_overlay_path else None),
        ("统计图", Path(record.width_chart_path) if record.width_chart_path else None),
    ]
    card_width = (REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 20) // 2
    card_height = 340
    for index, (label, image_path) in enumerate(image_labels):
        row = index // 2
        col = index % 2
        left = REPORT_MARGIN + col * (card_width + 20)
        top = current_y + row * (card_height + 18)
        draw.rounded_rectangle((left, top, left + card_width, top + card_height), radius=18, fill="#F8FBFC", outline="#D8E6E6")
        draw.text((left + 18, top + 18), label, fill="#183042", font=caption_font)
        image_box = (left + 18, top + 62, left + card_width - 18, top + card_height - 18)
        if image_path and image_path.exists():
            with Image.open(image_path) as image:
                thumbnail = ImageOps.exif_transpose(image).convert("RGB")
                thumbnail.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
                paste_x = image_box[0] + (image_box[2] - image_box[0] - thumbnail.width) // 2
                paste_y = image_box[1] + (image_box[3] - image_box[1] - thumbnail.height) // 2
                page.paste(thumbnail, (paste_x, paste_y))
        else:
            draw.rounded_rectangle(image_box, radius=12, fill="#FFFFFF", outline="#D8E6E6")
            draw.text((image_box[0] + 24, image_box[1] + 120), "当前无可用图像", fill="#7A8F9C", font=body_font)
    return page


def build_bundle_summary_page(
    records: list["ImageSegmentationRecord"],
    house: "HouseInfo | None",
    bundle_code: str,
    generated_at: datetime | None,
) -> Image.Image:
    page = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    current_y = add_report_header(draw, title_font, body_font, 1)
    current_y = add_section_title(draw, "一、三景总报告概览", current_y)
    current_y = add_key_value_block(
        draw,
        [
            ("批次编号", bundle_code),
            ("房屋编号", house.house_number if house else "未绑定"),
            ("上传时间", format_datetime_value(get_bundle_uploaded_at(records))),
            ("生成时间", format_datetime_value(generated_at)),
            ("记录数量", str(len(records))),
            ("检测任务", house.detection_type if house and house.detection_type else "未标注"),
            ("房屋类型", house.house_type if house and house.house_type else "未标注"),
        ],
        current_y,
        columns=2,
    )
    current_y += 20
    current_y = add_section_title(draw, "二、长景 / 中景 / 近景摘要", current_y)

    header_font = load_report_font(22, bold=True)
    cell_font = load_report_font(20)
    table_y = current_y + 10
    headers = ["尺度", "检测编号", "生成时间", "风险", "最大宽度", "裂缝长度"]
    widths = [110, 250, 240, 120, 180, 180]
    start_x = REPORT_MARGIN + 12
    x = start_x
    for header, width in zip(headers, widths):
        draw.rounded_rectangle((x, table_y, x + width, table_y + 44), radius=10, fill="#EAF4F1")
        draw.text((x + 10, table_y + 10), header, fill="#183042", font=header_font)
        x += width + 10

    row_y = table_y + 60
    for record in records:
        values = [
            record.capture_scale or "-",
            record.detection_code or "-",
            format_report_timestamp(record, include_seconds=True),
            record.risk_level or "-",
            format_metric_value(record.max_width_mm, record.max_width_px),
            format_metric_value(record.crack_length_mm, record.crack_length_px),
        ]
        x = start_x
        for value, width in zip(values, widths):
            draw.rounded_rectangle((x, row_y, x + width, row_y + 48), radius=10, fill="#FFFFFF", outline="#D8E6E6")
            draw.text((x + 10, row_y + 12), value[:24], fill="#425A6A", font=cell_font)
            x += width + 10
        row_y += 60

    current_y = row_y + 18
    summary_lines = "；".join(
        f"{record.capture_scale or '未标注'}：{record.risk_level or '-'}，最大宽度 {format_metric_value(record.max_width_mm, record.max_width_px)}"
        for record in records
    )
    add_paragraph_card(
        draw,
        "总览说明",
        summary_lines or "当前批次暂无可汇总的结果。",
        current_y,
        height=150,
    )
    return page


def build_bundle_projection_page(
    bundle_projection: dict[str, object],
    generated_at: datetime | None,
    page_index: int,
) -> Image.Image:
    page = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    caption_font = load_report_font(24, bold=True)
    current_y = add_report_header(draw, title_font, body_font, page_index)
    current_y = add_section_title(draw, "三、长景构件框选与中近景回投", current_y)
    current_y = add_key_value_block(
        draw,
        [
            ("结果生成时间", format_datetime_value(generated_at)),
            ("回投状态", "已生成" if bundle_projection.get("status") == "completed" else "未生成"),
            ("回投尺度", "、".join(item["capture_scale"] for item in bundle_projection.get("projected_items", [])) or "-"),
            (
                "关注框",
                (
                    f"x={bundle_projection['component_bbox']['x']} y={bundle_projection['component_bbox']['y']} "
                    f"w={bundle_projection['component_bbox']['width']} h={bundle_projection['component_bbox']['height']}"
                )
                if bundle_projection.get("component_bbox")
                else "-",
            ),
        ],
        current_y,
        columns=2,
    )
    current_y += 18
    current_y = add_paragraph_card(
        draw,
        "回投说明",
        bundle_projection.get("message") or "当前批次未生成长景构件回投结果。",
        current_y,
        height=132,
    )
    current_y += 18

    image_path = None
    image_url = bundle_projection.get("overview_image_url")
    if image_url and image_url.startswith("/files/"):
        image_path = UPLOAD_ROOT / image_url.replace("/files/", "", 1)

    draw.rounded_rectangle(
        (REPORT_MARGIN, current_y, REPORT_PAGE_SIZE[0] - REPORT_MARGIN, current_y + 760),
        radius=18,
        fill="#F8FBFC",
        outline="#D8E6E6",
    )
    draw.text((REPORT_MARGIN + 20, current_y + 18), "长景回投总览图", fill="#183042", font=caption_font)
    image_box = (REPORT_MARGIN + 20, current_y + 62, REPORT_PAGE_SIZE[0] - REPORT_MARGIN - 20, current_y + 720)
    if image_path and image_path.exists():
        with Image.open(image_path) as projection_image:
            thumbnail = ImageOps.exif_transpose(projection_image).convert("RGB")
            thumbnail.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
            paste_x = image_box[0] + (image_box[2] - image_box[0] - thumbnail.width) // 2
            paste_y = image_box[1] + (image_box[3] - image_box[1] - thumbnail.height) // 2
            page.paste(thumbnail, (paste_x, paste_y))
    else:
        draw.rounded_rectangle(image_box, radius=12, fill="#FFFFFF", outline="#D8E6E6")
        draw.text((image_box[0] + 24, image_box[1] + 180), "当前无可用回投图像", fill="#7A8F9C", font=body_font)

    return page


def build_bundle_record_page(
    record: "ImageSegmentationRecord",
    page_index: int,
    generated_at: datetime | None,
) -> Image.Image:
    page = Image.new("RGB", REPORT_PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_report_font(44, bold=True)
    body_font = load_report_font(24)
    caption_font = load_report_font(24, bold=True)
    current_y = add_report_header(draw, title_font, body_font, page_index)
    current_y = add_section_title(draw, f"{record.capture_scale or '未标注'}检测摘要", current_y)
    current_y = add_key_value_block(
        draw,
        [
            ("检测编号", record.detection_code or "-"),
            ("上传时间", format_uploaded_timestamp(record, include_seconds=True)),
            ("生成时间", format_report_timestamp(record, include_seconds=True)),
            ("风险等级", record.risk_level or "-"),
            ("最大宽度", format_metric_value(record.max_width_mm, record.max_width_px)),
            ("裂缝长度", format_metric_value(record.crack_length_mm, record.crack_length_px)),
            ("靶标状态", record.marker_status or "-"),
            ("物理比例", f"{record.physical_scale_mm_per_pixel:.6f} mm/pixel" if record.physical_scale_mm_per_pixel else "未建立"),
        ],
        current_y,
        columns=2,
    )
    current_y += 18
    current_y = add_paragraph_card(draw, "结论摘要", record.risk_summary or "暂无风险说明。", current_y, height=122)
    current_y += 18

    image_labels = [
        ("识别图", Path(record.overlay_path) if record.overlay_path else None),
        ("靶标图", Path(record.target_overlay_path) if record.target_overlay_path else None),
        ("量化图", Path(record.quant_overlay_path) if record.quant_overlay_path else None),
        ("统计图", Path(record.width_chart_path) if record.width_chart_path else None),
    ]
    card_width = (REPORT_PAGE_SIZE[0] - REPORT_MARGIN * 2 - 20) // 2
    card_height = 340
    for index, (label, image_path) in enumerate(image_labels):
        row = index // 2
        col = index % 2
        left = REPORT_MARGIN + col * (card_width + 20)
        top = current_y + row * (card_height + 18)
        draw.rounded_rectangle((left, top, left + card_width, top + card_height), radius=18, fill="#F8FBFC", outline="#D8E6E6")
        draw.text((left + 18, top + 18), label, fill="#183042", font=caption_font)
        image_box = (left + 18, top + 62, left + card_width - 18, top + card_height - 18)
        if image_path and image_path.exists():
            with Image.open(image_path) as image:
                thumbnail = ImageOps.exif_transpose(image).convert("RGB")
                thumbnail.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
                paste_x = image_box[0] + (image_box[2] - image_box[0] - thumbnail.width) // 2
                paste_y = image_box[1] + (image_box[3] - image_box[1] - thumbnail.height) // 2
                page.paste(thumbnail, (paste_x, paste_y))
        else:
            draw.rounded_rectangle(image_box, radius=12, fill="#FFFFFF", outline="#D8E6E6")
            draw.text((image_box[0] + 24, image_box[1] + 120), "当前无可用图像", fill="#7A8F9C", font=body_font)
    return page


def build_bundle_report_pdf(
    records: list["ImageSegmentationRecord"],
    house: "HouseInfo | None",
    bundle_code: str,
) -> BytesIO:
    generated_at = get_bundle_generated_at(records)
    pages: list[Image.Image] = [build_bundle_summary_page(records, house, bundle_code, generated_at)]
    try:
        bundle_projection = build_bundle_projection_from_records(records, bundle_code)
    except Exception:  # pragma: no cover
        bundle_projection = build_bundle_projection_unavailable("长景构件回投生成失败。")
    if bundle_projection.get("status") == "completed":
        pages.append(build_bundle_projection_page(bundle_projection, generated_at, len(pages) + 1))

    for index, record in enumerate(records, start=len(pages) + 1):
        pages.append(build_bundle_record_page(record, index, generated_at))

    buffer = BytesIO()
    rgb_pages = [page.convert("RGB") for page in pages]
    rgb_pages[0].save(buffer, format="PDF", save_all=True, append_images=rgb_pages[1:])
    buffer.seek(0)
    return buffer


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    return (
        jsonify(
            {
                "message": "上传失败：本次请求体过大。当前最多支持 100MB，请压缩图片或分批上传。",
            }
        ),
        413,
    )


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def serialize_segmentation_record(
    record: ImageSegmentationRecord,
    previous_record: "ImageSegmentationRecord | None" = None,
    risk_assessment: dict[str, object] | None = None,
) -> dict:
    marker_status = record.marker_status
    if marker_status == "anchor_rectified":
        detection_method = "qr_anchor_rectified"
        perspective_corrected = True
    elif marker_status == "qr_detected":
        detection_method = "qr_decode"
        perspective_corrected = False
    elif marker_status == "fallback_detected":
        detection_method = "legacy_rect"
        perspective_corrected = False
    else:
        detection_method = None
        perspective_corrected = False

    marker_detection = {
        "status": marker_status,
        "marker_count": record.marker_count,
        "physical_scale_mm_per_pixel": record.physical_scale_mm_per_pixel,
        "detection_method": detection_method,
        "detection_method_label": format_marker_mode_label(detection_method),
        "perspective_corrected": perspective_corrected,
        "annotated_image_url": build_file_url(Path(record.target_overlay_path)) if record.target_overlay_path else None,
    }
    quantification = {
        "status": record.quantification_status,
        "message": record.quantification_message,
        "plane_mode": "rectified_plane" if perspective_corrected and record.quantification_status == "completed" else "image_plane",
        "component_count": record.component_count,
        "sample_count": record.sample_count,
        "max_width_px": record.max_width_px,
        "avg_width_px": record.avg_width_px,
        "median_width_px": record.median_width_px,
        "crack_length_px": record.crack_length_px,
        "crack_angle_deg": record.crack_angle_deg,
        "max_width_mm": record.max_width_mm,
        "avg_width_mm": record.avg_width_mm,
        "median_width_mm": record.median_width_mm,
        "crack_length_mm": record.crack_length_mm,
        "quant_overlay_url": build_file_url(Path(record.quant_overlay_path)) if record.quant_overlay_path else None,
        "width_chart_url": build_file_url(Path(record.width_chart_path)) if record.width_chart_path else None,
    }
    if risk_assessment is None:
        risk_assessment = build_risk_assessment_for_record(record, previous_record=previous_record)
    uploaded_at = get_record_uploaded_at(record)
    generated_at = get_record_generated_at(record)
    return {
        "id": record.id,
        "house_id": record.house_id,
        "upload_bundle_code": record.upload_bundle_code,
        "target_id": record.target_id,
        "bundle_report_url": url_for("download_bundle_report", bundle_code=record.upload_bundle_code)
        if record.upload_bundle_code
        else None,
        "detection_code": record.detection_code,
        "report_url": url_for("download_detection_report", record_id=record.id),
        "analysis_status": record.analysis_status,
        "original_filename": record.original_filename,
        "source_image_url": build_file_url(Path(record.source_image_path)),
        "mask_url": build_file_url(Path(record.mask_path)) if record.mask_path else None,
        "overlay_url": build_file_url(Path(record.overlay_path)) if record.overlay_path else None,
        "crack_pixel_count": record.crack_pixel_count,
        "crack_area_ratio": record.crack_area_ratio,
        "inference_device": record.inference_device,
        "patch_count": record.patch_count,
        "capture_scale": record.capture_scale,
        "capture_scale_label": format_capture_scale_label(record.capture_scale),
        "component_type": record.component_type,
        "scenario_type": record.scenario_type,
        "quality_score": record.quality_score,
        "quality_status": record.quality_status,
        "quality_warnings": decode_json(record.quality_warnings_json, []),
        "risk_level": risk_assessment["risk_level"],
        "risk_score": risk_assessment["score"],
        "risk_summary": risk_assessment["risk_summary"],
        "recommendation": risk_assessment["recommendation"],
        "risk_assessment": risk_assessment,
        "marker_detection": marker_detection,
        "quantification": quantification,
        "analysis_stages": build_analysis_stages(marker_detection, quantification),
        "uploaded_at": serialize_datetime_value(uploaded_at),
        "uploaded_at_display": format_datetime_value(uploaded_at),
        "generated_at": serialize_datetime_value(generated_at),
        "generated_at_display": format_datetime_value(generated_at),
        "created_at": serialize_datetime_value(record.created_at),
        "created_at_display": format_datetime_value(record.created_at),
    }


def build_survey_summary(records: list["ImageSegmentationRecord"]) -> dict[str, object]:
    completed_records = [record for record in records if record.analysis_status == "completed"]
    risk_distribution = {"无风险": 0, "低风险": 0, "中风险": 0, "高风险": 0}
    houses_with_records = {record.house_id for record in completed_records if record.house_id}
    history_count_by_house: dict[int, int] = {}

    latest_by_house: dict[int, ImageSegmentationRecord] = {}
    for record in completed_records:
        if record.risk_level in risk_distribution:
            risk_distribution[record.risk_level] += 1
        if not record.house_id:
            continue
        history_count_by_house[record.house_id] = history_count_by_house.get(record.house_id, 0) + 1
        existing = latest_by_house.get(record.house_id)
        if existing is None or (record.created_at, record.id) > (existing.created_at, existing.id):
            latest_by_house[record.house_id] = record

    high_risk_houses: list[dict[str, object]] = []
    houses_with_history: list[dict[str, object]] = []
    for house_id, record in latest_by_house.items():
        house = db.session.get(HouseInfo, house_id)
        houses_with_history.append(
            {
                "house_id": house_id,
                "house_number": house.house_number if house else f"HOUSE-{house_id}",
                "history_count": history_count_by_house.get(house_id, 0),
                "latest_detection_code": record.detection_code,
                "latest_report_url": url_for("download_detection_report", record_id=record.id),
                "latest_report_view_url": url_for("download_detection_report", record_id=record.id, download=0),
                "latest_bundle_report_url": url_for("download_bundle_report", bundle_code=record.upload_bundle_code)
                if record.upload_bundle_code
                else None,
                "latest_component_type": record.component_type,
                "latest_scenario_type": record.scenario_type,
                "latest_risk_level": record.risk_level,
                "latest_created_at": serialize_datetime_value(record.created_at),
                "latest_created_at_display": format_datetime_value(record.created_at),
                "latest_uploaded_at": serialize_datetime_value(get_record_uploaded_at(record)),
                "latest_uploaded_at_display": format_datetime_value(get_record_uploaded_at(record)),
                "latest_generated_at": serialize_datetime_value(get_record_generated_at(record)),
                "latest_generated_at_display": format_datetime_value(get_record_generated_at(record)),
            }
        )
        if record.risk_level not in {"中风险", "高风险"}:
            continue
        house = db.session.get(HouseInfo, house_id)
        high_risk_houses.append(
            {
                "house_id": house_id,
                "house_number": house.house_number if house else f"HOUSE-{house_id}",
                "risk_level": record.risk_level,
                "max_width_mm": record.max_width_mm,
                "crack_length_mm": record.crack_length_mm,
                "component_type": record.component_type,
                "scenario_type": record.scenario_type,
                "detection_code": record.detection_code,
                "created_at": serialize_datetime_value(record.created_at),
                "created_at_display": format_datetime_value(record.created_at),
                "uploaded_at": serialize_datetime_value(get_record_uploaded_at(record)),
                "uploaded_at_display": format_datetime_value(get_record_uploaded_at(record)),
                "generated_at": serialize_datetime_value(get_record_generated_at(record)),
                "generated_at_display": format_datetime_value(get_record_generated_at(record)),
            }
        )

    high_risk_houses.sort(
        key=lambda item: (
            0 if item["risk_level"] == "高风险" else 1,
            -(item["max_width_mm"] or 0.0),
            -(item["crack_length_mm"] or 0.0),
        )
    )

    houses_with_history.sort(
        key=lambda item: (
            item["latest_created_at"] or "",
            item["house_id"],
        ),
        reverse=True,
    )

    return {
        "total_records": len(records),
        "completed_records": len(completed_records),
        "house_count": len(houses_with_records),
        "risk_distribution": risk_distribution,
        "high_risk_houses": high_risk_houses[:20],
        "houses_with_history": houses_with_history,
    }


def serialize_detection_task(task: "DetectionTask", include_results: bool | None = None) -> dict[str, object]:
    if include_results is None:
        include_results = task.status in {"completed", "failed"}

    quality_results = decode_json(task.quality_results_json, [])
    results = decode_json(task.results_json, [])
    bundle_projection = decode_json(task.bundle_projection_json, None)
    progress_percent = 0
    if task.total_items:
        progress_percent = min(100, round(task.processed_items / task.total_items * 100))
    if task.status == "completed":
        progress_percent = 100

    payload = {
        "task_code": task.task_code,
        "status": task.status,
        "message": task.status_message,
        "house_id": task.house_id,
        "bundle_code": task.bundle_code,
        "total_items": task.total_items,
        "processed_items": task.processed_items,
        "progress_percent": progress_percent,
        "uploaded_at": serialize_datetime_value(task.created_at),
        "uploaded_at_display": format_datetime_value(task.created_at),
        "generated_at": serialize_datetime_value(task.completed_at),
        "generated_at_display": format_datetime_value(task.completed_at),
        "created_at": serialize_datetime_value(task.created_at),
        "created_at_display": format_datetime_value(task.created_at),
        "started_at": serialize_datetime_value(task.started_at),
        "started_at_display": format_datetime_value(task.started_at),
        "completed_at": serialize_datetime_value(task.completed_at),
        "completed_at_display": format_datetime_value(task.completed_at),
        "result_status_code": task.result_status_code,
        "has_errors": task.status == "failed" or task.result_status_code not in (None, 200),
        "error_detail": task.error_detail,
        "quality_results": quality_results,
        "status_url": url_for("get_detection_task", task_code=task.task_code),
    }

    if include_results:
        for result in results:
            result.setdefault("uploaded_at", serialize_datetime_value(task.created_at))
            result.setdefault("uploaded_at_display", format_datetime_value(task.created_at))
            result.setdefault("generated_at", result.get("created_at") or serialize_datetime_value(task.completed_at))
            result.setdefault("generated_at_display", result.get("created_at_display") or format_datetime_value(task.completed_at))
        payload["results"] = results
        payload["bundle_projection"] = bundle_projection or build_bundle_projection_unavailable("Projection not ready.")
        payload["bundle_report_url"] = (
            url_for("download_bundle_report", bundle_code=task.bundle_code)
            if task.status == "completed" and task.bundle_code and results
            else None
        )
    else:
        payload["results"] = []
        payload["bundle_projection"] = None
        payload["bundle_report_url"] = None

    return payload


def get_target_spec_payload(spec_key: str | None) -> dict[str, object] | None:
    spec = TARGET_SPECS.get(clean_text(spec_key))
    if spec is None:
        return None
    return target_generator_service._serialize_spec(spec)


def build_qr_target_payload(target: "QrTarget") -> dict[str, object]:
    exact_records = (
        ImageSegmentationRecord.query.filter_by(target_id=target.target_id, analysis_status="completed")
        .order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .all()
    )
    house_records = (
        ImageSegmentationRecord.query.filter_by(house_id=target.house_id, analysis_status="completed")
        .order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .all()
        if target.house_id
        else []
    )

    if exact_records:
        record_source = "same_target"
        display_records = exact_records[:8]
    elif house_records:
        record_source = "same_house"
        display_records = house_records[:8]
    else:
        record_source = "none"
        display_records = []

    latest_record = display_records[0] if display_records else None
    latest_record_payload = serialize_segmentation_record(latest_record) if latest_record else None
    house = db.session.get(HouseInfo, target.house_id) if target.house_id else None
    detail_url, _warning = build_target_public_url(target.target_id)
    target_status = "待检测"
    target_status_tone = "neutral"
    if exact_records:
        target_status = "已关联本靶标报告"
        target_status_tone = "safe"
    elif house_records:
        target_status = "已关联同房屋历史"
        target_status_tone = "warning"

    return {
        "target_id": target.target_id,
        "house_id": target.house_id,
        "house_number": target.house_number,
        "house_type": house.house_type if house else None,
        "crack_location": house.crack_location if house else None,
        "detection_type": house.detection_type if house else None,
        "inspection_region": target.inspection_region,
        "scene_type": target.scene_type,
        "inspection_at": serialize_datetime_value(target.inspection_at),
        "inspection_at_display": format_datetime_value(target.inspection_at, include_seconds=False)
        if target.inspection_at
        else "-",
        "report_reference": target.report_reference,
        "notes": target.notes,
        "spec": get_target_spec_payload(target.spec_key),
        "created_at": serialize_datetime_value(target.created_at),
        "created_at_display": format_datetime_value(target.created_at),
        "scan_url": detail_url,
        "detail_url": detail_url,
        "api_url": url_for("get_qr_target_data", target_id=target.target_id),
        "preview_url": build_file_url(Path(target.preview_path)) if target.preview_path else None,
        "pdf_url": build_file_url(Path(target.pdf_path)) if target.pdf_path else None,
        "manifest_url": build_file_url(Path(target.manifest_path)) if target.manifest_path else None,
        "target_status": target_status,
        "target_status_tone": target_status_tone,
        "record_source": record_source,
        "record_source_label": {
            "same_target": "当前靶标对应的检测记录",
            "same_house": "当前房屋的最新检测记录",
            "none": "暂无检测记录",
        }.get(record_source, "暂无检测记录"),
        "latest_report_url": url_for("download_detection_report", record_id=latest_record.id) if latest_record else None,
        "latest_report_view_url": (
            url_for("download_detection_report", record_id=latest_record.id, download=0) if latest_record else None
        ),
        "latest_bundle_report_url": (
            url_for("download_bundle_report", bundle_code=latest_record.upload_bundle_code)
            if latest_record and latest_record.upload_bundle_code
            else None
        ),
        "latest_detection_code": latest_record.detection_code if latest_record else None,
        "latest_risk_level": latest_record.risk_level if latest_record else None,
        "latest_risk_summary": latest_record.risk_summary if latest_record else None,
        "latest_recommendation": latest_record.recommendation if latest_record else None,
        "latest_generated_at_display": format_datetime_value(get_record_generated_at(latest_record)) if latest_record else "-",
        "latest_capture_scale_label": format_capture_scale_label(latest_record.capture_scale) if latest_record else "-",
        "latest_component_type": latest_record.component_type if latest_record else None,
        "latest_scenario_type": latest_record.scenario_type if latest_record else None,
        "latest_max_width_mm": latest_record.max_width_mm if latest_record else None,
        "latest_crack_length_mm": latest_record.crack_length_mm if latest_record else None,
        "latest_quality_status": latest_record.quality_status if latest_record else None,
        "latest_quality_score": latest_record.quality_score if latest_record else None,
        "latest_record": latest_record_payload,
        "records": [serialize_segmentation_record(record) for record in display_records],
        "exact_record_count": len(exact_records),
        "house_record_count": len(house_records),
    }


def enqueue_detection_task(task_id: int) -> None:
    task_executor.submit(run_detection_task, task_id)


def run_detection_task(task_id: int) -> None:
    results: list[dict[str, object]] = []
    status_code = 200

    with app.app_context():
        try:
            with app.test_request_context("/"):
                task = db.session.get(DetectionTask, task_id)
                if task is None or task.status not in {"queued", "running"}:
                    return

                upload_items = decode_json(task.upload_items_json, [])
                house = db.session.get(HouseInfo, task.house_id) if task.house_id else None

                task.status = "running"
                task.started_at = task.started_at or utc_now()
                task.status_message = "Analysis is running."
                db.session.commit()

                for index, item in enumerate(upload_items, start=1):
                    capture_scale = clean_text(item.get("capture_scale"), f"image-{index}")
                    task.status_message = f"Processing {capture_scale} ({index}/{len(upload_items)})."
                    db.session.commit()

                    result_payload, item_status = process_saved_image(
                        saved_path=Path(str(item["saved_path"])),
                        original_name=str(item["original_name"]),
                        house=house,
                        metadata=dict(item["metadata"]),
                        allow_low_quality=True,
                        run_segmentation=bool(task.run_segmentation),
                        bundle_code=task.bundle_code,
                        uploaded_at=task.created_at,
                    )
                    result_payload["capture_scale"] = capture_scale
                    results.append(result_payload)
                    if item_status != 200 and status_code == 200:
                        status_code = item_status

                    task.processed_items = index
                    task.results_json = encode_json(results)
                    task.result_status_code = status_code
                    db.session.commit()

                try:
                    bundle_projection = build_bundle_projection_from_results(results, task.bundle_code)
                except Exception as exc:  # pragma: no cover
                    bundle_projection = build_bundle_projection_unavailable(f"Projection unavailable: {exc}")

                task.processed_items = len(upload_items)
                task.results_json = encode_json(results)
                task.bundle_projection_json = encode_json(bundle_projection)
                task.result_status_code = status_code
                task.status = "completed"
                task.status_message = "Analysis completed." if status_code == 200 else "Analysis completed with errors."
                task.completed_at = utc_now()
                db.session.commit()
        except Exception:  # pragma: no cover
            task = db.session.get(DetectionTask, task_id)
            if task is not None:
                task.status = "failed"
                task.status_message = "Analysis failed."
                task.processed_items = len(results)
                task.results_json = encode_json(results)
                task.result_status_code = status_code if status_code != 200 else 500
                task.error_detail = traceback.format_exc()
                task.completed_at = utc_now()
                db.session.commit()
        finally:
            db.session.remove()


def recover_incomplete_detection_tasks() -> None:
    with app.app_context():
        stale_tasks = DetectionTask.query.filter(DetectionTask.status.in_(["queued", "running"])).all()
        if not stale_tasks:
            return

        recovered_at = utc_now()
        for task in stale_tasks:
            task.status = "failed"
            task.status_message = "Task interrupted before completion."
            task.result_status_code = task.result_status_code or 500
            task.error_detail = task.error_detail or "The service restarted while this task was pending."
            task.completed_at = recovered_at
        db.session.commit()


def save_uploaded_file(file_storage) -> tuple[str, Path]:
    original_name = secure_filename(file_storage.filename)
    unique_name = f"{uuid4().hex}_{original_name}"
    saved_path = UPLOAD_ROOT / unique_name
    file_storage.save(saved_path)
    return original_name, saved_path


def process_saved_image(
    saved_path: Path,
    original_name: str,
    house: "HouseInfo | None",
    metadata: dict[str, str],
    allow_low_quality: bool,
    run_segmentation: bool,
    bundle_code: str | None = None,
    uploaded_at: datetime | None = None,
) -> tuple[dict[str, object], int]:
    uploaded_at = uploaded_at or utc_now()
    quality_report = analyze_image_quality(saved_path)

    if quality_report["status"] == "reject" and not allow_low_quality:
        saved_path.unlink(missing_ok=True)
        return (
            {
                "message": "图像质量预检未通过，建议补拍后再分析。",
                "original_filename": original_name,
                "capture_scale": metadata["capture_scale"],
                "quality_report": serialize_quality_report(quality_report),
                "uploaded_at": serialize_datetime_value(uploaded_at),
                "uploaded_at_display": format_datetime_value(uploaded_at),
            },
            422,
        )

    response_payload: dict[str, object] = {
        "message": "图像上传成功。",
        "original_filename": original_name,
        "capture_scale": metadata["capture_scale"],
        "file_path": str(saved_path),
        "file_url": build_file_url(saved_path),
        "house_id": house.id if house else None,
        "upload_bundle_code": bundle_code,
        "metadata": metadata,
        "quality_report": serialize_quality_report(quality_report),
        "uploaded_at": serialize_datetime_value(uploaded_at),
        "uploaded_at_display": format_datetime_value(uploaded_at),
    }
    response_payload["detection_code"] = generate_detection_code(house.house_number if house else original_name)

    if not run_segmentation:
        return response_payload, 200

    try:
        segmentation_result = segmentation_service.segment_image(
            saved_path,
            app.config["SEGMENTATION_FOLDER"],
        )
    except (InferenceDependencyError, ModelConfigurationError, FileNotFoundError) as exc:
        response_payload["message"] = "图像已上传，但自动裂缝分析失败。"
        response_payload["segmentation_error"] = format_inference_error(exc)
        return response_payload, 503
    except Exception as exc:  # pragma: no cover
        response_payload["message"] = "图像已上传，但自动裂缝分析失败。"
        response_payload["segmentation_error"] = format_inference_error(exc)
        return response_payload, 500

    quantification_result = run_quantification_analysis(saved_path, segmentation_result)
    marker_detection = quantification_result["marker_detection"]
    quantification = quantification_result["quantification"]
    primary_target = extract_primary_target(marker_detection)
    if house is None:
        house = resolve_house_from_target(primary_target)
    detection_code = generate_detection_code(house.house_number if house else original_name)
    response_payload["house_id"] = house.id if house else None
    response_payload["detection_code"] = detection_code
    response_payload["target_id"] = clean_text(primary_target.get("target_id"))
    history_records = get_recent_house_records(house.id if house else None, limit=5)
    risk_assessment = assess_detection_risk(
        segmentation_result=segmentation_result,
        quantification=quantification,
        quality_report=quality_report,
        metadata=metadata,
        history_records=history_records,
    )
    response_payload["segmentation"] = {
        **segmentation_result,
        "mask_url": build_file_url(Path(segmentation_result["mask_path"])),
        "overlay_url": build_file_url(Path(segmentation_result["overlay_path"])),
    }
    response_payload["risk_assessment"] = risk_assessment
    response_payload["marker_detection"] = {
        **marker_detection,
        "annotated_image_url": build_file_url(Path(marker_detection["annotated_image_path"]))
        if marker_detection.get("annotated_image_path")
        else None,
    }
    response_payload["quantification"] = {
        **quantification,
        "quant_overlay_url": build_file_url(Path(quantification["quant_overlay_path"]))
        if quantification.get("quant_overlay_path")
        else None,
        "width_chart_url": build_file_url(Path(quantification["width_chart_path"]))
        if quantification.get("width_chart_path")
        else None,
    }
    response_payload["analysis_stages"] = build_analysis_stages(
        response_payload["marker_detection"],
        response_payload["quantification"],
    )

    generated_at = utc_now()
    record = ImageSegmentationRecord(
        house_id=house.id if house else None,
        upload_bundle_code=bundle_code,
        detection_code=detection_code,
        target_id=clean_text(primary_target.get("target_id")),
        analysis_status="completed",
        original_filename=original_name,
        source_image_path=str(saved_path),
        uploaded_at=uploaded_at,
        mask_path=segmentation_result["mask_path"],
        overlay_path=segmentation_result["overlay_path"],
        crack_pixel_count=segmentation_result["crack_pixel_count"],
        crack_area_ratio=segmentation_result["crack_area_ratio"],
        inference_device=segmentation_result["device"],
        patch_count=segmentation_result["patch_count"],
        capture_scale=metadata["capture_scale"],
        component_type=metadata["component_type"],
        scenario_type=metadata["scenario_type"],
        quality_score=quality_report["score"],
        quality_status=quality_report["status"],
        quality_warnings_json=encode_json(quality_report["warnings"]),
        risk_level=risk_assessment["risk_level"],
        risk_summary=risk_assessment["risk_summary"],
        recommendation=risk_assessment["recommendation"],
        marker_status=marker_detection.get("status"),
        marker_count=marker_detection.get("marker_count"),
        physical_scale_mm_per_pixel=marker_detection.get("physical_scale_mm_per_pixel"),
        target_overlay_path=marker_detection.get("annotated_image_path"),
        quantification_status=quantification.get("status"),
        quantification_message=quantification.get("message"),
        component_count=quantification.get("component_count"),
        sample_count=quantification.get("sample_count"),
        max_width_px=quantification.get("max_width_px"),
        avg_width_px=quantification.get("avg_width_px"),
        median_width_px=quantification.get("median_width_px"),
        crack_length_px=quantification.get("crack_length_px"),
        crack_angle_deg=quantification.get("crack_angle_deg"),
        max_width_mm=quantification.get("max_width_mm"),
        avg_width_mm=quantification.get("avg_width_mm"),
        median_width_mm=quantification.get("median_width_mm"),
        crack_length_mm=quantification.get("crack_length_mm"),
        quant_overlay_path=quantification.get("quant_overlay_path"),
        width_chart_path=quantification.get("width_chart_path"),
        created_at=generated_at,
    )
    db.session.add(record)
    bind_qr_target_house(clean_text(primary_target.get("target_id")), house)
    db.session.commit()
    response_payload["segmentation_record_id"] = record.id
    response_payload["report_url"] = url_for("download_detection_report", record_id=record.id)
    response_payload["generated_at"] = serialize_datetime_value(generated_at)
    response_payload["generated_at_display"] = format_datetime_value(generated_at)
    response_payload["created_at"] = serialize_datetime_value(generated_at)
    response_payload["created_at_display"] = format_datetime_value(generated_at)
    return response_payload, 200


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "未检测到图像文件。"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "请选择要上传的图像。"}), 400
    if not allowed_file(file.filename):
        return jsonify({"message": "仅支持 png / jpg / jpeg / gif 图像格式。"}), 400

    metadata = collect_upload_metadata(request.form)
    allow_low_quality = request.form.get("allow_low_quality", "false").lower() == "true"
    run_segmentation = request.form.get("run_segmentation", "true").lower() != "false"
    house, error = resolve_house_from_request(request.form.get("house_id"))
    if error:
        return jsonify(error[0]), error[1]

    ensure_runtime_directories()
    uploaded_at = utc_now()
    original_name, saved_path = save_uploaded_file(file)
    response_payload, status_code = process_saved_image(
        saved_path=saved_path,
        original_name=original_name,
        house=house,
        metadata=metadata,
        allow_low_quality=allow_low_quality,
        run_segmentation=run_segmentation,
        uploaded_at=uploaded_at,
    )
    return jsonify(response_payload), status_code


@app.route("/upload-batch", methods=["POST"])
def upload_batch():
    ensure_runtime_directories()
    uploaded_at = utc_now()
    allow_low_quality = request.form.get("allow_low_quality", "false").lower() == "true"
    run_segmentation = request.form.get("run_segmentation", "true").lower() != "false"
    base_metadata = collect_upload_metadata(request.form)
    house, error = resolve_house_from_request(request.form.get("house_id"))
    if error:
        return jsonify(error[0]), error[1]
    bundle_code = generate_bundle_code(house.house_number if house else "BUNDLE")

    saved_items: list[dict[str, object]] = []
    for field_name, capture_scale in BUNDLE_CAPTURE_SCALES:
        if field_name not in request.files:
            return jsonify({"message": f"缺少 {capture_scale} 图像，请按长景/中景/近景完整上传。"}), 400

        file = request.files[field_name]
        if file.filename == "":
            return jsonify({"message": f"{capture_scale} 图像为空，请重新选择文件。"}), 400
        if not allowed_file(file.filename):
            return jsonify({"message": f"{capture_scale} 图像格式无效，仅支持 png / jpg / jpeg / gif。"}), 400

        original_name, saved_path = save_uploaded_file(file)
        metadata = {
            **base_metadata,
            "capture_scale": capture_scale,
        }
        quality_report = analyze_image_quality(saved_path)
        saved_items.append(
            {
                "capture_scale": capture_scale,
                "original_name": original_name,
                "saved_path": saved_path,
                "metadata": metadata,
                "quality_report": serialize_quality_report(quality_report),
            }
        )

    rejected_items = [
        item for item in saved_items if item["quality_report"]["status"] == "reject"
    ]
    if rejected_items and not allow_low_quality:
        for item in saved_items:
            Path(item["saved_path"]).unlink(missing_ok=True)
        return (
            jsonify(
                {
                    "message": "三张图像中存在质量未通过的照片，请补拍后重新上传；也可以选择强制继续分析。",
                    "results": [
                        {
                            "capture_scale": item["capture_scale"],
                            "original_filename": item["original_name"],
                            "quality_report": item["quality_report"],
                        }
                        for item in saved_items
                    ],
                }
            ),
            422,
        )

    results: list[dict[str, object]] = []
    status_code = 200
    for item in saved_items:
        result_payload, item_status = process_saved_image(
            saved_path=Path(item["saved_path"]),
            original_name=str(item["original_name"]),
            house=house,
            metadata=dict(item["metadata"]),
            allow_low_quality=True,
            run_segmentation=run_segmentation,
            bundle_code=bundle_code,
            uploaded_at=uploaded_at,
        )
        result_payload["capture_scale"] = item["capture_scale"]
        results.append(result_payload)
        if item_status != 200 and status_code == 200:
            status_code = item_status

    try:
        bundle_projection = build_bundle_projection_from_results(results, bundle_code)
    except Exception as exc:  # pragma: no cover
        bundle_projection = build_bundle_projection_unavailable(f"长景回投生成失败：{exc}")
    generated_records = [
        record
        for record in (
            db.session.get(ImageSegmentationRecord, result_payload.get("segmentation_record_id"))
            for result_payload in results
            if result_payload.get("segmentation_record_id")
        )
        if record is not None
    ]
    generated_at = get_bundle_generated_at(generated_records)

    return (
        jsonify(
            {
                "message": "三张图像已完成批量处理。" if status_code == 200 else "三张图像已上传，但部分分析失败。",
                "house_id": house.id if house else None,
                "bundle_code": bundle_code,
                "uploaded_at": serialize_datetime_value(uploaded_at),
                "uploaded_at_display": format_datetime_value(uploaded_at),
                "generated_at": serialize_datetime_value(generated_at),
                "generated_at_display": format_datetime_value(generated_at),
                "bundle_report_url": url_for("download_bundle_report", bundle_code=bundle_code),
                "bundle_projection": bundle_projection,
                "results": results,
            }
        ),
        status_code,
    )


@app.route("/tasks/upload-batch", methods=["POST"])
def create_detection_task():
    ensure_runtime_directories()
    uploaded_at = utc_now()
    allow_low_quality = request.form.get("allow_low_quality", "false").lower() == "true"
    run_segmentation = request.form.get("run_segmentation", "true").lower() != "false"
    base_metadata = collect_upload_metadata(request.form)
    house, error = resolve_house_from_request(request.form.get("house_id"))
    if error:
        return jsonify(error[0]), error[1]

    bundle_code = generate_bundle_code(house.house_number if house else "BUNDLE")
    task_code = generate_task_code(house.house_number if house else "TASK")
    saved_items: list[dict[str, object]] = []

    for field_name, capture_scale in BUNDLE_CAPTURE_SCALES:
        if field_name not in request.files:
            return jsonify({"message": f"Missing {capture_scale} image."}), 400

        file = request.files[field_name]
        if file.filename == "":
            return jsonify({"message": f"{capture_scale} image is empty."}), 400
        if not allowed_file(file.filename):
            return jsonify({"message": f"{capture_scale} image format is invalid."}), 400

        original_name, saved_path = save_uploaded_file(file)
        metadata = {
            **base_metadata,
            "capture_scale": capture_scale,
        }
        quality_report = analyze_image_quality(saved_path)
        saved_items.append(
            {
                "capture_scale": capture_scale,
                "original_name": original_name,
                "saved_path": str(saved_path),
                "metadata": metadata,
                "quality_report": serialize_quality_report(quality_report),
            }
        )

    rejected_items = [item for item in saved_items if item["quality_report"]["status"] == "reject"]
    if rejected_items and not allow_low_quality:
        for item in saved_items:
            Path(str(item["saved_path"])).unlink(missing_ok=True)
        return (
            jsonify(
                {
                    "message": "At least one image failed precheck. Re-upload or force analysis.",
                    "results": [
                        {
                            "capture_scale": item["capture_scale"],
                            "original_filename": item["original_name"],
                            "quality_report": item["quality_report"],
                        }
                        for item in saved_items
                    ],
                }
            ),
            422,
        )

    task = DetectionTask(
        task_code=task_code,
        house_id=house.id if house else None,
        bundle_code=bundle_code,
        status="queued",
        status_message="Task queued for analysis.",
        total_items=len(saved_items),
        processed_items=0,
        allow_low_quality=allow_low_quality,
        run_segmentation=run_segmentation,
        request_metadata_json=encode_json(base_metadata),
        upload_items_json=encode_json(
            [
                {
                    "capture_scale": item["capture_scale"],
                    "original_name": item["original_name"],
                    "saved_path": item["saved_path"],
                    "metadata": item["metadata"],
                }
                for item in saved_items
            ]
        ),
        quality_results_json=encode_json(
            [
                {
                    "capture_scale": item["capture_scale"],
                    "original_filename": item["original_name"],
                    "quality_report": item["quality_report"],
                }
                for item in saved_items
            ]
        ),
        results_json=encode_json([]),
        created_at=uploaded_at,
    )
    db.session.add(task)
    db.session.commit()
    enqueue_detection_task(task.id)

    return jsonify(serialize_detection_task(task, include_results=False)), 202


@app.route("/tasks/<task_code>", methods=["GET"])
def get_detection_task(task_code: str):
    task = DetectionTask.query.filter_by(task_code=task_code).first()
    if task is None:
        return jsonify({"message": "Detection task not found"}), 404
    return jsonify(serialize_detection_task(task))


@app.route("/files/<path:filename>", methods=["GET"])
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_ROOT, filename)


@app.route("/health/inference", methods=["GET"])
def inference_health():
    return jsonify(
        {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "preferred_python": str(PREFERRED_PYTHON),
            "preferred_python_exists": PREFERRED_PYTHON.exists(),
            "dependencies": get_runtime_dependency_status(),
        }
    )


@app.route("/targets/specs", methods=["GET"])
def list_target_specs():
    return jsonify({"specs": target_generator_service.list_specs()})


@app.route("/api/targets/<target_id>", methods=["GET"])
def get_qr_target_data(target_id: str):
    qr_target = QrTarget.query.filter_by(target_id=target_id).first()
    if qr_target is None:
        return jsonify({"message": "QR target not found"}), 404
    return jsonify(build_qr_target_payload(qr_target))


@app.route("/targets/<target_id>", methods=["GET"])
def view_qr_target(target_id: str):
    qr_target = QrTarget.query.filter_by(target_id=target_id).first()
    if qr_target is None:
        return render_template("target_detail.html", payload=None, target_id=target_id), 404
    return render_template("target_detail.html", payload=build_qr_target_payload(qr_target), target_id=target_id)


@app.route("/targets/generate", methods=["POST"])
def generate_target():
    ensure_runtime_directories()
    data = request.get_json() or {}
    house_number = clean_text(data.get("house_number"))
    inspection_region = clean_text(data.get("inspection_region"))
    scene_type = clean_text(data.get("scene_type"))
    spec_key = clean_text(data.get("spec_key"), "square_80")
    report_reference = clean_text(data.get("report_reference"))
    notes = clean_text(data.get("notes"))
    inspection_at = parse_local_datetime_input(data.get("inspection_at"))

    if not house_number:
        return jsonify({"message": "house_number 不能为空"}), 400
    if not inspection_region:
        return jsonify({"message": "inspection_region 不能为空"}), 400
    if not scene_type:
        return jsonify({"message": "scene_type 不能为空"}), 400
    if data.get("inspection_at") and inspection_at is None:
        return jsonify({"message": "inspection_at 格式无效"}), 400

    house = resolve_qr_target_house(data.get("house_id"), house_number)
    target_id = f"TGT-{uuid4().hex[:10].upper()}"
    created_at = utc_now()
    created_at_text = serialize_datetime_value(created_at)
    scan_url, scan_url_warning = build_target_public_url(target_id)

    try:
        result = target_generator_service.generate_target(
            house_number=house_number,
            inspection_region=inspection_region,
            scene_type=scene_type,
            spec_key=spec_key,
            target_id=target_id,
            created_at=created_at_text,
            scan_url=scan_url,
            inspection_label=format_datetime_value(inspection_at, include_seconds=False) if inspection_at else None,
            report_reference=report_reference or None,
            notes=notes or None,
        )
    except ValueError as exc:
        return jsonify({"message": str(exc)}), 400
    except TargetGenerationError as exc:
        return jsonify({"message": f"靶标生成失败: {exc}"}), 500

    png_path = Path(result["paths"]["png"])
    pdf_path = Path(result["paths"]["pdf"])
    manifest_path = Path(result["paths"]["manifest"])
    qr_target = QrTarget(
        target_id=result["target_id"],
        house_id=house.id if house else None,
        house_number=house_number,
        inspection_region=inspection_region,
        scene_type=scene_type,
        inspection_at=inspection_at,
        report_reference=report_reference or None,
        notes=notes or None,
        spec_key=spec_key,
        qr_payload=result["qr_payload"],
        preview_path=str(png_path),
        pdf_path=str(pdf_path),
        manifest_path=str(manifest_path),
        created_at=created_at,
    )
    db.session.add(qr_target)
    db.session.commit()

    profile_payload = build_qr_target_payload(qr_target)
    return jsonify(
        {
            "message": "二维码靶标已生成",
            "target_id": result["target_id"],
            "payload": result["payload"],
            "metadata": result.get("metadata") or {},
            "qr_payload": result["qr_payload"],
            "spec": result["spec"],
            "house_id": house.id if house else None,
            "inspection_at": serialize_datetime_value(inspection_at),
            "inspection_at_display": format_datetime_value(inspection_at, include_seconds=False) if inspection_at else "-",
            "report_reference": report_reference or None,
            "notes": notes or None,
            "scan_url": scan_url,
            "scan_url_warning": scan_url_warning,
            "detail_url": profile_payload["detail_url"],
            "api_url": profile_payload["api_url"],
            "preview_url": profile_payload["preview_url"],
            "profile": profile_payload,
            "files": {
                "png_url": profile_payload["preview_url"],
                "pdf_url": profile_payload["pdf_url"],
                "manifest_url": profile_payload["manifest_url"],
            },
        }
    ), 201


@app.route("/targets/precheck", methods=["POST"])
def precheck_target():
    if "file" not in request.files:
        return jsonify({"message": "未检测到中景图像文件"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"message": "请选择要预检的中景图像"}), 400

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"message": "中景图像内容为空"}), 400

    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"message": "无法读取中景图像内容"}), 400

    marker_detection = quantification_service.inspect_target_image(image)
    passed = marker_detection.get("status") == "anchor_rectified"
    if passed:
        message = "二维码和四角锚点识别成功，可进入上传分析。"
        status_code = 200
    elif marker_detection.get("status") == "qr_detected":
        message = "二维码可解码，但四角锚点未稳定识别，请调整拍摄角度、距离或清晰度后重拍中景图。"
        status_code = 422
    else:
        message = "未识别到可用二维码靶标，请确保裂缝与二维码靶标同框、同平面且清晰可见。"
        status_code = 422

    return jsonify(
        {
            "message": message,
            "passed": passed,
            "marker_detection": marker_detection,
        }
    ), status_code


@app.route("/submit_house_info", methods=["POST"])
def submit_house_info():
    data = request.get_json() or {}
    house_number = clean_text(data.get("house_number"))
    house_type = clean_text(data.get("house_type"))
    crack_location = clean_text(data.get("crack_location"))
    detection_type = clean_text(data.get("detection_type"))

    if not house_number:
        return jsonify({"message": "请先填写房屋编号。"}), 400

    existing_house = HouseInfo.query.filter_by(house_number=house_number).first()
    if existing_house:
        return jsonify({"message": "House with this number already exists", "house_id": existing_house.id}), 409

    new_house = HouseInfo(
        house_number=house_number,
        house_type=house_type,
        crack_location=crack_location,
        detection_type=detection_type,
    )
    db.session.add(new_house)
    db.session.commit()
    return jsonify({"message": "House info submitted successfully", "house_id": new_house.id}), 201


@app.route("/house/<int:house_id>/detections", methods=["GET"])
def list_house_detections(house_id: int):
    house = db.session.get(HouseInfo, house_id)
    if house is None:
        return jsonify({"message": "House not found"}), 404

    records = (
        ImageSegmentationRecord.query.filter_by(house_id=house_id)
        .order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .limit(12)
        .all()
    )
    previous_lookup = build_previous_record_lookup(records)

    return jsonify(
        {
            "house_id": house.id,
            "house_number": house.house_number,
            "house_type": house.house_type,
            "crack_location": house.crack_location,
            "detection_type": house.detection_type,
            "trend_summary": build_trend_summary(records),
            "records": [
                serialize_segmentation_record(record, previous_record=previous_lookup.get(record.id)) for record in records
            ],
        }
    )


@app.route("/survey/summary", methods=["GET"])
def survey_summary():
    records = (
        ImageSegmentationRecord.query.order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .all()
    )
    summary = build_survey_summary(records)
    return jsonify(summary)


@app.route("/survey/summary.csv", methods=["GET"])
def download_survey_summary_csv():
    records = (
        ImageSegmentationRecord.query.order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .all()
    )
    summary = build_survey_summary(records)
    csv_buffer = BytesIO()
    text_buffer = StringIO()
    writer = csv.writer(text_buffer)
    writer.writerow(["section", "name", "value_1", "value_2", "value_3"])
    writer.writerow(["overview", "total_records", summary["total_records"], "", ""])
    writer.writerow(["overview", "completed_records", summary["completed_records"], "", ""])
    writer.writerow(["overview", "house_count", summary["house_count"], "", ""])
    for level, count in summary["risk_distribution"].items():
        writer.writerow(["risk_distribution", level, count, "", ""])
    for item in summary["high_risk_houses"]:
        writer.writerow(
            [
                "high_risk_house",
                item["house_number"],
                item["risk_level"],
                item["max_width_mm"] or "",
                item["crack_length_mm"] or "",
            ]
        )
    csv_buffer.write(text_buffer.getvalue().encode("utf-8-sig"))
    csv_buffer.seek(0)
    return send_file(
        csv_buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="survey-summary.csv",
    )


@app.route("/detections/<int:record_id>/report", methods=["GET"])
def download_detection_report(record_id: int):
    record = db.session.get(ImageSegmentationRecord, record_id)
    if record is None:
        return jsonify({"message": "Detection record not found"}), 404

    download = request.args.get("download", "1").strip().lower() not in {"0", "false", "no", "inline"}

    house = db.session.get(HouseInfo, record.house_id) if record.house_id else None
    records = (
        ImageSegmentationRecord.query.filter_by(house_id=record.house_id)
        .order_by(ImageSegmentationRecord.created_at.desc(), ImageSegmentationRecord.id.desc())
        .limit(12)
        .all()
        if record.house_id
        else [record]
    )
    trend_summary = build_trend_summary(records)
    pdf_buffer = build_report_pdf(record, house, trend_summary)
    filename = f"{record.detection_code or f'detection-{record.id}'}-report.pdf"
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=download,
        download_name=filename,
    )


@app.route("/bundles/<bundle_code>/report", methods=["GET"])
def download_bundle_report(bundle_code: str):
    records = get_bundle_records(bundle_code)
    if not records:
        return jsonify({"message": "Batch report not found"}), 404

    download = request.args.get("download", "1").strip().lower() not in {"0", "false", "no", "inline"}

    first_record = records[0]
    house = db.session.get(HouseInfo, first_record.house_id) if first_record.house_id else None
    pdf_buffer = build_bundle_report_pdf(records, house, bundle_code)
    filename = f"{bundle_code}-summary-report.pdf"
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=download,
        download_name=filename,
    )


@app.route("/detections/<int:record_id>", methods=["DELETE"])
def delete_detection_record(record_id: int):
    record = db.session.get(ImageSegmentationRecord, record_id)
    if record is None:
        return jsonify({"message": "Detection record not found"}), 404

    source_path = Path(record.source_image_path)
    mask_path = Path(record.mask_path) if record.mask_path else None
    overlay_path = Path(record.overlay_path) if record.overlay_path else None
    target_overlay_path = Path(record.target_overlay_path) if record.target_overlay_path else None
    quant_overlay_path = Path(record.quant_overlay_path) if record.quant_overlay_path else None
    width_chart_path = Path(record.width_chart_path) if record.width_chart_path else None
    result_dir = mask_path.parent if mask_path else None
    house_id = record.house_id

    db.session.delete(record)
    db.session.commit()

    for candidate in [source_path, mask_path, overlay_path, target_overlay_path, quant_overlay_path, width_chart_path]:
        if candidate and candidate.exists():
            candidate.unlink()

    if result_dir and result_dir.exists() and not any(result_dir.iterdir()):
        result_dir.rmdir()

    return jsonify({"message": "Detection record deleted", "record_id": record_id, "house_id": house_id})


@app.route("/detections/<int:record_id>/rerun", methods=["POST"])
def rerun_detection_record(record_id: int):
    record = db.session.get(ImageSegmentationRecord, record_id)
    if record is None:
        return jsonify({"message": "Detection record not found"}), 404

    source_path = Path(record.source_image_path)
    if not source_path.exists():
        return jsonify({"message": "Source image not found for this detection record"}), 404

    quality_report = analyze_image_quality(source_path)
    metadata = {
        "capture_scale": record.capture_scale or "未标注",
        "component_type": record.component_type or "未标注",
        "scenario_type": record.scenario_type or "常规巡检",
    }

    try:
        segmentation_result = segmentation_service.segment_image(
            source_path,
            app.config["SEGMENTATION_FOLDER"],
        )
    except (InferenceDependencyError, ModelConfigurationError, FileNotFoundError) as exc:
        return jsonify({"message": "Rerun failed", "segmentation_error": format_inference_error(exc)}), 503
    except Exception as exc:  # pragma: no cover
        return jsonify({"message": "Rerun failed", "segmentation_error": format_inference_error(exc)}), 500

    quantification_result = run_quantification_analysis(source_path, segmentation_result)
    marker_detection = quantification_result["marker_detection"]
    quantification = quantification_result["quantification"]
    history_records = get_recent_house_records(record.house_id, limit=5)
    history_records = [item for item in history_records if item.id != record.id]
    risk_assessment = assess_detection_risk(
        segmentation_result=segmentation_result,
        quantification=quantification,
        quality_report=quality_report,
        metadata=metadata,
        history_records=history_records,
    )
    record.analysis_status = "completed"
    record.mask_path = segmentation_result["mask_path"]
    record.overlay_path = segmentation_result["overlay_path"]
    record.crack_pixel_count = segmentation_result["crack_pixel_count"]
    record.crack_area_ratio = segmentation_result["crack_area_ratio"]
    record.inference_device = segmentation_result["device"]
    record.patch_count = segmentation_result["patch_count"]
    record.quality_score = quality_report["score"]
    record.quality_status = quality_report["status"]
    record.quality_warnings_json = encode_json(quality_report["warnings"])
    record.risk_level = risk_assessment["risk_level"]
    record.risk_summary = risk_assessment["risk_summary"]
    record.recommendation = risk_assessment["recommendation"]
    record.marker_status = marker_detection.get("status")
    record.marker_count = marker_detection.get("marker_count")
    record.physical_scale_mm_per_pixel = marker_detection.get("physical_scale_mm_per_pixel")
    record.target_overlay_path = marker_detection.get("annotated_image_path")
    record.quantification_status = quantification.get("status")
    record.quantification_message = quantification.get("message")
    record.component_count = quantification.get("component_count")
    record.sample_count = quantification.get("sample_count")
    record.max_width_px = quantification.get("max_width_px")
    record.avg_width_px = quantification.get("avg_width_px")
    record.median_width_px = quantification.get("median_width_px")
    record.crack_length_px = quantification.get("crack_length_px")
    record.crack_angle_deg = quantification.get("crack_angle_deg")
    record.max_width_mm = quantification.get("max_width_mm")
    record.avg_width_mm = quantification.get("avg_width_mm")
    record.median_width_mm = quantification.get("median_width_mm")
    record.crack_length_mm = quantification.get("crack_length_mm")
    record.quant_overlay_path = quantification.get("quant_overlay_path")
    record.width_chart_path = quantification.get("width_chart_path")
    db.session.commit()

    return jsonify(
        {
            "message": "Detection rerun successfully",
            "record": serialize_segmentation_record(
                record,
                previous_record=history_records[0] if history_records else None,
                risk_assessment=risk_assessment,
            ),
            "quality_report": serialize_quality_report(quality_report),
            "risk_assessment": risk_assessment,
            "marker_detection": {
                **marker_detection,
                "annotated_image_url": build_file_url(Path(marker_detection["annotated_image_path"]))
                if marker_detection.get("annotated_image_path")
                else None,
            },
            "quantification": {
                **quantification,
                "quant_overlay_url": build_file_url(Path(quantification["quant_overlay_path"]))
                if quantification.get("quant_overlay_path")
                else None,
                "width_chart_url": build_file_url(Path(quantification["width_chart_path"]))
                if quantification.get("width_chart_path")
                else None,
            },
        }
    )


def render_frontend_entry():
    return render_template("index.html")


if FRONTEND_ENTRY_PATH != "/":
    app.add_url_rule(FRONTEND_ENTRY_PATH, "frontend_entry", render_frontend_entry)
    app.add_url_rule(f"{FRONTEND_ENTRY_PATH}/", "frontend_entry_slash", render_frontend_entry)


@app.route("/")
def index():
    if FRONTEND_ENTRY_PATH == "/":
        return render_frontend_entry()
    return redirect(f"{FRONTEND_ENTRY_PATH}/", code=302)


ensure_database_tables()
ensure_sqlite_columns()
recover_incomplete_detection_tasks()


if __name__ == "__main__":
    ensure_runtime_directories()
    ensure_database_tables()
    ensure_sqlite_columns()
    recover_incomplete_detection_tasks()
    host = os.getenv("HOUSE_UPLOAD_HOST", "0.0.0.0")
    port = int(os.getenv("HOUSE_UPLOAD_PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)

from __future__ import annotations

try:
    from .db import db
except ImportError:
    from data.db import db


class HouseInfo(db.Model):
    __tablename__ = "house_info"

    id = db.Column(db.Integer, primary_key=True)
    house_number = db.Column(db.String(50), unique=True, nullable=False)
    house_type = db.Column(db.String(50))
    crack_location = db.Column(db.String(100))
    detection_type = db.Column(db.String(50))
    upload_time = db.Column(db.DateTime, server_default=db.func.now())
    crack_results = db.relationship("CrackDetectionResults", backref="house_info", lazy=True)
    image_detections = db.relationship("ImageSegmentationRecord", backref="house_info", lazy=True)
    detection_tasks = db.relationship("DetectionTask", backref="house_info", lazy=True)
    qr_targets = db.relationship("QrTarget", backref="house_info", lazy=True)


class CrackDetectionResults(db.Model):
    __tablename__ = "crack_detection_results"

    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house_info.id"), nullable=False)
    crack_width = db.Column(db.Float)
    crack_length = db.Column(db.Float)
    crack_angle = db.Column(db.Float)
    damage_level = db.Column(db.String(50))


class DetectionTask(db.Model):
    __tablename__ = "detection_task"

    id = db.Column(db.Integer, primary_key=True)
    task_code = db.Column(db.String(40), unique=True, nullable=False)
    house_id = db.Column(db.Integer, db.ForeignKey("house_info.id"), nullable=True)
    bundle_code = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default="queued", nullable=False)
    status_message = db.Column(db.String(500))
    total_items = db.Column(db.Integer, default=0)
    processed_items = db.Column(db.Integer, default=0)
    allow_low_quality = db.Column(db.Boolean, default=False)
    run_segmentation = db.Column(db.Boolean, default=True)
    request_metadata_json = db.Column(db.Text)
    upload_items_json = db.Column(db.Text)
    quality_results_json = db.Column(db.Text)
    results_json = db.Column(db.Text)
    bundle_projection_json = db.Column(db.Text)
    result_status_code = db.Column(db.Integer)
    error_detail = db.Column(db.Text)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class QrTarget(db.Model):
    __tablename__ = "qr_target"

    id = db.Column(db.Integer, primary_key=True)
    target_id = db.Column(db.String(40), unique=True, nullable=False)
    house_id = db.Column(db.Integer, db.ForeignKey("house_info.id"), nullable=True)
    house_number = db.Column(db.String(50), nullable=False)
    inspection_region = db.Column(db.String(120), nullable=False)
    scene_type = db.Column(db.String(50), nullable=False)
    inspection_at = db.Column(db.DateTime)
    report_reference = db.Column(db.String(120))
    notes = db.Column(db.Text)
    spec_key = db.Column(db.String(40))
    qr_payload = db.Column(db.Text)
    preview_path = db.Column(db.String(500))
    pdf_path = db.Column(db.String(500))
    manifest_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class ImageSegmentationRecord(db.Model):
    __tablename__ = "image_segmentation_record"

    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house_info.id"), nullable=True)
    upload_bundle_code = db.Column(db.String(50))
    detection_code = db.Column(db.String(40), unique=True, nullable=False)
    target_id = db.Column(db.String(40))
    analysis_status = db.Column(db.String(20), default="completed")
    original_filename = db.Column(db.String(255), nullable=False)
    source_image_path = db.Column(db.String(500), nullable=False)
    uploaded_at = db.Column(db.DateTime)
    mask_path = db.Column(db.String(500))
    overlay_path = db.Column(db.String(500))
    crack_pixel_count = db.Column(db.Integer)
    crack_area_ratio = db.Column(db.Float)
    inference_device = db.Column(db.String(50))
    patch_count = db.Column(db.Integer)
    capture_scale = db.Column(db.String(20))
    component_type = db.Column(db.String(50))
    scenario_type = db.Column(db.String(50))
    quality_score = db.Column(db.Float)
    quality_status = db.Column(db.String(20))
    quality_warnings_json = db.Column(db.Text)
    risk_level = db.Column(db.String(20))
    risk_summary = db.Column(db.String(500))
    recommendation = db.Column(db.String(500))
    marker_status = db.Column(db.String(20))
    marker_count = db.Column(db.Integer)
    physical_scale_mm_per_pixel = db.Column(db.Float)
    target_overlay_path = db.Column(db.String(500))
    quantification_status = db.Column(db.String(20))
    quantification_message = db.Column(db.String(500))
    component_count = db.Column(db.Integer)
    sample_count = db.Column(db.Integer)
    max_width_px = db.Column(db.Float)
    avg_width_px = db.Column(db.Float)
    median_width_px = db.Column(db.Float)
    crack_length_px = db.Column(db.Float)
    crack_angle_deg = db.Column(db.Float)
    max_width_mm = db.Column(db.Float)
    avg_width_mm = db.Column(db.Float)
    median_width_mm = db.Column(db.Float)
    crack_length_mm = db.Column(db.Float)
    quant_overlay_path = db.Column(db.String(500))
    width_chart_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, server_default=db.func.now())

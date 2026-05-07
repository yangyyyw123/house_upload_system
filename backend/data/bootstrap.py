from __future__ import annotations

try:
    from .db import db
except ImportError:
    from data.db import db


def ensure_database_tables(app) -> None:
    with app.app_context():
        db.create_all()


def ensure_sqlite_columns(app) -> None:
    migrations = {
        "image_segmentation_record": {
            "detection_code": "TEXT",
            "target_id": "TEXT",
            "upload_bundle_code": "TEXT",
            "uploaded_at": "DATETIME",
            "analysis_status": "TEXT DEFAULT 'completed'",
            "capture_scale": "TEXT",
            "component_type": "TEXT",
            "scenario_type": "TEXT",
            "quality_score": "FLOAT",
            "quality_status": "TEXT",
            "quality_warnings_json": "TEXT",
            "risk_level": "TEXT",
            "risk_summary": "TEXT",
            "recommendation": "TEXT",
            "marker_status": "TEXT",
            "marker_count": "INTEGER",
            "physical_scale_mm_per_pixel": "FLOAT",
            "target_overlay_path": "TEXT",
            "quantification_status": "TEXT",
            "quantification_message": "TEXT",
            "component_count": "INTEGER",
            "sample_count": "INTEGER",
            "max_width_px": "FLOAT",
            "avg_width_px": "FLOAT",
            "median_width_px": "FLOAT",
            "crack_length_px": "FLOAT",
            "crack_angle_deg": "FLOAT",
            "max_width_mm": "FLOAT",
            "avg_width_mm": "FLOAT",
            "median_width_mm": "FLOAT",
            "crack_length_mm": "FLOAT",
            "quant_overlay_path": "TEXT",
            "width_chart_path": "TEXT",
        },
        "qr_target": {
            "house_id": "INTEGER",
            "house_number": "TEXT",
            "inspection_region": "TEXT",
            "scene_type": "TEXT",
            "inspection_at": "DATETIME",
            "report_reference": "TEXT",
            "notes": "TEXT",
            "spec_key": "TEXT",
            "qr_payload": "TEXT",
            "preview_path": "TEXT",
            "pdf_path": "TEXT",
            "manifest_path": "TEXT",
            "created_at": "DATETIME",
        },
    }

    with app.app_context():
        with db.engine.begin() as connection:
            for table_name, columns in migrations.items():
                existing_columns = {
                    row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
                }
                for column_name, column_definition in columns.items():
                    if column_name in existing_columns:
                        continue
                    try:
                        connection.exec_driver_sql(
                            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
                        )
                    except Exception as exc:
                        if "duplicate column name" not in str(exc).lower():
                            raise

            connection.exec_driver_sql(
                """
                UPDATE image_segmentation_record
                SET detection_code = COALESCE(detection_code, 'LEGACY-' || id)
                WHERE detection_code IS NULL OR detection_code = ''
                """
            )
            connection.exec_driver_sql(
                """
                UPDATE image_segmentation_record
                SET analysis_status = COALESCE(analysis_status, 'completed')
                WHERE analysis_status IS NULL OR analysis_status = ''
                """
            )
            connection.exec_driver_sql(
                """
                UPDATE image_segmentation_record
                SET uploaded_at = (
                    SELECT detection_task.created_at
                    FROM detection_task
                    WHERE detection_task.bundle_code = image_segmentation_record.upload_bundle_code
                    ORDER BY detection_task.created_at ASC, detection_task.id ASC
                    LIMIT 1
                )
                WHERE uploaded_at IS NULL
                  AND upload_bundle_code IS NOT NULL
                  AND upload_bundle_code <> ''
                """
            )
            connection.exec_driver_sql(
                """
                UPDATE image_segmentation_record
                SET uploaded_at = COALESCE(uploaded_at, created_at)
                WHERE uploaded_at IS NULL
                """
            )

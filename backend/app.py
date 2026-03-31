import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
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


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_ROOT = BASE_DIR / "uploads"
SEGMENTATION_ROOT = UPLOAD_ROOT / "segmentation"
PREFERRED_PYTHON = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"

app = Flask(__name__)
app.static_folder = str((BASE_DIR.parent / "frontend").resolve())
app.static_url_path = "/static"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///house.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = str(UPLOAD_ROOT)
app.config["SEGMENTATION_FOLDER"] = str(SEGMENTATION_ROOT)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

db = SQLAlchemy(app)
segmentation_service = CrackSegmentationService()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_runtime_directories() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_ROOT.mkdir(parents=True, exist_ok=True)


def build_file_url(path: Path) -> str:
    relative_path = path.resolve().relative_to(UPLOAD_ROOT.resolve())
    return url_for("uploaded_file", filename=relative_path.as_posix())


def get_runtime_dependency_status() -> dict[str, bool]:
    modules = {
        "flask": "flask",
        "flask_sqlalchemy": "flask_sqlalchemy",
        "numpy": "numpy",
        "opencv_python_headless": "cv2",
        "pillow": "PIL",
        "torch": "torch",
        "torchvision": "torchvision",
        "segmentation_models_pytorch": "segmentation_models_pytorch",
    }
    return {name: importlib.util.find_spec(module) is not None for name, module in modules.items()}


def format_inference_error(exc: Exception) -> str:
    base_message = str(exc)
    dependency_status = get_runtime_dependency_status()
    missing_dependencies = [name for name, installed in dependency_status.items() if not installed]

    details = [
        base_message,
        f"Current Python: {sys.executable}",
        f"Python version: {sys.version.split()[0]}",
    ]

    if missing_dependencies:
        details.append("Missing modules in current environment: " + ", ".join(missing_dependencies))

    if PREFERRED_PYTHON.exists() and Path(sys.executable).resolve() != PREFERRED_PYTHON.resolve():
        details.append(f"Start the backend with: {PREFERRED_PYTHON} backend\\app.py")

    return " | ".join(details)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class HouseInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    house_number = db.Column(db.String(50), unique=True, nullable=False)
    house_type = db.Column(db.String(50))
    crack_location = db.Column(db.String(100))
    detection_type = db.Column(db.String(50))
    upload_time = db.Column(db.DateTime, server_default=db.func.now())
    crack_results = db.relationship("CrackDetectionResults", backref="house_info", lazy=True)
    image_detections = db.relationship("ImageSegmentationRecord", backref="house_info", lazy=True)


class CrackDetectionResults(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house_info.id"), nullable=False)
    crack_width = db.Column(db.Float)
    crack_length = db.Column(db.Float)
    crack_angle = db.Column(db.Float)
    damage_level = db.Column(db.String(50))


class ImageSegmentationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    house_id = db.Column(db.Integer, db.ForeignKey("house_info.id"), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    source_image_path = db.Column(db.String(500), nullable=False)
    mask_path = db.Column(db.String(500))
    overlay_path = db.Column(db.String(500))
    crack_pixel_count = db.Column(db.Integer)
    crack_area_ratio = db.Column(db.Float)
    inference_device = db.Column(db.String(50))
    patch_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


def ensure_database_tables() -> None:
    with app.app_context():
        db.create_all()


def serialize_segmentation_record(record: ImageSegmentationRecord) -> dict:
    return {
        "id": record.id,
        "house_id": record.house_id,
        "original_filename": record.original_filename,
        "source_image_url": build_file_url(Path(record.source_image_path)),
        "mask_url": build_file_url(Path(record.mask_path)) if record.mask_path else None,
        "overlay_url": build_file_url(Path(record.overlay_path)) if record.overlay_path else None,
        "crack_pixel_count": record.crack_pixel_count,
        "crack_area_ratio": record.crack_area_ratio,
        "inference_device": record.inference_device,
        "patch_count": record.patch_count,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type"}), 400

    ensure_runtime_directories()
    original_name = secure_filename(file.filename)
    unique_name = f"{uuid4().hex}_{original_name}"
    saved_path = UPLOAD_ROOT / unique_name
    file.save(saved_path)

    run_segmentation = request.form.get("run_segmentation", "true").lower() != "false"
    house_id_value = request.form.get("house_id")
    house = None
    if house_id_value:
        try:
            house_id = int(house_id_value)
        except ValueError:
            return jsonify({"message": "Invalid house_id"}), 400

        house = HouseInfo.query.get(house_id)
        if house is None:
            return jsonify({"message": "House info not found for the provided house_id"}), 404

    response_payload = {
        "message": "File uploaded successfully",
        "file_path": str(saved_path),
        "file_url": build_file_url(saved_path),
        "house_id": house.id if house else None,
    }

    if not run_segmentation:
        return jsonify(response_payload), 200

    try:
        segmentation_result = segmentation_service.segment_image(
            saved_path,
            app.config["SEGMENTATION_FOLDER"],
        )
    except (InferenceDependencyError, ModelConfigurationError, FileNotFoundError) as exc:
        response_payload["message"] = "File uploaded, but crack segmentation failed"
        response_payload["segmentation_error"] = format_inference_error(exc)
        return jsonify(response_payload), 503
    except Exception as exc:  # pragma: no cover
        response_payload["message"] = "File uploaded, but crack segmentation failed"
        response_payload["segmentation_error"] = format_inference_error(exc)
        return jsonify(response_payload), 500

    response_payload["segmentation"] = {
        **segmentation_result,
        "mask_url": build_file_url(Path(segmentation_result["mask_path"])),
        "overlay_url": build_file_url(Path(segmentation_result["overlay_path"])),
    }

    record = ImageSegmentationRecord(
        house_id=house.id if house else None,
        original_filename=original_name,
        source_image_path=str(saved_path),
        mask_path=segmentation_result["mask_path"],
        overlay_path=segmentation_result["overlay_path"],
        crack_pixel_count=segmentation_result["crack_pixel_count"],
        crack_area_ratio=segmentation_result["crack_area_ratio"],
        inference_device=segmentation_result["device"],
        patch_count=segmentation_result["patch_count"],
    )
    db.session.add(record)
    db.session.commit()
    response_payload["segmentation_record_id"] = record.id
    return jsonify(response_payload), 200


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


@app.route("/submit_house_info", methods=["POST"])
def submit_house_info():
    data = request.get_json() or {}
    house_number = data.get("house_number")
    house_type = data.get("house_type")
    crack_location = data.get("crack_location")
    detection_type = data.get("detection_type")

    existing_house = HouseInfo.query.filter_by(house_number=house_number).first()
    if existing_house:
        return jsonify(
            {
                "message": "House with this number already exists",
                "house_id": existing_house.id,
            }
        ), 409

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
    house = HouseInfo.query.get(house_id)
    if house is None:
        return jsonify({"message": "House not found"}), 404

    records = (
        ImageSegmentationRecord.query.filter_by(house_id=house_id)
        .order_by(ImageSegmentationRecord.created_at.desc())
        .limit(10)
        .all()
    )

    return jsonify(
        {
            "house_id": house.id,
            "house_number": house.house_number,
            "records": [serialize_segmentation_record(record) for record in records],
        }
    )


@app.route("/detections/<int:record_id>", methods=["DELETE"])
def delete_detection_record(record_id: int):
    record = ImageSegmentationRecord.query.get(record_id)
    if record is None:
        return jsonify({"message": "Detection record not found"}), 404

    source_path = Path(record.source_image_path)
    mask_path = Path(record.mask_path) if record.mask_path else None
    overlay_path = Path(record.overlay_path) if record.overlay_path else None
    result_dir = mask_path.parent if mask_path else None
    house_id = record.house_id

    db.session.delete(record)
    db.session.commit()

    for candidate in [mask_path, overlay_path]:
        if candidate and candidate.exists():
            candidate.unlink()

    if result_dir and result_dir.exists() and not any(result_dir.iterdir()):
        result_dir.rmdir()

    return jsonify({"message": "Detection record deleted", "record_id": record_id, "house_id": house_id})


@app.route("/detections/<int:record_id>/rerun", methods=["POST"])
def rerun_detection_record(record_id: int):
    record = ImageSegmentationRecord.query.get(record_id)
    if record is None:
        return jsonify({"message": "Detection record not found"}), 404

    source_path = Path(record.source_image_path)
    if not source_path.exists():
        return jsonify({"message": "Source image not found for this detection record"}), 404

    try:
        segmentation_result = segmentation_service.segment_image(
            source_path,
            app.config["SEGMENTATION_FOLDER"],
        )
    except (InferenceDependencyError, ModelConfigurationError, FileNotFoundError) as exc:
        return jsonify({"message": "Rerun failed", "segmentation_error": format_inference_error(exc)}), 503
    except Exception as exc:  # pragma: no cover
        return jsonify({"message": "Rerun failed", "segmentation_error": format_inference_error(exc)}), 500

    record.mask_path = segmentation_result["mask_path"]
    record.overlay_path = segmentation_result["overlay_path"]
    record.crack_pixel_count = segmentation_result["crack_pixel_count"]
    record.crack_area_ratio = segmentation_result["crack_area_ratio"]
    record.inference_device = segmentation_result["device"]
    record.patch_count = segmentation_result["patch_count"]
    db.session.commit()

    return jsonify(
        {
            "message": "Detection rerun successfully",
            "record": serialize_segmentation_record(record),
        }
    )


@app.route("/")
def index():
    return render_template("index.html")


ensure_database_tables()


if __name__ == "__main__":
    ensure_runtime_directories()
    ensure_database_tables()
    app.run(debug=False, use_reloader=False)

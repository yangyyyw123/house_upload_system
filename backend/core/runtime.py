from __future__ import annotations

import importlib
import importlib.util
import os
import socket
import sys
from pathlib import Path

from flask import request, url_for

from .common import clean_text
from .settings import (
    ALLOWED_EXTENSIONS,
    CAPTURE_SCALE_LABELS,
    MARKER_MODE_LABELS,
    PREFERRED_PYTHON,
    TARGET_ROOT,
    UPLOAD_ROOT,
    SEGMENTATION_ROOT,
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_runtime_directories() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_ROOT.mkdir(parents=True, exist_ok=True)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)


def build_file_url(path: Path) -> str | None:
    try:
        relative_path = path.resolve().relative_to(UPLOAD_ROOT.resolve())
    except ValueError:
        return None
    return url_for("uploaded_file", filename=relative_path.as_posix())


def detect_local_ipv4() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip_address = clean_text(sock.getsockname()[0])
            if ip_address and not ip_address.startswith(("127.", "169.254.")):
                return ip_address
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip_address = clean_text(info[4][0])
            if ip_address and not ip_address.startswith(("127.", "169.254.")):
                return ip_address
    except OSError:
        pass
    return None


def format_capture_scale_label(value: str | None) -> str:
    return CAPTURE_SCALE_LABELS.get(clean_text(value), clean_text(value, "-"))


def format_marker_mode_label(value: str | None) -> str:
    return MARKER_MODE_LABELS.get(clean_text(value), clean_text(value, "-"))


def get_target_public_base_url() -> tuple[str, str | None]:
    configured_base = clean_text(os.environ.get("TARGET_PUBLIC_BASE_URL") or os.environ.get("PUBLIC_BASE_URL"))
    if configured_base:
        return configured_base.rstrip("/"), None

    base_url = request.host_url.rstrip("/")
    host = clean_text(request.host.split(":", 1)[0]).lower()
    warning = None
    if host in {"127.0.0.1", "localhost", "::1"}:
        lan_ip = detect_local_ipv4()
        if lan_ip:
            port = request.host.split(":", 1)[1] if ":" in request.host else ""
            port_suffix = f":{port}" if port else ""
            warning = f"当前通过 localhost 访问，二维码已自动改写为局域网地址 {lan_ip}{port_suffix}，手机与电脑需处于同一网络。"
            return f"{request.scheme}://{lan_ip}{port_suffix}", warning

        warning = "当前二维码详情链接使用 localhost，仅本机可访问。请配置 TARGET_PUBLIC_BASE_URL 或改用局域网 IP 打开系统。"
    return base_url, warning


def build_target_public_url(target_id: str) -> tuple[str, str | None]:
    base_url, warning = get_target_public_base_url()
    relative_path = url_for("view_qr_target", target_id=target_id)
    return f"{base_url}{relative_path}", warning


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


def get_torch_runtime_details() -> list[str]:
    if importlib.util.find_spec("torch") is None:
        return []

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return [f"Unable to inspect torch runtime: {exc}"]

    details = [f"Torch version: {getattr(torch, '__version__', 'unknown')}"]
    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    details.append(f"Torch CUDA build: {cuda_build or 'cpu-only'}")

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        details.append(f"Torch CUDA availability check failed: {exc}")
        return details

    details.append(f"Torch CUDA available: {cuda_available}")
    if cuda_available:
        try:
            details.append(f"Torch GPU count: {torch.cuda.device_count()}")
            details.append(f"Torch GPU 0: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            details.append(f"Torch GPU query failed: {exc}")

    return details


def format_inference_error(exc: Exception, segmentation_service=None) -> str:
    base_message = str(exc)
    dependency_status = get_runtime_dependency_status()
    missing_dependencies = [name for name, installed in dependency_status.items() if not installed]

    details = [
        base_message,
        f"Current Python: {sys.executable}",
        f"Python version: {sys.version.split()[0]}",
        f"INFERENCE_DEVICE: {getattr(segmentation_service, 'device_preference', 'auto')}",
    ]

    if missing_dependencies:
        details.append("Missing modules in current environment: " + ", ".join(missing_dependencies))

    details.extend(get_torch_runtime_details())

    if PREFERRED_PYTHON.exists() and Path(sys.executable).resolve() != PREFERRED_PYTHON.resolve():
        details.append(f"Start the backend with: {PREFERRED_PYTHON} backend\\app.py")

    return " | ".join(details)

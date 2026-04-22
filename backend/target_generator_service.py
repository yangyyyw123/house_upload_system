from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MM_PER_INCH = 25.4


class TargetGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    description: str
    page_width_mm: float
    page_height_mm: float
    qr_size_mm: float
    frame_padding_mm: float
    anchor_size_mm: float
    dpi: int
    crack_size_category: str
    crack_size_label: str
    recommended: bool = False
    is_legacy: bool = False

    @property
    def frame_size_mm(self) -> float:
        return self.qr_size_mm + self.frame_padding_mm * 2


TARGET_SPECS: dict[str, TargetSpec] = {
    "square_120": TargetSpec(
        key="square_120",
        label="120 × 120 mm 大靶标",
        description="适合检测区域较大、裂缝较长或需要更远识别距离的场景。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=96.0,
        frame_padding_mm=12.0,
        anchor_size_mm=12.0,
        dpi=300,
        crack_size_category="large",
        crack_size_label="较大裂缝 / 大范围检测区",
    ),
    "square_80": TargetSpec(
        key="square_80",
        label="80 × 80 mm 标准靶标",
        description="默认推荐，适合常规检测区域和大多数墙体裂缝测量。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=60.0,
        frame_padding_mm=10.0,
        anchor_size_mm=10.0,
        dpi=300,
        crack_size_category="standard",
        crack_size_label="常规裂缝 / 默认推荐",
        recommended=True,
    ),
    "square_40": TargetSpec(
        key="square_40",
        label="40 × 40 mm 小靶标",
        description="适合局部细裂缝和近距离拍摄，打印时建议保持原始比例。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=28.0,
        frame_padding_mm=6.0,
        anchor_size_mm=5.0,
        dpi=600,
        crack_size_category="fine",
        crack_size_label="细裂缝 / 局部检测区",
    ),
    "square_20": TargetSpec(
        key="square_20",
        label="20 × 20 mm 微型靶标",
        description="适合微细裂缝或张贴空间非常受限的部位，建议 600 DPI 打印并近距离正拍。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=14.0,
        frame_padding_mm=3.0,
        anchor_size_mm=3.0,
        dpi=600,
        crack_size_category="micro",
        crack_size_label="微细裂缝 / 狭小张贴区",
    ),
    "a4_qr80": TargetSpec(
        key="a4_qr80",
        label="A4 / 80mm 主靶标（兼容）",
        description="历史兼容规格，建议新项目改用按裂缝尺度选择的新尺寸靶标。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=80.0,
        frame_padding_mm=12.0,
        anchor_size_mm=12.0,
        dpi=300,
        crack_size_category="legacy",
        crack_size_label="历史兼容规格",
        is_legacy=True,
    ),
    "a4_qr60": TargetSpec(
        key="a4_qr60",
        label="A4 / 60mm 紧凑靶标（兼容）",
        description="历史兼容规格，建议新项目改用按裂缝尺度选择的新尺寸靶标。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=60.0,
        frame_padding_mm=10.0,
        anchor_size_mm=10.0,
        dpi=300,
        crack_size_category="legacy",
        crack_size_label="历史兼容规格",
        is_legacy=True,
    ),
    "a5_qr60": TargetSpec(
        key="a5_qr60",
        label="A5 / 60mm 便携靶标（兼容）",
        description="历史兼容规格，建议新项目改用按裂缝尺度选择的新尺寸靶标。",
        page_width_mm=148.0,
        page_height_mm=210.0,
        qr_size_mm=60.0,
        frame_padding_mm=9.0,
        anchor_size_mm=9.0,
        dpi=300,
        crack_size_category="legacy",
        crack_size_label="历史兼容规格",
        is_legacy=True,
    ),
}


class TargetGeneratorService:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).resolve()

    def list_specs(self) -> list[dict[str, Any]]:
        return [self._serialize_spec(spec) for spec in TARGET_SPECS.values() if not spec.is_legacy]

    def generate_target(
        self,
        house_number: str,
        inspection_region: str,
        scene_type: str,
        spec_key: str,
        *,
        target_id: str | None = None,
        created_at: str | None = None,
        scan_url: str | None = None,
        inspection_label: str | None = None,
        report_reference: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        house_number = house_number.strip()
        inspection_region = inspection_region.strip()
        scene_type = scene_type.strip()
        if not house_number:
            raise ValueError("house_number is required")
        if not inspection_region:
            raise ValueError("inspection_region is required")
        if not scene_type:
            raise ValueError("scene_type is required")

        spec = TARGET_SPECS.get(spec_key)
        if spec is None:
            raise ValueError("unsupported spec_key")

        created_at = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        target_id = target_id or f"TGT-{uuid4().hex[:10].upper()}"
        payload = {
            "v": 2,
            "target_id": target_id,
            "house_number": house_number,
            "scene_type": scene_type,
            "qr_size_mm": spec.qr_size_mm,
            "frame_size_mm": spec.frame_size_mm,
            "anchor_size_mm": spec.anchor_size_mm,
        }
        qr_payload = self._build_qr_payload(payload, scan_url)

        output_dir = self.output_root / target_id
        output_dir.mkdir(parents=True, exist_ok=True)

        qr_image = self._build_qr_image(qr_payload, spec)
        page_image = self._compose_page(
            house_number=house_number,
            inspection_region=inspection_region,
            scene_type=scene_type,
            spec=spec,
            target_id=target_id,
            created_at=created_at,
            inspection_label=inspection_label,
            report_reference=report_reference,
            scan_url=scan_url,
            qr_payload=qr_payload,
            qr_image=qr_image,
        )
        self._verify_qr(page_image, qr_payload, spec)

        png_path = output_dir / f"{target_id}.png"
        pdf_path = output_dir / f"{target_id}.pdf"
        manifest_path = output_dir / f"{target_id}.json"

        page_image.save(png_path, format="PNG")
        page_image.save(pdf_path, format="PDF", resolution=spec.dpi)
        manifest_path.write_text(
            json.dumps(
                {
                    "payload": payload,
                    "metadata": {
                        "house_number": house_number,
                        "inspection_region": inspection_region,
                        "scene_type": scene_type,
                        "spec_key": spec.key,
                        "inspection_label": inspection_label,
                        "report_reference": report_reference,
                        "notes": notes,
                        "created_at": created_at,
                        "scan_url": scan_url,
                    },
                    "qr_payload": qr_payload,
                    "spec": {
                        **self._serialize_spec(spec),
                    },
                    "files": {
                        "png": png_path.name,
                        "pdf": pdf_path.name,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "target_id": target_id,
            "payload": payload,
            "metadata": {
                "house_number": house_number,
                "inspection_region": inspection_region,
                "scene_type": scene_type,
                "spec_key": spec.key,
                "inspection_label": inspection_label,
                "report_reference": report_reference,
                "notes": notes,
                "created_at": created_at,
                "scan_url": scan_url,
            },
            "qr_payload": qr_payload,
            "spec": self._serialize_spec(spec),
            "paths": {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "manifest": str(manifest_path),
            },
        }

    @staticmethod
    def _serialize_spec(spec: TargetSpec) -> dict[str, Any]:
        return {
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "page_size_mm": [spec.page_width_mm, spec.page_height_mm],
            "qr_size_mm": spec.qr_size_mm,
            "frame_size_mm": spec.frame_size_mm,
            "target_size_mm": [spec.frame_size_mm, spec.frame_size_mm],
            "anchor_size_mm": spec.anchor_size_mm,
            "dpi": spec.dpi,
            "crack_size_category": spec.crack_size_category,
            "crack_size_label": spec.crack_size_label,
            "recommended": spec.recommended,
        }

    def _build_qr_image(self, payload: str, spec: TargetSpec) -> Image.Image:
        params = cv2.QRCodeEncoder_Params()
        params.mode = cv2.QRCODE_ENCODER_MODE_BYTE
        params.correction_level = cv2.QRCODE_ENCODER_CORRECT_LEVEL_H
        params.version = 0
        encoder = cv2.QRCodeEncoder_create(params)
        qr_matrix = encoder.encode(payload)
        if qr_matrix is None or qr_matrix.size == 0:
            raise TargetGenerationError("failed to generate QR code matrix")

        qr_pixels = self._mm_to_px(spec.qr_size_mm, spec.dpi)
        qr_image = Image.fromarray(qr_matrix).convert("L")
        return qr_image.resize((qr_pixels, qr_pixels), Image.Resampling.NEAREST)

    def _build_qr_payload(self, payload: dict[str, Any], scan_url: str | None) -> str:
        if not scan_url:
            fallback_payload = dict(payload)
            return json.dumps(fallback_payload, ensure_ascii=False, separators=(",", ":"))

        split_result = urlsplit(scan_url)
        query_items = dict(parse_qsl(split_result.query, keep_blank_values=True))
        query_items.update(
            {
                "v": str(payload.get("v") or 2),
                "q": self._format_qr_number(payload.get("qr_size_mm")),
                "f": self._format_qr_number(payload.get("frame_size_mm")),
                "a": self._format_qr_number(payload.get("anchor_size_mm")),
            }
        )
        rebuilt_query = urlencode(query_items)
        return urlunsplit(
            (
                split_result.scheme,
                split_result.netloc,
                split_result.path,
                rebuilt_query,
                split_result.fragment,
            )
        )

    @staticmethod
    def _format_qr_number(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        return f"{numeric:g}"

    def _compose_page(
        self,
        house_number: str,
        inspection_region: str,
        scene_type: str,
        spec: TargetSpec,
        target_id: str,
        created_at: str,
        inspection_label: str | None,
        report_reference: str | None,
        scan_url: str | None,
        qr_payload: str,
        qr_image: Image.Image,
    ) -> Image.Image:
        page_width_px = self._mm_to_px(spec.page_width_mm, spec.dpi)
        page_height_px = self._mm_to_px(spec.page_height_mm, spec.dpi)
        page = Image.new("RGB", (page_width_px, page_height_px), "white")
        draw = ImageDraw.Draw(page)

        title_font = self._load_font(60)
        heading_font = self._load_font(30)
        body_font = self._load_font(22)
        small_font = self._load_font(18)
        mono_font = self._load_font(20)

        outer_margin = self._mm_to_px(12.0, spec.dpi)
        title_top = self._mm_to_px(10.0, spec.dpi)
        draw.text((outer_margin, title_top), "裂缝测量二维码靶标", fill="#173042", font=title_font)
        draw.text(
            (outer_margin, title_top + 78),
            "打印后保持平整，中景照片务必让裂缝与靶标同框、同平面。",
            fill="#516978",
            font=heading_font,
        )

        frame_padding_px = self._mm_to_px(spec.frame_padding_mm, spec.dpi)
        frame_size_px = self._mm_to_px(spec.frame_size_mm, spec.dpi)
        frame_left = (page_width_px - frame_size_px) // 2
        frame_top = self._mm_to_px(58.0, spec.dpi)
        frame_right = frame_left + frame_size_px
        frame_bottom = frame_top + frame_size_px

        draw.rounded_rectangle(
            (frame_left, frame_top, frame_right, frame_bottom),
            radius=28,
            outline="#183042",
            width=5,
        )

        self._draw_corner_anchor(draw, frame_left, frame_top, spec)
        self._draw_corner_anchor(draw, frame_right, frame_top, spec)
        self._draw_corner_anchor(draw, frame_left, frame_bottom, spec)
        self._draw_corner_anchor(draw, frame_right, frame_bottom, spec)

        qr_left = frame_left + frame_padding_px
        qr_top = frame_top + frame_padding_px
        page.paste(qr_image.convert("RGB"), (qr_left, qr_top))

        draw.text(
            (frame_left, frame_bottom + self._mm_to_px(6.0, spec.dpi)),
            f"QR 有效边长: {spec.qr_size_mm:.0f} mm    靶标尺寸: {spec.frame_size_mm:.0f} × {spec.frame_size_mm:.0f} mm",
            fill="#173042",
            font=heading_font,
        )

        meta_top = frame_bottom + self._mm_to_px(18.0, spec.dpi)
        meta_rows = [
            ("房屋编号", house_number),
            ("检测区域", inspection_region),
            ("场景类型", scene_type),
            ("靶标规格", spec.label),
            ("适用裂缝", spec.crack_size_label),
            ("靶标编号", target_id),
            ("生成时间", created_at),
        ]
        if inspection_label:
            meta_rows.insert(3, ("检测时间", inspection_label))
        if report_reference:
            meta_rows.append(("报告编号", report_reference))
        self._draw_meta_panel(
            draw=draw,
            left=outer_margin,
            top=meta_top,
            right=page_width_px - outer_margin,
            rows=meta_rows,
            heading_font=heading_font,
            body_font=body_font,
        )

        footer_top = page_height_px - self._mm_to_px(34.0, spec.dpi)
        draw.text(
            (outer_margin, footer_top),
            "识别规则: 优先使用二维码解码 + 四角锚点定位。请勿裁切靶标边框，避免反光和透视过大。",
            fill="#516978",
            font=small_font,
        )
        if scan_url:
            scan_preview = scan_url if len(scan_url) <= 96 else f"{scan_url[:93]}..."
            draw.text((outer_margin, footer_top + 28), f"扫码查看: {scan_preview}", fill="#516978", font=small_font)
            payload_top = footer_top + 56
        else:
            payload_top = footer_top + 30
        payload_preview = qr_payload if len(qr_payload) <= 120 else f"{qr_payload[:117]}..."
        draw.text((outer_margin, payload_top), payload_preview, fill="#738896", font=mono_font)
        return page

    def _verify_qr(self, page_image: Image.Image, qr_payload: str, spec: TargetSpec) -> None:
        detector = cv2.QRCodeDetector()
        page_bgr = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        decoded_text, _points, _ = detector.detectAndDecode(page_bgr)
        if decoded_text == qr_payload:
            return

        frame_size_px = self._mm_to_px(spec.frame_size_mm, spec.dpi)
        page_width_px, _page_height_px = page_image.size
        frame_left = (page_width_px - frame_size_px) // 2
        frame_top = self._mm_to_px(58.0, spec.dpi)
        crop = page_image.crop((frame_left, frame_top, frame_left + frame_size_px, frame_top + frame_size_px))
        crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        decoded_text, _points, _ = detector.detectAndDecode(crop_bgr)
        if decoded_text == qr_payload:
            return

        frame_padding_px = self._mm_to_px(spec.frame_padding_mm, spec.dpi)
        qr_size_px = self._mm_to_px(spec.qr_size_mm, spec.dpi)
        qr_left = frame_left + frame_padding_px
        qr_top = frame_top + frame_padding_px
        qr_crop = page_image.crop((qr_left, qr_top, qr_left + qr_size_px, qr_top + qr_size_px))
        qr_canvas = Image.new("RGB", (qr_crop.width + 80, qr_crop.height + 80), "white")
        qr_canvas.paste(qr_crop, (40, 40))
        qr_bgr = cv2.cvtColor(np.array(qr_canvas), cv2.COLOR_RGB2BGR)
        decoded_text, _points, _ = detector.detectAndDecode(qr_bgr)
        if decoded_text != qr_payload:
            raise TargetGenerationError("generated QR target failed decode verification")

    def _draw_meta_panel(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        right: int,
        rows: list[tuple[str, str]],
        heading_font: ImageFont.ImageFont,
        body_font: ImageFont.ImageFont,
    ) -> None:
        row_height = 58
        height = 28 + row_height * len(rows)
        draw.rounded_rectangle((left, top, right, top + height), radius=24, fill="#f6fafc", outline="#d7e2ea", width=3)
        draw.text((left + 24, top + 18), "编码元数据", fill="#173042", font=heading_font)
        text_top = top + 74
        midpoint = left + (right - left) // 2
        for index, (label, value) in enumerate(rows):
            current_top = text_top + index * row_height
            draw.text((left + 28, current_top), label, fill="#738896", font=body_font)
            draw.text((midpoint, current_top), value, fill="#173042", font=body_font)

    def _draw_corner_anchor(self, draw: ImageDraw.ImageDraw, center_x: int, center_y: int, spec: TargetSpec) -> None:
        outer = self._mm_to_px(spec.anchor_size_mm, spec.dpi)
        inner = int(round(outer * 0.58))
        core = int(round(outer * 0.28))

        draw.rectangle(
            (
                center_x - outer // 2,
                center_y - outer // 2,
                center_x + outer // 2,
                center_y + outer // 2,
            ),
            fill="black",
        )
        draw.rectangle(
            (
                center_x - inner // 2,
                center_y - inner // 2,
                center_x + inner // 2,
                center_y + inner // 2,
            ),
            fill="white",
        )
        draw.rectangle(
            (
                center_x - core // 2,
                center_y - core // 2,
                center_x + core // 2,
                center_y + core // 2,
            ),
            fill="black",
        )

    @staticmethod
    def _mm_to_px(mm: float, dpi: int) -> int:
        return int(round(mm / MM_PER_INCH * dpi))

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/msyhbd.ttc"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

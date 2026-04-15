from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
    recommended: bool = False

    @property
    def frame_size_mm(self) -> float:
        return self.qr_size_mm + self.frame_padding_mm * 2


TARGET_SPECS: dict[str, TargetSpec] = {
    "a4_qr80": TargetSpec(
        key="a4_qr80",
        label="A4 / 80mm 主靶标",
        description="推荐。优先保证中景照片可识别，适合常规巡检与裂缝精细测量。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=80.0,
        frame_padding_mm=12.0,
        anchor_size_mm=12.0,
        dpi=300,
        recommended=True,
    ),
    "a4_qr60": TargetSpec(
        key="a4_qr60",
        label="A4 / 60mm 紧凑靶标",
        description="适合打印空间受限的现场，但识别距离略短于 80mm 主靶标。",
        page_width_mm=210.0,
        page_height_mm=297.0,
        qr_size_mm=60.0,
        frame_padding_mm=10.0,
        anchor_size_mm=10.0,
        dpi=300,
    ),
    "a5_qr60": TargetSpec(
        key="a5_qr60",
        label="A5 / 60mm 便携靶标",
        description="便于现场随身携带，建议贴近裂缝区域拍摄。",
        page_width_mm=148.0,
        page_height_mm=210.0,
        qr_size_mm=60.0,
        frame_padding_mm=9.0,
        anchor_size_mm=9.0,
        dpi=300,
    ),
}


class TargetGeneratorService:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).resolve()

    def list_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "page_size_mm": [spec.page_width_mm, spec.page_height_mm],
                "qr_size_mm": spec.qr_size_mm,
                "frame_size_mm": spec.frame_size_mm,
                "anchor_size_mm": spec.anchor_size_mm,
                "dpi": spec.dpi,
                "recommended": spec.recommended,
            }
            for spec in TARGET_SPECS.values()
        ]

    def generate_target(
        self,
        house_number: str,
        inspection_region: str,
        scene_type: str,
        spec_key: str,
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

        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        target_id = f"TGT-{uuid4().hex[:10].upper()}"
        payload = {
            "v": 1,
            "target_id": target_id,
            "house_number": house_number,
            "inspection_region": inspection_region,
            "scene_type": scene_type,
            "spec_key": spec.key,
            "qr_size_mm": spec.qr_size_mm,
            "frame_size_mm": spec.frame_size_mm,
            "anchor_size_mm": spec.anchor_size_mm,
            "created_at": created_at,
        }
        qr_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

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
                    "qr_payload": qr_payload,
                    "spec": {
                        "key": spec.key,
                        "label": spec.label,
                        "description": spec.description,
                        "page_size_mm": [spec.page_width_mm, spec.page_height_mm],
                        "qr_size_mm": spec.qr_size_mm,
                        "frame_size_mm": spec.frame_size_mm,
                        "anchor_size_mm": spec.anchor_size_mm,
                        "dpi": spec.dpi,
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
            "qr_payload": qr_payload,
            "spec": {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "page_size_mm": [spec.page_width_mm, spec.page_height_mm],
                "qr_size_mm": spec.qr_size_mm,
                "frame_size_mm": spec.frame_size_mm,
                "anchor_size_mm": spec.anchor_size_mm,
                "dpi": spec.dpi,
            },
            "paths": {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "manifest": str(manifest_path),
            },
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

    def _compose_page(
        self,
        house_number: str,
        inspection_region: str,
        scene_type: str,
        spec: TargetSpec,
        target_id: str,
        created_at: str,
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
            f"QR 有效边长: {spec.qr_size_mm:.0f} mm    靶标外框: {spec.frame_size_mm:.0f} mm",
            fill="#173042",
            font=heading_font,
        )

        meta_top = frame_bottom + self._mm_to_px(18.0, spec.dpi)
        meta_rows = [
            ("房屋编号", house_number),
            ("检测区域", inspection_region),
            ("场景类型", scene_type),
            ("靶标规格", spec.label),
            ("靶标编号", target_id),
            ("生成时间", created_at),
        ]
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
        payload_preview = qr_payload if len(qr_payload) <= 120 else f"{qr_payload[:117]}..."
        draw.text((outer_margin, footer_top + 30), payload_preview, fill="#738896", font=mono_font)
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

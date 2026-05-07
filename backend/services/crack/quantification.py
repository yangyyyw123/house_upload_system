from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGET_SIZE_BY_RATIO = {
    1: 21.7559,
    2: 6.23505,
    5: 1.3334,
    6: 5.833,
    8: 1.875,
}


class QuantificationError(RuntimeError):
    pass


class CrackQuantificationService:
    def __init__(
        self,
        marker_epsilon_factor: float = 0.03,
        marker_aspect_ratio_tolerance: float = 0.45,
    ) -> None:
        self.marker_epsilon_factor = marker_epsilon_factor
        self.marker_aspect_ratio_tolerance = marker_aspect_ratio_tolerance

    def analyze(
        self,
        source_image_path: str | Path,
        mask_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        source_image_path = Path(source_image_path).resolve()
        mask_path = Path(mask_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        source_image = cv2.imread(str(source_image_path))
        if source_image is None:
            raise QuantificationError(f"Unable to read source image: {source_image_path}")

        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise QuantificationError(f"Unable to read mask image: {mask_path}")

        binary_mask = (mask_image > 127).astype(np.uint8)
        marker_detection = self._detect_marker(source_image, output_dir, source_image_path.stem)
        rectification = marker_detection.pop("_rectification", None)
        working_source = source_image
        working_mask = binary_mask
        plane_mode = "image_plane"
        if rectification:
            working_source = self._warp_image_to_rectified_plane(
                source_image,
                rectification["matrix"],
                rectification["size"],
            )
            working_mask = self._warp_mask_to_rectified_plane(
                binary_mask,
                rectification["matrix"],
                rectification["size"],
            )
            plane_mode = "rectified_plane"
        quantification = self._quantify_mask(
            source_image=working_source,
            binary_mask=working_mask,
            mm_per_pixel=marker_detection["physical_scale_mm_per_pixel"],
            output_dir=output_dir,
            stem=source_image_path.stem,
            plane_mode=plane_mode,
        )

        return {
            "marker_detection": marker_detection,
            "quantification": quantification,
        }

    def inspect_target_image(self, image: np.ndarray, include_geometry: bool = False) -> dict[str, Any]:
        detection = self._detect_marker(image, output_dir=None, stem="precheck", include_geometry=include_geometry)
        detection.pop("_rectification", None)
        detection["annotated_image_path"] = None
        return detection

    def _detect_marker(
        self,
        image: np.ndarray,
        output_dir: Path | None,
        stem: str,
        include_geometry: bool = False,
    ) -> dict[str, Any]:
        annotated = image.copy()
        qr_candidates = self._detect_qr_candidates(image)
        candidates = qr_candidates
        rectification = None
        detection_method = "qr_decode"
        status = "qr_detected"
        message = "qr marker detected"

        if candidates:
            rectified_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                refined = self._refine_qr_candidate_with_anchors(image, candidate)
                rectified_candidates.append(refined)
            candidates = rectified_candidates
            best_candidate = max(
                candidates,
                key=lambda item: (
                    1 if item.get("perspective_corrected") else 0,
                    int(item.get("anchor_count") or 0),
                    float(item.get("avg_side_px") or 0.0),
                ),
            )
            candidates = [best_candidate]
            if best_candidate.get("perspective_corrected"):
                detection_method = "qr_anchor_rectified"
                status = "anchor_rectified"
                message = "qr marker and four corner anchors detected, perspective rectified"
                rectification = best_candidate.get("rectification")
            else:
                detection_method = "qr_decode"
                status = "qr_detected"
                message = "qr marker detected, but corner anchors were insufficient for perspective rectification"
        else:
            candidates = self._detect_legacy_candidates(image)
            detection_method = "legacy_rect"
            if candidates:
                status = "fallback_detected"
                message = "qr marker not found, fallback rectangle marker estimated"
            else:
                status = "not_found"
                message = "marker not found"

        self._annotate_marker_candidates(annotated, candidates, detection_method)
        annotated_path = output_dir / f"{stem}_target_overlay.png" if output_dir else None
        if not candidates:
            cv2.putText(
                annotated,
                "Marker not found",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (36, 64, 220),
                2,
                cv2.LINE_AA,
            )
        elif detection_method == "qr_anchor_rectified":
            cv2.putText(
                annotated,
                "Perspective rectification enabled",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (18, 112, 72),
                2,
                cv2.LINE_AA,
            )
        if annotated_path is not None:
            cv2.imwrite(str(annotated_path), annotated)

        mm_per_pixel = self._calculate_mm_per_pixel(candidates, detection_method)
        return {
            "status": status,
            "message": message,
            "marker_count": len(candidates),
            "physical_scale_mm_per_pixel": mm_per_pixel,
            "annotated_image_path": str(annotated_path) if annotated_path else None,
            "detection_method": detection_method if candidates else None,
            "anchor_count": max((int(info.get("anchor_count") or 0) for info in candidates), default=0),
            "perspective_corrected": detection_method == "qr_anchor_rectified",
            "targets": [
                self._serialize_target(
                    index=index,
                    info=info,
                    detection_method=detection_method,
                    include_geometry=include_geometry,
                )
                for index, info in enumerate(candidates)
            ],
            "_rectification": rectification,
        }

    def _detect_qr_candidates(self, image: np.ndarray) -> list[dict[str, Any]]:
        detector = cv2.QRCodeDetector()
        candidates: list[dict[str, Any]] = []
        payloads: list[tuple[str, np.ndarray]] = []

        multi_detect = getattr(detector, "detectAndDecodeMulti", None)
        if callable(multi_detect):
            ok, decoded_info, points, _ = multi_detect(image)
            if ok and decoded_info and points is not None:
                for decoded_text, qr_points in zip(decoded_info, points):
                    if decoded_text:
                        payloads.append((decoded_text, np.asarray(qr_points, dtype=np.float32)))

        if not payloads:
            decoded_text, points, _ = detector.detectAndDecode(image)
            if decoded_text and points is not None:
                payloads.append((decoded_text, np.asarray(points, dtype=np.float32).reshape(4, 2)))

        for decoded_text, points in payloads:
            payload = self._parse_qr_payload(decoded_text)
            qr_size_mm = payload.get("qr_size_mm") if payload else None
            if not qr_size_mm:
                continue

            qr_world = np.array(
                [[0.0, 0.0], [qr_size_mm, 0.0], [qr_size_mm, qr_size_mm], [0.0, qr_size_mm]],
                dtype=np.float32,
            )
            target_to_image_matrix = cv2.getPerspectiveTransform(qr_world, points.astype(np.float32))
            image_to_target_matrix = cv2.getPerspectiveTransform(points.astype(np.float32), qr_world)

            center = points.mean(axis=0)
            side_lengths = self._polygon_side_lengths(points)
            avg_side_px = float(sum(side_lengths) / len(side_lengths))
            if avg_side_px <= 0:
                continue

            height_px = float((side_lengths[1] + side_lengths[3]) / 2.0)
            width_px = float((side_lengths[0] + side_lengths[2]) / 2.0)
            angle_deg = math.degrees(math.atan2(float(points[1][1] - points[0][1]), float(points[1][0] - points[0][0])))

            candidates.append(
                {
                    "contour": points,
                    "center": (float(center[0]), float(center[1])),
                    "width": width_px,
                    "height": height_px,
                    "angle": float(angle_deg),
                    "size": float(qr_size_mm),
                    "avg_side_px": avg_side_px,
                    "decoded_payload": decoded_text,
                    "target_id": payload.get("target_id"),
                    "house_number": payload.get("house_number"),
                    "scene_type": payload.get("scene_type"),
                    "frame_size_mm": float(payload.get("frame_size_mm") or qr_size_mm),
                    "anchor_size_mm": float(payload.get("anchor_size_mm") or 0.0),
                    "target_to_image_matrix": target_to_image_matrix.astype(np.float32),
                    "image_to_target_matrix": image_to_target_matrix.astype(np.float32),
                }
            )

        return self._deduplicate_candidates(candidates, distance_threshold=24.0)

    def _refine_qr_candidate_with_anchors(self, image: np.ndarray, candidate: dict[str, Any]) -> dict[str, Any]:
        qr_size_mm = float(candidate["size"])
        frame_size_mm = float(candidate.get("frame_size_mm") or qr_size_mm)
        anchor_size_mm = float(candidate.get("anchor_size_mm") or 0.0)
        padding_mm = max((frame_size_mm - qr_size_mm) / 2.0, 0.0)
        qr_points = np.asarray(candidate["contour"], dtype=np.float32).reshape(4, 2)

        qr_world = np.array(
            [[0.0, 0.0], [qr_size_mm, 0.0], [qr_size_mm, qr_size_mm], [0.0, qr_size_mm]],
            dtype=np.float32,
        )
        frame_world = np.array(
            [
                [-padding_mm, -padding_mm],
                [qr_size_mm + padding_mm, -padding_mm],
                [qr_size_mm + padding_mm, qr_size_mm + padding_mm],
                [-padding_mm, qr_size_mm + padding_mm],
            ],
            dtype=np.float32,
        )
        qr_homography = cv2.getPerspectiveTransform(qr_world, qr_points.astype(np.float32))
        predicted_frame = cv2.perspectiveTransform(frame_world.reshape(1, -1, 2), qr_homography).reshape(4, 2)

        matched_anchor_centers: list[np.ndarray | None] = []
        for corner_index in range(4):
            expected_quad = self._predict_anchor_quad(
                qr_homography,
                frame_world[corner_index],
                anchor_size_mm,
            )
            expected_center = predicted_frame[corner_index]
            expected_side = max(6.0, float(sum(self._polygon_side_lengths(expected_quad)) / 4.0))
            matched_center, score = self._match_anchor_center(image, expected_center, expected_side)
            if matched_center is None or score < 0.35:
                matched_anchor_centers.append(None)
                continue
            matched_anchor_centers.append(np.asarray(matched_center, dtype=np.float32))

        anchor_count = sum(1 for center in matched_anchor_centers if center is not None)
        refined = dict(candidate)
        refined["anchor_count"] = anchor_count
        refined["predicted_frame_corners"] = predicted_frame
        refined["perspective_corrected"] = False

        if anchor_count == 4:
            frame_corners = np.array([center for center in matched_anchor_centers if center is not None], dtype=np.float32)
            rectification = self._build_rectification(frame_corners, frame_size_mm, image.shape)
            refined["frame_corners"] = frame_corners
            refined["perspective_corrected"] = True
            refined["rectification"] = rectification
            refined["avg_frame_side_px"] = rectification["pixels_per_mm"] * frame_size_mm
            refined["avg_side_px"] = rectification["pixels_per_mm"] * qr_size_mm
        else:
            refined["frame_corners"] = predicted_frame

        return refined

    def _match_anchor_center(
        self,
        image: np.ndarray,
        expected_center: np.ndarray,
        expected_side_px: float,
    ) -> tuple[np.ndarray | None, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        expected_side_px = max(10.0, expected_side_px)
        roi_radius = int(round(expected_side_px * 1.5))
        center_x = int(round(float(expected_center[0])))
        center_y = int(round(float(expected_center[1])))
        x0 = max(0, center_x - roi_radius)
        y0 = max(0, center_y - roi_radius)
        x1 = min(gray.shape[1], center_x + roi_radius)
        y1 = min(gray.shape[0], center_y + roi_radius)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return None, 0.0

        best_center = None
        best_score = -1.0
        for scale in (0.8, 1.0, 1.2):
            side_px = max(12, int(round(expected_side_px * scale)))
            template = self._build_anchor_template(side_px)
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                continue
            response = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(response)
            if max_val > best_score:
                best_score = float(max_val)
                best_center = np.array(
                    [
                        x0 + max_loc[0] + template.shape[1] / 2.0,
                        y0 + max_loc[1] + template.shape[0] / 2.0,
                    ],
                    dtype=np.float32,
                )

        return best_center, best_score

    @staticmethod
    def _build_anchor_template(side_px: int) -> np.ndarray:
        side_px = max(12, side_px)
        template = np.full((side_px, side_px), 255, dtype=np.uint8)
        inner = max(4, int(round(side_px * 0.58)))
        core = max(2, int(round(side_px * 0.28)))
        outer_offset = 0
        inner_offset = (side_px - inner) // 2
        core_offset = (side_px - core) // 2
        template[outer_offset:side_px, outer_offset:side_px] = 0
        template[inner_offset:inner_offset + inner, inner_offset:inner_offset + inner] = 255
        template[core_offset:core_offset + core, core_offset:core_offset + core] = 0
        return template

    @staticmethod
    def _predict_anchor_quad(qr_homography: np.ndarray, center_world: np.ndarray, anchor_size_mm: float) -> np.ndarray:
        if anchor_size_mm <= 0:
            return np.repeat(center_world.reshape(1, 2), 4, axis=0)
        half_size = anchor_size_mm / 2.0
        anchor_world = np.array(
            [
                [center_world[0] - half_size, center_world[1] - half_size],
                [center_world[0] + half_size, center_world[1] - half_size],
                [center_world[0] + half_size, center_world[1] + half_size],
                [center_world[0] - half_size, center_world[1] + half_size],
            ],
            dtype=np.float32,
        )
        return cv2.perspectiveTransform(anchor_world.reshape(1, -1, 2), qr_homography).reshape(4, 2)

    def _build_rectification(
        self,
        frame_corners: np.ndarray,
        frame_size_mm: float,
        image_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        image_height, image_width = image_shape[:2]
        frame_world = np.array(
            [[0.0, 0.0], [frame_size_mm, 0.0], [frame_size_mm, frame_size_mm], [0.0, frame_size_mm]],
            dtype=np.float32,
        )
        image_to_world = cv2.getPerspectiveTransform(frame_corners.astype(np.float32), frame_world)
        frame_side_lengths = self._polygon_side_lengths(frame_corners.astype(np.float32))
        pixels_per_mm = max(3.0, min(40.0, float(sum(frame_side_lengths) / len(frame_side_lengths)) / max(frame_size_mm, 1e-6)))

        image_corners = np.array(
            [[0.0, 0.0], [image_width - 1.0, 0.0], [image_width - 1.0, image_height - 1.0], [0.0, image_height - 1.0]],
            dtype=np.float32,
        )
        world_corners = cv2.perspectiveTransform(image_corners.reshape(1, -1, 2), image_to_world).reshape(4, 2)
        min_x = float(world_corners[:, 0].min())
        min_y = float(world_corners[:, 1].min())
        max_x = float(world_corners[:, 0].max())
        max_y = float(world_corners[:, 1].max())

        translate_and_scale = np.array(
            [
                [pixels_per_mm, 0.0, -min_x * pixels_per_mm],
                [0.0, pixels_per_mm, -min_y * pixels_per_mm],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        warp_matrix = translate_and_scale @ image_to_world
        width_px = max(64, int(math.ceil((max_x - min_x) * pixels_per_mm)))
        height_px = max(64, int(math.ceil((max_y - min_y) * pixels_per_mm)))
        return {
            "matrix": warp_matrix.astype(np.float32),
            "size": (width_px, height_px),
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": round(1.0 / pixels_per_mm, 6),
            "frame_size_mm": frame_size_mm,
        }

    @staticmethod
    def _warp_image_to_rectified_plane(image: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        return cv2.warpPerspective(
            image,
            matrix,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    @staticmethod
    def _warp_mask_to_rectified_plane(mask: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        warped = cv2.warpPerspective(
            (mask > 0).astype(np.uint8) * 255,
            matrix,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return (warped > 127).astype(np.uint8)

    def _detect_legacy_candidates(self, image: np.ndarray) -> list[dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(600, int(image.shape[0] * image.shape[1] * 0.00008))
        candidates: list[dict[str, Any]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            epsilon = self.marker_epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            angle_deviation = self._quadrilateral_angle_deviation(approx.reshape(4, 2))
            if angle_deviation > 18:
                continue

            rect = cv2.minAreaRect(approx)
            (center_x, center_y), (w, h), angle = rect
            if min(w, h) <= 0:
                continue
            width, height = (w, h) if w >= h else (h, w)
            aspect_ratio = width / max(height, 1e-6)
            marker_size = self._infer_marker_size(aspect_ratio)
            if marker_size is None:
                continue

            candidates.append(
                {
                    "contour": approx.reshape(4, 2),
                    "center": (float(center_x), float(center_y)),
                    "width": float(width),
                    "height": float(height),
                    "angle": float(angle),
                    "size": marker_size,
                    "aspect_ratio": float(aspect_ratio),
                }
            )
        return self._deduplicate_candidates(candidates, distance_threshold=12.0)

    def _annotate_marker_candidates(
        self,
        annotated: np.ndarray,
        candidates: list[dict[str, Any]],
        detection_method: str,
    ) -> None:
        for index, info in enumerate(candidates, start=1):
            box = np.asarray(info["contour"], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(annotated, [box], True, (48, 176, 96), 2, cv2.LINE_AA)
            for point in box:
                cv2.circle(annotated, tuple(int(v) for v in point), 5, (32, 96, 224), -1, cv2.LINE_AA)

            if detection_method in {"qr_decode", "qr_anchor_rectified"}:
                target_id = info.get("target_id") or f"QR-{index}"
                prefix = "QR+ANCHOR" if detection_method == "qr_anchor_rectified" else "QR"
                label = f"#{index} {prefix} {info['size']:.0f}mm {target_id}"
            else:
                label = f"#{index} fallback {info['size']:.0f}mm"
            anchor = tuple(box[0])
            cv2.putText(
                annotated,
                label,
                (int(anchor[0]), max(28, int(anchor[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (32, 96, 224),
                2,
                cv2.LINE_AA,
            )

    def _quantify_mask(
        self,
        source_image: np.ndarray,
        binary_mask: np.ndarray,
        mm_per_pixel: float | None,
        output_dir: Path,
        stem: str,
        plane_mode: str = "image_plane",
    ) -> dict[str, Any]:
        cleaned_mask, cleaned_labels, component_stats = self._clean_mask(binary_mask)
        if not component_stats:
            overlay_path = output_dir / f"{stem}_quant_overlay.png"
            chart_path = output_dir / f"{stem}_width_chart.png"
            cv2.imwrite(str(overlay_path), source_image)
            self._build_placeholder_chart(chart_path, "No crack pixels available")
            return {
                "status": "unavailable",
                "message": "no usable crack mask",
                "plane_mode": plane_mode,
                "component_count": 0,
                "max_width_px": None,
                "avg_width_px": None,
                "median_width_px": None,
                "crack_length_px": None,
                "crack_angle_deg": None,
                "max_width_mm": None,
                "avg_width_mm": None,
                "median_width_mm": None,
                "crack_length_mm": None,
                "quant_overlay_path": str(overlay_path),
                "width_chart_path": str(chart_path),
            }

        skeleton = self._skeletonize(cleaned_mask)
        distance_map = cv2.distanceTransform((cleaned_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
        skeleton_coords = np.argwhere(skeleton > 0)
        if skeleton_coords.size == 0:
            overlay_path = output_dir / f"{stem}_quant_overlay.png"
            chart_path = output_dir / f"{stem}_width_chart.png"
            cv2.imwrite(str(overlay_path), source_image)
            self._build_placeholder_chart(chart_path, "Skeleton extraction failed")
            return {
                "status": "unavailable",
                "message": "unable to build skeleton",
                "plane_mode": plane_mode,
                "component_count": len(component_stats),
                "max_width_px": None,
                "avg_width_px": None,
                "median_width_px": None,
                "crack_length_px": None,
                "crack_angle_deg": None,
                "max_width_mm": None,
                "avg_width_mm": None,
                "median_width_mm": None,
                "crack_length_mm": None,
                "quant_overlay_path": str(overlay_path),
                "width_chart_path": str(chart_path),
            }

        width_samples_px = np.array([distance_map[y, x] * 2.0 for y, x in skeleton_coords], dtype=np.float32)
        max_index = int(np.argmax(width_samples_px))
        max_point = tuple(int(v) for v in skeleton_coords[max_index])

        total_length_px = 0.0
        component_angles: list[tuple[float, int]] = []
        for label_value, area_px in component_stats:
            component_mask = (cleaned_labels == label_value).astype(np.uint8)
            component_skeleton = (skeleton > 0) & (component_mask > 0)
            length_px = self._estimate_skeleton_length(component_skeleton.astype(np.uint8))
            total_length_px += length_px
            coords = np.argwhere(component_mask > 0)
            angle = self._estimate_component_angle(coords)
            if angle is not None:
                component_angles.append((angle, area_px))

        representative_angle = None
        if component_angles:
            representative_angle = max(component_angles, key=lambda item: item[1])[0]

        overlay_path = output_dir / f"{stem}_quant_overlay.png"
        chart_path = output_dir / f"{stem}_width_chart.png"
        self._build_quant_overlay(
            source_image=source_image,
            cleaned_mask=cleaned_mask,
            skeleton=skeleton,
            max_point=max_point,
            max_width_px=float(width_samples_px[max_index]),
            angle_deg=representative_angle,
            mm_per_pixel=mm_per_pixel,
            output_path=overlay_path,
        )
        self._build_width_chart(width_samples_px, mm_per_pixel, chart_path)

        max_width_px = round(float(width_samples_px.max()), 3)
        avg_width_px = round(float(width_samples_px.mean()), 3)
        median_width_px = round(float(np.median(width_samples_px)), 3)
        crack_length_px = round(float(total_length_px), 3)

        return {
            "status": "completed" if mm_per_pixel else "pixel_only",
            "message": (
                "perspective-rectified quantification completed"
                if mm_per_pixel and plane_mode == "rectified_plane"
                else "quantification completed"
                if mm_per_pixel
                else "pixel-only quantification completed"
            ),
            "plane_mode": plane_mode,
            "component_count": len(component_stats),
            "sample_count": int(width_samples_px.size),
            "max_width_px": max_width_px,
            "avg_width_px": avg_width_px,
            "median_width_px": median_width_px,
            "crack_length_px": crack_length_px,
            "crack_angle_deg": round(representative_angle, 2) if representative_angle is not None else None,
            "max_width_mm": round(max_width_px * mm_per_pixel, 3) if mm_per_pixel else None,
            "avg_width_mm": round(avg_width_px * mm_per_pixel, 3) if mm_per_pixel else None,
            "median_width_mm": round(median_width_px * mm_per_pixel, 3) if mm_per_pixel else None,
            "crack_length_mm": round(crack_length_px * mm_per_pixel, 3) if mm_per_pixel else None,
            "quant_overlay_path": str(overlay_path),
            "width_chart_path": str(chart_path),
        }

    @staticmethod
    def _quadrilateral_angle_deviation(points: np.ndarray) -> float:
        deviations: list[float] = []
        for index in range(4):
            current = points[index].astype(np.float32)
            previous = points[index - 1].astype(np.float32)
            nxt = points[(index + 1) % 4].astype(np.float32)
            v1 = previous - current
            v2 = nxt - current
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom == 0:
                return 180.0
            cos_theta = np.clip(float(np.dot(v1, v2) / denom), -1.0, 1.0)
            angle = math.degrees(math.acos(cos_theta))
            deviations.append(abs(angle - 90.0))
        return max(deviations, default=180.0)

    @staticmethod
    def _polygon_side_lengths(points: np.ndarray) -> list[float]:
        return [
            float(np.linalg.norm(points[(index + 1) % len(points)] - points[index]))
            for index in range(len(points))
        ]

    @staticmethod
    def _parse_qr_payload(decoded_text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(decoded_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload

        parsed_url = urlparse(decoded_text)
        if not parsed_url.scheme or not parsed_url.netloc:
            return None

        path_segments = [segment for segment in parsed_url.path.split("/") if segment]
        target_id = None
        if len(path_segments) >= 2 and path_segments[-2] == "targets":
            target_id = unquote(path_segments[-1])
        elif path_segments:
            target_id = unquote(path_segments[-1])

        query = parse_qs(parsed_url.query)

        def get_float(name: str) -> float | None:
            values = query.get(name)
            if not values:
                return None
            try:
                return float(values[0])
            except (TypeError, ValueError):
                return None

        qr_size_mm = get_float("q") or get_float("qr") or get_float("qr_size_mm")
        frame_size_mm = get_float("f") or get_float("frame") or get_float("frame_size_mm")
        anchor_size_mm = get_float("a") or get_float("anchor") or get_float("anchor_size_mm")
        if qr_size_mm is None:
            return None

        return {
            "v": int((query.get("v") or ["2"])[0] or 2),
            "target_id": target_id,
            "qr_size_mm": qr_size_mm,
            "frame_size_mm": frame_size_mm or qr_size_mm,
            "anchor_size_mm": anchor_size_mm or 0.0,
            "scan_url": decoded_text,
        }

    def _infer_marker_size(self, aspect_ratio: float) -> int | None:
        best_size = None
        best_delta = None
        for size_mm, reference_ratio in TARGET_SIZE_BY_RATIO.items():
            delta = abs(aspect_ratio - reference_ratio)
            if best_delta is None or delta < best_delta:
                best_size = size_mm
                best_delta = delta
        if best_delta is None or best_delta > self.marker_aspect_ratio_tolerance:
            return None
        return best_size

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[dict[str, Any]],
        distance_threshold: float = 12.0,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item["width"] * item["height"], reverse=True):
            center_x, center_y = candidate["center"]
            if any(
                math.dist((center_x, center_y), existing["center"]) < distance_threshold
                for existing in filtered
            ):
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _calculate_mm_per_pixel(targets: list[dict[str, Any]], detection_method: str) -> float | None:
        if not targets:
            return None
        if detection_method == "qr_anchor_rectified":
            ratios = [info["rectification"]["mm_per_pixel"] for info in targets if info.get("rectification")]
        elif detection_method == "qr_decode":
            ratios = [info["size"] / info["avg_side_px"] for info in targets if info.get("avg_side_px", 0) > 0]
        else:
            ratios = [info["size"] / info["height"] for info in targets if info["height"] > 0]
        if not ratios:
            return None
        return round(float(sum(ratios) / len(ratios)), 6)

    @staticmethod
    def _serialize_target(
        index: int,
        info: dict[str, Any],
        detection_method: str,
        include_geometry: bool = False,
    ) -> dict[str, Any]:
        target = {
            "id": index + 1,
            "center": [round(info["center"][0], 1), round(info["center"][1], 1)],
            "width_px": round(info["width"], 2),
            "height_px": round(info["height"], 2),
            "angle_deg": round(info["angle"], 2),
            "size_mm": info["size"],
            "detection_method": detection_method,
        }
        if detection_method == "qr_decode":
            target["target_id"] = info.get("target_id")
            target["house_number"] = info.get("house_number")
            target["scene_type"] = info.get("scene_type")
        if detection_method == "qr_anchor_rectified":
            target["target_id"] = info.get("target_id")
            target["house_number"] = info.get("house_number")
            target["scene_type"] = info.get("scene_type")
            target["anchor_count"] = int(info.get("anchor_count") or 0)
            target["perspective_corrected"] = True
        if include_geometry:
            contour_value = info.get("contour")
            frame_corners_value = info.get("frame_corners")
            contour = np.asarray(contour_value if contour_value is not None else [], dtype=np.float32).reshape(-1, 2)
            frame_corners = np.asarray(
                frame_corners_value if frame_corners_value is not None else [],
                dtype=np.float32,
            ).reshape(-1, 2)
            target["contour"] = [[round(float(x), 2), round(float(y), 2)] for x, y in contour]
            target["frame_corners"] = [[round(float(x), 2), round(float(y), 2)] for x, y in frame_corners]
            target["image_to_target_matrix"] = (
                np.asarray(info.get("image_to_target_matrix"), dtype=np.float32).round(6).tolist()
                if info.get("image_to_target_matrix") is not None
                else None
            )
            target["target_to_image_matrix"] = (
                np.asarray(info.get("target_to_image_matrix"), dtype=np.float32).round(6).tolist()
                if info.get("target_to_image_matrix") is not None
                else None
            )
            target["plane_unit"] = "mm"
        return target

    @staticmethod
    def _clean_mask(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        cleaned = np.zeros_like(binary_mask, dtype=np.uint8)
        cleaned_labels = np.zeros_like(labels, dtype=np.int32)
        kept_components: list[tuple[int, int]] = []
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            cleaned[labels == label] = 1
            cleaned_labels[labels == label] = label
            kept_components.append((label, area))
        return cleaned, cleaned_labels, kept_components

    @staticmethod
    def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        working = (binary_mask > 0).astype(np.uint8) * 255
        skeleton = np.zeros_like(working)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(working, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(working, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            working = eroded
            if cv2.countNonZero(working) == 0:
                break
        return (skeleton > 0).astype(np.uint8)

    def _estimate_skeleton_length(self, skeleton_mask: np.ndarray) -> float:
        coords = np.argwhere(skeleton_mask > 0)
        if coords.shape[0] < 2:
            return float(coords.shape[0])

        coord_set = {tuple(int(v) for v in coord): index for index, coord in enumerate(coords)}
        endpoints = [tuple(int(v) for v in coord) for coord in coords if self._neighbor_count(tuple(coord), coord_set) <= 1]
        start = endpoints[0] if endpoints else tuple(int(v) for v in coords[0])
        _, farthest = self._dijkstra_longest_distance(start, coord_set)
        max_distance, _ = self._dijkstra_longest_distance(farthest, coord_set)
        return round(max_distance, 3)

    @staticmethod
    def _neighbor_count(coord: tuple[int, int], coord_set: dict[tuple[int, int], int]) -> int:
        count = 0
        y, x = coord
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                if offset_y == 0 and offset_x == 0:
                    continue
                if (y + offset_y, x + offset_x) in coord_set:
                    count += 1
        return count

    @staticmethod
    def _dijkstra_longest_distance(
        start: tuple[int, int],
        coord_set: dict[tuple[int, int], int],
    ) -> tuple[float, tuple[int, int]]:
        distances: dict[tuple[int, int], float] = {start: 0.0}
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        farthest = start
        farthest_distance = 0.0
        while heap:
            current_distance, current = heapq.heappop(heap)
            if current_distance > distances.get(current, float("inf")):
                continue
            if current_distance > farthest_distance:
                farthest_distance = current_distance
                farthest = current

            y, x = current
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_y == 0 and offset_x == 0:
                        continue
                    neighbor = (y + offset_y, x + offset_x)
                    if neighbor not in coord_set:
                        continue
                    step_distance = math.sqrt(2.0) if offset_y and offset_x else 1.0
                    next_distance = current_distance + step_distance
                    if next_distance < distances.get(neighbor, float("inf")):
                        distances[neighbor] = next_distance
                        heapq.heappush(heap, (next_distance, neighbor))
        return farthest_distance, farthest

    @staticmethod
    def _estimate_component_angle(coords: np.ndarray) -> float | None:
        if coords.shape[0] < 2:
            return None
        xy = np.column_stack((coords[:, 1], coords[:, 0])).astype(np.float32)
        centered = xy - xy.mean(axis=0, keepdims=True)
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
        angle = math.degrees(math.atan2(float(principal_vector[1]), float(principal_vector[0])))
        return angle

    def _build_quant_overlay(
        self,
        source_image: np.ndarray,
        cleaned_mask: np.ndarray,
        skeleton: np.ndarray,
        max_point: tuple[int, int],
        max_width_px: float,
        angle_deg: float | None,
        mm_per_pixel: float | None,
        output_path: Path,
    ) -> None:
        overlay = source_image.copy()
        contours, _ = cv2.findContours((cleaned_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (48, 176, 255), 2)
        overlay[skeleton > 0] = (255, 255, 0)

        center = (int(max_point[1]), int(max_point[0]))
        cv2.circle(overlay, center, 6, (0, 0, 255), -1)

        label = f"Max width {max_width_px:.2f}px"
        if mm_per_pixel:
            label += f" / {max_width_px * mm_per_pixel:.2f}mm"
        cv2.putText(overlay, label, (center[0] + 10, max(24, center[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        if angle_deg is not None:
            radians = math.radians(angle_deg)
            length = 100
            dx = int(math.cos(radians) * length)
            dy = int(math.sin(radians) * length)
            cv2.line(overlay, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), (60, 220, 120), 2)
            cv2.putText(overlay, f"Angle {angle_deg:.1f} deg", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 220, 120), 2, cv2.LINE_AA)

        cv2.imwrite(str(output_path), overlay)

    def _build_width_chart(self, width_samples_px: np.ndarray, mm_per_pixel: float | None, output_path: Path) -> None:
        values = width_samples_px * mm_per_pixel if mm_per_pixel else width_samples_px
        unit = "mm" if mm_per_pixel else "px"
        hist, edges = np.histogram(values, bins=min(7, max(4, int(np.sqrt(values.size)))))
        chart = Image.new("RGB", (1200, 620), "white")
        draw = ImageDraw.Draw(chart)
        title_font = self._load_font(34)
        body_font = self._load_font(22)
        small_font = self._load_font(18)
        draw.text((54, 32), f"Width Histogram ({unit})", fill="#173042", font=title_font)
        draw.text((54, 78), f"Samples: {values.size}", fill="#607887", font=body_font)

        left, right = 90, 1120
        top, bottom = 160, 520
        draw.line((left, top, left, bottom), fill="#8AA0AE", width=3)
        draw.line((left, bottom, right, bottom), fill="#8AA0AE", width=3)

        max_count = int(hist.max()) if hist.size else 1
        bar_gap = 18
        bar_width = max(36, int((right - left - bar_gap * (len(hist) - 1)) / max(len(hist), 1)))

        for index, count in enumerate(hist):
            bar_left = left + index * (bar_width + bar_gap) + 12
            bar_right = bar_left + bar_width
            ratio = count / max(max_count, 1)
            bar_top = bottom - int((bottom - top - 20) * ratio)
            draw.rounded_rectangle((bar_left, bar_top, bar_right, bottom), radius=10, fill="#59A37A")
            draw.text((bar_left + 4, max(top, bar_top - 28)), str(int(count)), fill="#173042", font=small_font)
            range_label = f"{edges[index]:.2f}-{edges[index + 1]:.2f}"
            draw.text((bar_left - 10, bottom + 18), range_label, fill="#607887", font=small_font)

        chart.save(output_path)

    def _build_placeholder_chart(self, output_path: Path, message: str) -> None:
        chart = Image.new("RGB", (1200, 620), "white")
        draw = ImageDraw.Draw(chart)
        title_font = self._load_font(32)
        body_font = self._load_font(24)
        draw.text((54, 42), "Width Histogram", fill="#173042", font=title_font)
        draw.text((54, 114), message, fill="#7A8F9C", font=body_font)
        chart.save(output_path)

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        candidates = [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

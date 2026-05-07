from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "model"


class InferenceDependencyError(RuntimeError):
    pass


class ModelConfigurationError(RuntimeError):
    pass


def _pick_first_file(pattern: str) -> Path:
    matches = sorted(MODEL_DIR.glob(pattern))
    if not matches:
        raise ModelConfigurationError(f"Unable to find model asset matching: {pattern}")
    return matches[0]


class CrackSegmentationService:
    def __init__(
        self,
        model_path: Path | None = None,
        hyperparams_path: Path | None = None,
        threshold: float = 0.5,
        device_preference: str | None = None,
    ) -> None:
        self.model_path = model_path or _pick_first_file("*_best_model.pth")
        self.hyperparams_path = hyperparams_path or _pick_first_file("*_hyperparams.json")
        self.threshold = threshold
        requested_device = (device_preference or os.getenv("INFERENCE_DEVICE", "auto")).strip().lower()
        self._hyperparams: dict[str, Any] | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._device_name: str | None = None
        self._device_reason: str | None = None
        self._torch_cuda_version: str | None = None
        self._torch: Any | None = None
        self.device_preference = requested_device or "auto"

    def segment_image(self, image_path: str | Path, output_root: str | Path) -> dict[str, Any]:
        image_path = Path(image_path).resolve()
        output_root = Path(output_root).resolve()

        cv2 = self._import_dependency("cv2", "opencv-python-headless")
        np = self._import_dependency("numpy", "numpy")
        from PIL import Image

        model = self._get_model()
        patch_size = int(self._get_hyperparams()["image_height"])

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        original_height, original_width = image.shape[:2]
        padded_image, padding = self._pad_to_multiple(image, patch_size, cv2)
        patch_masks, patch_count = self._predict_patches(padded_image, patch_size, model, np)
        full_mask = self._stitch_masks(patch_masks, padded_image.shape[:2], patch_size, np)
        cropped_mask = self._crop_padding(full_mask, padding, original_height, original_width)

        mask_uint8 = (cropped_mask * 255).astype(np.uint8)
        overlay = self._build_overlay(image, mask_uint8, cv2, np)

        result_dir = output_root / image_path.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        mask_path = result_dir / f"{image_path.stem}_mask.png"
        overlay_path = result_dir / f"{image_path.stem}_overlay.png"
        Image.fromarray(mask_uint8).save(mask_path)
        cv2.imwrite(str(overlay_path), overlay)

        crack_pixel_count = int(cropped_mask.sum())
        crack_area_ratio = float(crack_pixel_count / max(1, original_height * original_width))

        return {
            "source_image_path": str(image_path),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "original_size": {"width": int(original_width), "height": int(original_height)},
            "patch_size": patch_size,
            "patch_count": patch_count,
            "crack_pixel_count": crack_pixel_count,
            "crack_area_ratio": round(crack_area_ratio, 6),
            "threshold": self.threshold,
            "device": str(self._device),
            "device_requested": self.device_preference,
            "device_name": self._device_name,
            "device_reason": self._device_reason,
            "torch_cuda_version": self._torch_cuda_version,
        }

    def _get_hyperparams(self) -> dict[str, Any]:
        if self._hyperparams is None:
            with self.hyperparams_path.open("r", encoding="utf-8") as handle:
                self._hyperparams = json.load(handle)
        return self._hyperparams

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        torch = self._import_dependency("torch", "torch")
        smp = self._import_dependency("segmentation_models_pytorch", "segmentation-models-pytorch")

        hyperparams = self._get_hyperparams()
        model_arch_name = hyperparams["pretrained_model_arch"]
        try:
            model_arch = getattr(smp, model_arch_name)
        except AttributeError as exc:
            raise ModelConfigurationError(f"Unsupported model architecture: {model_arch_name}") from exc

        self._torch = torch
        self._torch_cuda_version = getattr(torch.version, "cuda", None)
        self._device, self._device_reason = self._resolve_device(torch)
        self._device_name = self._resolve_device_name(torch, self._device)
        self._model = model_arch(
            encoder_name=hyperparams["pretrained_model_encoder"],
            encoder_weights=None,
            in_channels=hyperparams["input_channels"],
            classes=hyperparams["num_classes"],
        ).to(self._device)

        checkpoint = torch.load(str(self.model_path), map_location=self._device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self._model.load_state_dict(checkpoint)
        self._model.eval()
        return self._model

    def _resolve_device(self, torch: Any) -> tuple[Any, str]:
        requested = self.device_preference
        if requested in {"gpu", "cuda"}:
            requested = "cuda"

        if requested == "auto":
            if getattr(torch.version, "cuda", None) is None:
                return torch.device("cpu"), "auto-selected cpu because the installed PyTorch build is CPU-only"
            if not torch.cuda.is_available():
                return torch.device("cpu"), "auto-selected cpu because PyTorch cannot access a CUDA device"
            return torch.device("cuda"), "auto-selected cuda because a CUDA-capable GPU is available"

        if requested == "cpu":
            return torch.device("cpu"), "forced to cpu by INFERENCE_DEVICE"

        try:
            device = torch.device(requested)
        except RuntimeError as exc:
            raise ModelConfigurationError(
                f"Unsupported INFERENCE_DEVICE '{self.device_preference}'. Use auto, cpu, cuda, or cuda:<index>."
            ) from exc

        if device.type != "cuda":
            raise ModelConfigurationError(
                f"Unsupported INFERENCE_DEVICE '{self.device_preference}'. Use auto, cpu, cuda, or cuda:<index>."
            )

        if getattr(torch.version, "cuda", None) is None:
            raise InferenceDependencyError(
                "INFERENCE_DEVICE requests CUDA, but the installed PyTorch build is CPU-only. "
                "Install a CUDA-enabled PyTorch build in the backend environment."
            )

        if not torch.cuda.is_available():
            raise InferenceDependencyError(
                "INFERENCE_DEVICE requests CUDA, but PyTorch cannot access an NVIDIA GPU in the current environment."
            )

        device_index = device.index if device.index is not None else torch.cuda.current_device()
        if device_index >= torch.cuda.device_count():
            raise InferenceDependencyError(
                f"INFERENCE_DEVICE requests {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )

        return device, f"forced to {device} by INFERENCE_DEVICE"

    @staticmethod
    def _resolve_device_name(torch: Any, device: Any) -> str | None:
        if device.type != "cuda":
            return None

        device_index = device.index if device.index is not None else torch.cuda.current_device()
        return str(torch.cuda.get_device_name(device_index))

    def _predict_patches(self, image: Any, patch_size: int, model: Any, np: Any) -> tuple[list[Any], int]:
        torch = self._torch
        assert torch is not None

        patch_masks: list[Any] = []
        rows = image.shape[0] // patch_size
        cols = image.shape[1] // patch_size

        for row in range(rows):
            for col in range(cols):
                patch = image[
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ]
                patch_rgb = patch[:, :, ::-1].copy()
                patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                patch_tensor = patch_tensor.unsqueeze(0).to(self._device)

                with torch.no_grad():
                    prediction = model(patch_tensor)
                    prediction = torch.sigmoid(prediction)
                    prediction = prediction.detach().cpu().numpy()

                binary_mask = np.where(prediction >= self.threshold, 1, 0).astype(np.uint8)
                patch_masks.append(binary_mask.squeeze())

        return patch_masks, len(patch_masks)

    @staticmethod
    def _pad_to_multiple(image: Any, patch_size: int, cv2: Any) -> tuple[Any, tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        padded_height = ((height + patch_size - 1) // patch_size) * patch_size
        padded_width = ((width + patch_size - 1) // patch_size) * patch_size

        top = (padded_height - height) // 2
        bottom = padded_height - height - top
        left = (padded_width - width) // 2
        right = padded_width - width - left

        padded_image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        return padded_image, (top, bottom, left, right)

    @staticmethod
    def _stitch_masks(patch_masks: list[Any], padded_shape: tuple[int, int], patch_size: int, np: Any) -> Any:
        rows = padded_shape[0] // patch_size
        cols = padded_shape[1] // patch_size
        stitched = np.zeros((rows * patch_size, cols * patch_size), dtype=np.uint8)

        index = 0
        for row in range(rows):
            for col in range(cols):
                stitched[
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ] = patch_masks[index]
                index += 1
        return stitched

    @staticmethod
    def _crop_padding(mask: Any, padding: tuple[int, int, int, int], height: int, width: int) -> Any:
        top, _, left, _ = padding
        return mask[top : top + height, left : left + width]

    @staticmethod
    def _build_overlay(image: Any, mask_uint8: Any, cv2: Any, np: Any) -> Any:
        overlay = image.copy()
        red_mask = np.zeros_like(image)
        red_mask[:, :, 2] = mask_uint8
        return cv2.addWeighted(overlay, 1.0, red_mask, 0.45, 0)

    @staticmethod
    def _import_dependency(module_name: str, package_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise InferenceDependencyError(
                f"Missing runtime dependency '{package_name}'. Install the backend requirements first."
            ) from exc

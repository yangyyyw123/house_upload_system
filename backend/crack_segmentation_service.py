from __future__ import annotations

try:
    from .services.crack_segmentation_service import (
        CrackSegmentationService,
        InferenceDependencyError,
        ModelConfigurationError,
    )
except ImportError:
    from services.crack_segmentation_service import (
        CrackSegmentationService,
        InferenceDependencyError,
        ModelConfigurationError,
    )

__all__ = [
    "CrackSegmentationService",
    "InferenceDependencyError",
    "ModelConfigurationError",
]

from __future__ import annotations

try:
    from .services.target_generator_service import TARGET_SPECS, TargetGenerationError, TargetGeneratorService
except ImportError:
    from services.target_generator_service import TARGET_SPECS, TargetGenerationError, TargetGeneratorService

__all__ = [
    "TARGET_SPECS",
    "TargetGenerationError",
    "TargetGeneratorService",
]

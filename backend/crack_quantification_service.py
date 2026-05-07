from __future__ import annotations

try:
    from .services.crack_quantification_service import CrackQuantificationService, QuantificationError
except ImportError:
    from services.crack_quantification_service import CrackQuantificationService, QuantificationError

__all__ = [
    "CrackQuantificationService",
    "QuantificationError",
]

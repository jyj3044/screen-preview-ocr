"""화면 감지: 키워드 OCR + 템플릿 매칭."""

from .common import DetectionConfig, OCR_VARIANT_GROUPS_DISABLED, OverlayRect
from .keywords import OCR_VARIANT_UI_CHOICES, check_plain_text, ocr_runtime_ok
from .ocr_backends import ALL_OCR_ENGINES, DEFAULT_OCR_ENGINE, normalize_ocr_engine
from .overlay_store import get_overlay_store
from .pipeline import run_detection, run_detection_with_overlays

__all__ = [
    "ALL_OCR_ENGINES",
    "DEFAULT_OCR_ENGINE",
    "DetectionConfig",
    "get_overlay_store",
    "OCR_VARIANT_GROUPS_DISABLED",
    "OCR_VARIANT_UI_CHOICES",
    "OverlayRect",
    "check_plain_text",
    "normalize_ocr_engine",
    "ocr_runtime_ok",
    "run_detection",
    "run_detection_with_overlays",
]

"""
GUI 시작 전·캡처 공통 진입점: OS 별 초기화, 창 캡처 가능 여부, 창 목록·캡처 팩토리.
"""

from __future__ import annotations

import sys
from typing import List

from app_platform.frame_source import BgrWindowCapture
from app_platform.models import WindowEntry


def window_pick_supported() -> bool:
    """창 선택·창 단위 캡처 UI/로직을 켤 수 있는지 (Windows·macOS)."""
    return sys.platform in ("win32", "darwin")


def ensure_pre_gui_init() -> None:
    """Tk() 전에 호출: DPI 등 OS 별 준비."""
    if sys.platform == "win32":
        from windows_capture import ensure_windows_dpi_awareness

        ensure_windows_dpi_awareness()


def enumerate_windows(min_width: int = 80, min_height: int = 80) -> List[WindowEntry]:
    if sys.platform == "win32":
        from windows_capture import enumerate_windows as _ew

        return _ew(min_width=min_width, min_height=min_height)
    if sys.platform == "darwin":
        from darwin_capture import enumerate_windows as _ew

        return _ew(min_width=min_width, min_height=min_height)
    return []


def make_window_capture(window_id: int) -> BgrWindowCapture:
    """
    WindowEntry.hwnd 와 동일한 식별자로 캡처 객체를 만듭니다.
    grab_bgr() · close() 를 제공합니다.
    """
    if sys.platform == "win32":
        from windows_capture import WindowCapture

        return WindowCapture(window_id)
    if sys.platform == "darwin":
        from darwin_capture import MacWindowCapture

        return MacWindowCapture(window_id)
    raise OSError("창 캡처는 이 운영체제에서 지원되지 않습니다.")

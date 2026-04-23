"""플랫폼 공통: 창 목록 한 줄 표현."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowEntry:
    """
    hwnd: Windows 에서는 HWND, macOS 에서는 CGWindowID.
    게임 창 선택·캡처 식별용 불투명 정수입니다.
    """

    hwnd: int
    title: str
    process_name: str

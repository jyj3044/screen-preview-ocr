"""알림 오버레이 싱글톤: 여러 스레드에서 touch 가능, 위치별 마지막 갱신 후 TTL 초과 시 제거."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from .common import OverlayRect

# 동일 (x,y,w,h)에 대해 이 시간 동안 touch가 없으면 제거
OVERLAY_TTL_SEC = 2.0


class OverlayStore:
    """전역 하나. touch / snapshot / clear 는 스레드 안전."""

    _singleton: Optional["OverlayStore"] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[
            Tuple[int, int, int, int], Tuple[OverlayRect, float]
        ] = {}

    @classmethod
    def instance(cls) -> "OverlayStore":
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def touch(self, rects: List[OverlayRect]) -> None:
        """감지 결과로 박스를 갱신한다. 같은 위치는 최신 색·라벨로 덮어쓴다."""
        now = time.monotonic()
        with self._lock:
            for ov in rects:
                key = (ov.x, ov.y, ov.w, ov.h)
                self._entries[key] = (
                    OverlayRect(
                        ov.x, ov.y, ov.w, ov.h, ov.color_bgr, ov.label
                    ),
                    now,
                )

    def snapshot(self) -> List[OverlayRect]:
        """TTL 지난 항목을 제거한 뒤 현재 유효한 박스 목록(복사본)."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - OVERLAY_TTL_SEC
            dead = [k for k, (_, ts) in self._entries.items() if ts < cutoff]
            for k in dead:
                del self._entries[k]
            return [ov for ov, _ in self._entries.values()]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def get_overlay_store() -> OverlayStore:
    return OverlayStore.instance()

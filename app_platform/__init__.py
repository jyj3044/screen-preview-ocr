"""OS별 창·오디오 추상화 (표준 라이브러리 `platform` 과 이름 충돌 방지)."""

from app_platform.audio import play_alert_sound, stop_queued_alert_sounds
from app_platform.host import (
    ensure_pre_gui_init,
    enumerate_windows,
    make_window_capture,
    window_pick_supported,
)

__all__ = [
    "ensure_pre_gui_init",
    "enumerate_windows",
    "make_window_capture",
    "window_pick_supported",
    "play_alert_sound",
    "stop_queued_alert_sounds",
]

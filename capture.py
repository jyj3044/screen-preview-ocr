"""
1단계: 화면을 파이썬으로 캡처(송출에 해당하는 프레임 획득).

- 실제로는 “인터넷 브라우저가 원격 스트림을 받는 것”과는 다릅니다.
  네트워크로 게임 영상을 받는 구조가 아니라, 본인 PC에서 **이미 모니터에
  출력된 화면**만 OS가 제공하는 캡처 경로로 읽습니다. (OBS·Discord 화면 공유와
  같은 계열: 디스플레이/데스크톱 픽셀만 접근.)
- **게임 프로세스 메모리를 읽거나 쓰지 않습니다.** DLL 주입·후킹·pymem 등도
  사용하지 않습니다. 안티치트가 주로 겨냥하는 “메모리/코드 무결성 위반”과는
  다른 방식입니다. (다만 게임·안티치트 정책은 제품마다 다를 수 있음.)

실시간 GUI와 감지 로직은 이 프레임을 공유하면 됩니다.
"""

from __future__ import annotations

import threading
import time
import sys
from typing import Callable, Optional, Union

import mss
import numpy as np


class ScreenCapture:
    """단일 모니터 화면을 BGR uint8 numpy 배열로 캡처."""

    def __init__(self, monitor_index: int = 1):
        """
        monitor_index: mss 규칙 — 0은 모든 모니터 합친 가상 화면, 1부터 각 모니터.
        """
        self.monitor_index = monitor_index
        self._sct: Optional[object] = None

    def _mss(self):
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def grab_bgr(self) -> np.ndarray:
        """현재 프레임을 BGR (H, W, 3) 로 반환."""
        sct = self._mss()
        mon = sct.monitors[self.monitor_index]
        shot = sct.grab(mon)
        # BGRA -> BGR
        frame = np.asarray(shot, dtype=np.uint8)
        return frame[:, :, :3].copy()

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


def _make_capture(
    *,
    monitor_index: int,
    window_hwnd: Optional[int],
) -> Union[ScreenCapture, object]:
    if window_hwnd is not None:
        if sys.platform != "win32":
            raise OSError("창 캡처는 Windows에서만 지원됩니다.")
        from windows_capture import WindowCapture

        return WindowCapture(window_hwnd)
    return ScreenCapture(monitor_index=monitor_index)


class CaptureThread(threading.Thread):
    """
    백그라운드에서 주기적으로 캡처하여 최신 프레임만 보관.
    GUI/감지 스레드는 get_frame() 으로 읽습니다.
    """

    def __init__(
        self,
        monitor_index: int = 1,
        target_fps: float = 30.0,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        window_hwnd: Optional[int] = None,
    ):
        super().__init__(daemon=True)
        self._monitor_index = monitor_index
        self._window_hwnd = window_hwnd
        self._interval = 1.0 / max(target_fps, 1.0)
        self._on_frame = on_frame
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        self._running.set()
        cap = _make_capture(
            monitor_index=self._monitor_index, window_hwnd=self._window_hwnd
        )
        try:
            while self._running.is_set():
                t0 = time.perf_counter()
                try:
                    frame = cap.grab_bgr()
                    with self._lock:
                        self._latest = frame
                    if self._on_frame:
                        self._on_frame(frame)
                except Exception:
                    pass
                elapsed = time.perf_counter() - t0
                sleep_for = self._interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            cap.close()


if __name__ == "__main__":
    # 단독 실행: 한 장 캡처 후 저장 (동작 확인용)
    cap = ScreenCapture(monitor_index=1)
    img = cap.grab_bgr()
    cap.close()
    import cv2

    cv2.imwrite("capture_test.png", img)
    print("저장됨: capture_test.png", img.shape)

"""창 단위 캡처 객체가 따르는 공통 형태 (정적 검사용)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BgrWindowCapture(Protocol):
    def grab_bgr(self) -> np.ndarray:
        ...

    def close(self) -> None:
        ...

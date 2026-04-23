"""
macOS (Quartz): 화면에 올라온 창 나열 + CGWindowID 기준 창 픽셀 캡처.

화면 녹화 권한이 없으면 CGWindowListCreateImage 가 실패할 수 있습니다.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any, List, Optional

import cv2
import numpy as np

from app_platform.models import WindowEntry

if sys.platform != "darwin":
    raise ImportError("darwin_capture는 macOS에서만 사용할 수 있습니다.")

try:
    import Quartz  # type: ignore
except ImportError as e:
    raise ImportError(
        "macOS 창 캡처에는 pyobjc-framework-Quartz 가 필요합니다. "
        "`pip install pyobjc-framework-Quartz` 후 다시 실행하세요."
    ) from e


def _cg_field(window_dict: Any, name: str) -> Any:
    const = getattr(Quartz, name, None)
    if const is not None:
        try:
            if const in window_dict:
                return window_dict[const]
        except Exception:
            pass
        try:
            return window_dict.get(const)
        except Exception:
            pass
    try:
        return window_dict.get(name)
    except Exception:
        return None


def _bounds_width_height(bounds: Any) -> tuple[int, int]:
    if bounds is None:
        return 0, 0
    if isinstance(bounds, dict):
        for kw, kh in (("Width", "Height"), ("width", "height")):
            if kw in bounds and kh in bounds:
                return int(bounds[kw]), int(bounds[kh])
    try:
        return int(bounds.size.width), int(bounds.size.height)  # NSRect/CGRect
    except Exception:
        return 0, 0


def _cgimage_to_bgr(cg_image) -> np.ndarray:
    w = int(Quartz.CGImageGetWidth(cg_image))
    h = int(Quartz.CGImageGetHeight(cg_image))
    if w <= 0 or h <= 0:
        raise OSError("캡처 크기가 0입니다.")
    color_space = Quartz.CGColorSpaceCreateDeviceRGB()
    if not color_space:
        raise OSError("CGColorSpaceCreateDeviceRGB 실패")
    bytes_per_row = w * 4
    nbytes = bytes_per_row * h
    buf = (ctypes.c_ubyte * nbytes)()
    ctx = Quartz.CGBitmapContextCreate(
        buf,
        w,
        h,
        8,
        bytes_per_row,
        color_space,
        Quartz.kCGImageAlphaPremultipliedLast | Quartz.kCGBitmapByteOrder32Big,
    )
    try:
        if not ctx:
            raise OSError("CGBitmapContextCreate 실패")
        Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, w, h), cg_image)
    finally:
        if ctx:
            Quartz.CGContextRelease(ctx)
        Quartz.CGColorSpaceRelease(color_space)

    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def enumerate_windows(min_width: int = 80, min_height: int = 80) -> List[WindowEntry]:
    raw = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
    )
    out: List[WindowEntry] = []
    for win in raw:
        layer = _cg_field(win, "kCGWindowLayer")
        if layer is None or int(layer) != 0:
            continue
        title_obj = _cg_field(win, "kCGWindowName")
        title = (str(title_obj).strip() if title_obj else "") or ""
        if not title:
            continue
        bounds = _cg_field(win, "kCGWindowBounds")
        bw, bh = _bounds_width_height(bounds)
        if bw < min_width or bh < min_height:
            continue
        wid_obj = _cg_field(win, "kCGWindowNumber")
        if wid_obj is None:
            continue
        wid = int(wid_obj)
        owner = _cg_field(win, "kCGWindowOwnerName")
        process_name = (
            str(owner).strip()
            if owner
            else f"pid:{_cg_field(win, 'kCGWindowOwnerPID') or '?'}"
        )
        on_screen = _cg_field(win, "kCGWindowIsOnScreen")
        if on_screen is not None and not bool(on_screen):
            continue
        out.append(WindowEntry(hwnd=wid, title=title, process_name=process_name))
    out.sort(key=lambda e: (e.process_name.lower(), e.title.lower()))
    return out


class MacWindowCapture:
    """CGWindowID 기준 단일 창 이미지 (BGR)."""

    def __init__(self, window_id: int):
        self._wid = int(window_id)

    def grab_bgr(self) -> np.ndarray:
        img_opts = Quartz.kCGWindowImageDefault
        if hasattr(Quartz, "kCGWindowImageNominalResolution"):
            img_opts |= Quartz.kCGWindowImageNominalResolution
        img = Quartz.CGWindowListCreateImage(
            Quartz.CGRectInfinite,
            Quartz.kCGWindowListOptionIncludingWindow,
            self._wid,
            img_opts,
        )
        if img is None:
            raise OSError(
                "창 이미지를 가져오지 못했습니다. "
                "시스템 설정 > 개인 정보 보호 및 보안 > 화면 기록에서 "
                "Python(또는 터미널·이 앱)을 허용했는지 확인하세요."
            )
        try:
            return _cgimage_to_bgr(img)
        finally:
            release = getattr(Quartz, "CGImageRelease", None)
            if callable(release):
                try:
                    release(img)
                except Exception:
                    pass

    def close(self) -> None:
        pass

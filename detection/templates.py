"""템플릿(부분 이미지) 다중 매칭."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .common import OverlayRect, stable_overlay_bgr

# 약간의 스케일·압축 차이를 흡수 (캡처와 템플릿 파일 해상도 불일치 완화)
_TEMPLATE_SCALE_FACTORS: Tuple[float, ...] = (0.92, 0.96, 1.0, 1.04, 1.08)


def load_template_bgr(path: str) -> Optional[np.ndarray]:
    """BGR uint8. PNG 알파는 흰 배경에 합성."""
    if not path or not os.path.isfile(path):
        return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.size == 0:
        return None
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 3:
        return np.ascontiguousarray(im)
    if im.shape[2] == 4:
        bgr = im[:, :, :3].astype(np.float32)
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        out = bgr * a + white * (1.0 - a)
        return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))
    return None


def _match_at_scale(
    frame_bgr: np.ndarray, tpl_bgr: np.ndarray, threshold: float
) -> Optional[Tuple[int, int, int, int, float]]:
    fh, fw = frame_bgr.shape[:2]
    th0, tw0 = tpl_bgr.shape[:2]
    if th0 < 2 or tw0 < 2:
        return None
    if th0 > fh or tw0 > fw:
        return None
    frame_bgr = np.ascontiguousarray(frame_bgr)
    tpl_bgr = np.ascontiguousarray(tpl_bgr)
    try:
        res = cv2.matchTemplate(frame_bgr, tpl_bgr, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(res)
    if max_v < threshold:
        return None
    return (int(max_loc[0]), int(max_loc[1]), tw0, th0, float(max_v))


def match_one_template(
    frame_bgr: np.ndarray, template_path: str, threshold: float
) -> Optional[Tuple[OverlayRect, float]]:
    tpl0 = load_template_bgr(template_path)
    if tpl0 is None:
        return None
    best: Optional[Tuple[int, int, int, int, float]] = None
    best_sc = threshold
    for sf in _TEMPLATE_SCALE_FACTORS:
        tw = max(2, int(round(tpl0.shape[1] * sf)))
        th = max(2, int(round(tpl0.shape[0] * sf)))
        if tw != tpl0.shape[1] or th != tpl0.shape[0]:
            tpl = cv2.resize(
                tpl0,
                (tw, th),
                interpolation=cv2.INTER_AREA if sf < 1.0 else cv2.INTER_LINEAR,
            )
        else:
            tpl = tpl0
        hit = _match_at_scale(frame_bgr, tpl, threshold)
        if hit is not None and hit[4] > best_sc:
            best = hit
            best_sc = hit[4]
    if best is None:
        return None
    x, y, tw, th, sc = best
    base = os.path.basename(template_path)
    label = f"템플릿:{base}"
    ov = OverlayRect(
        x, y, tw, th, stable_overlay_bgr("tpl", x, y, tw, th), label
    )
    return ov, sc


def match_all_templates(
    frame_bgr: np.ndarray,
    template_paths: Tuple[str, ...],
    threshold: float,
) -> List[OverlayRect]:
    out: List[OverlayRect] = []
    seen_rect: set[Tuple[int, int, int, int]] = set()
    for p in template_paths:
        p = (p or "").strip()
        if not p:
            continue
        hit = match_one_template(frame_bgr, p, threshold)
        if hit is None:
            continue
        ov, _sc = hit
        key = (ov.x, ov.y, ov.w, ov.h)
        if key in seen_rect:
            continue
        seen_rect.add(key)
        out.append(ov)
    return out


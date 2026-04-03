"""알림 키워드 OCR (프레임에서 텍스트·단어 박스 감지)."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, FrozenSet, List, Optional, Tuple

import cv2
import numpy as np

from .common import OCR_VARIANT_GROUPS_DISABLED, OverlayRect, stable_overlay_bgr
from .ocr_backends import (
    ALL_OCR_ENGINES,
    DEFAULT_OCR_ENGINE,
    ENGINE_TESSERACT,
    boxes_from_tesseract_data,
    joined_text_from_rgb,
    normalize_ocr_engine,
    ocr_engine_runtime_ok,
    ocr_word_boxes,
    tesseract_call_image_to_string,
    tesseract_image_to_data,
)

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore

_OCR_LANG_SEQUENCE: Tuple[str, ...] = ("kor+eng", "kor", "eng+kor", "eng")
# 게임 UI: 블록 / 희소 텍스트 / 자동 / 단일 열 순으로 시도
_OCR_PSM_CONFIGS: Tuple[str, ...] = (
    "--oem 3 --psm 6",
    "--oem 3 --psm 11",
    "--oem 3 --psm 3",
    "--oem 3 --psm 4",
)
_OCR_PSM_FAST: Tuple[str, ...] = ("--oem 3 --psm 6", "--oem 3 --psm 11")
_OCR_LANG_FAST: Tuple[str, ...] = ("kor+eng", "eng")
_OCR_CONFIG = _OCR_PSM_CONFIGS[0]
_OCR_MAX_SIDE = 1600
_OCR_STRING_SCORE_OK = 20
_OCR_BOX_COUNT_OK = 10
# Python 3.9+ 에서만 미시작 Future 취소
_OCR_POOL_CANCEL = sys.version_info >= (3, 9)

# PyInstaller exe: Tesseract 는 호출마다 subprocess → 변형×PSM 이 많으면 UI 프리즈 수준
_TESSERACT_FROZEN_MAX_VARIANTS = 5


def _tesseract_exe_light_mode() -> bool:
    return getattr(sys, "frozen", False)


def _cap_tesseract_variants(
    variants: List[Tuple[np.ndarray, int, int, str]],
) -> List[Tuple[np.ndarray, int, int, str]]:
    if not _tesseract_exe_light_mode() or len(variants) <= _TESSERACT_FROZEN_MAX_VARIANTS:
        return variants
    return variants[:_TESSERACT_FROZEN_MAX_VARIANTS]


def _tesseract_overlay_psms() -> Tuple[str, ...]:
    if _tesseract_exe_light_mode():
        return ("--oem 3 --psm 6", "--oem 3 --psm 11")
    return ("--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 3")


# UI·설정: 비어 있으면 전부 사용. id → 로그 라벨 그룹 매핑은 _VARIANT_LOG_TO_GROUP
OCR_VARIANT_UI_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("raw", "원본 (리사이즈만, CLAHE·이진 없음)"),
    ("invert", "전체 색 반전 (BGR)"),
    ("gray_clahe", "CLAHE 그레이 (기본)"),
    ("bilateral_clahe", "양방향 필터 + CLAHE"),
    ("unsharp", "언샤프 선명화"),
    ("adaptive", "적응형 이진 (2종)"),
    ("otsu", "오츠 이진 (2종)"),
    ("tophat", "탑햇+이진"),
    ("scale2x_clahe", "2배 확대 + CLAHE"),
)

_VARIANT_LOG_TO_GROUP: Dict[str, str] = {
    "raw_bgr": "raw",
    "bgr_invert": "invert",
    "gray_clahe": "gray_clahe",
    "bilateral_clahe": "bilateral_clahe",
    "unsharp": "unsharp",
    "adaptive_bin": "adaptive",
    "adaptive_bin_inv": "adaptive",
    "otsu": "otsu",
    "otsu_inv": "otsu",
    "tophat_bin": "tophat",
    "scale2x_clahe": "scale2x_clahe",
}


def _enabled_variant_frozen(
    variant_groups: Tuple[str, ...],
) -> Optional[FrozenSet[str]]:
    if not variant_groups:
        return None
    if variant_groups == OCR_VARIANT_GROUPS_DISABLED:
        return frozenset()
    return frozenset(variant_groups)


def _include_variant_log_label(
    log_label: str, enabled: Optional[FrozenSet[str]]
) -> bool:
    if enabled is None:
        return True
    g = _VARIANT_LOG_TO_GROUP.get(log_label, log_label)
    return g in enabled


# run_keyword_detection 진입 시만 비-None 으로 두며, UI에서 엔진·전처리 변경 시 set 해 진행 중 OCR 을 끊는다.
_kw_abort_token: Optional[threading.Event] = None


def _stop_req(ev: Optional[threading.Event]) -> bool:
    if ev is not None and ev.is_set():
        return True
    t = _kw_abort_token
    return t is not None and t.is_set()


def _coerce_ocr_engines(engines: Tuple[str, ...]) -> Tuple[str, ...]:
    """정규화·중복 제거(첫 순서 유지)."""
    seen: set[str] = set()
    out: List[str] = []
    for e in engines:
        n = normalize_ocr_engine(e)
        if not n or n not in ALL_OCR_ENGINES:
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def _variant_parallel_workers(n: int) -> int:
    try:
        w = int(os.environ.get("MAPLEALERT_OCR_PARALLEL_WORKERS", "4"))
    except ValueError:
        w = 4
    return max(1, min(w, max(1, n)))


def _pool_shutdown_cancel(ex: ThreadPoolExecutor) -> None:
    if _OCR_POOL_CANCEL:
        ex.shutdown(wait=False, cancel_futures=True)
    else:
        ex.shutdown(wait=False)


def _neural_joined_parallel_run_pass(
    rgb_list: List[Tuple[np.ndarray, int, int, str]],
    eng: str,
    seek_keywords: Optional[Tuple[str, ...]],
    stop_event: Optional[threading.Event],
    log_alert_keywords: Tuple[str, ...] = (),
) -> Tuple[str, int, List[str]]:
    """
    변형별 joined_text를 병렬 수집. 한 변형 문자열에 키워드가 있으면 미완료 작업 취소.
    반환: (best, score, frags_vi_order) — frags 는 vi 오름차순으로 비어 있지 않은 텍스트.
    """
    _KW_FOUND = 10**9
    n = len(rgb_list)
    if n == 0:
        return "", 0, []

    def work(vi: int) -> Tuple[int, str]:
        if _stop_req(stop_event):
            return vi, ""
        rgb, _, _, lab = rgb_list[vi]
        return vi, joined_text_from_rgb(
            rgb,
            eng,
            preprocess_label=lab,
            alert_keywords=log_alert_keywords if log_alert_keywords else None,
        )

    mw = _variant_parallel_workers(n)
    out: Dict[int, str] = {}
    ex = ThreadPoolExecutor(max_workers=mw)
    futs = [ex.submit(work, vi) for vi in range(n)]
    early_kw: Optional[str] = None
    try:
        for fut in as_completed(futs):
            if _stop_req(stop_event):
                break
            vi, s = fut.result()
            out[vi] = s
            if seek_keywords is not None:
                t = (s or "").strip()
                if t and any(k in t.lower() for k in seek_keywords):
                    early_kw = t
                    break
    finally:
        if early_kw is not None or _stop_req(stop_event):
            _pool_shutdown_cancel(ex)
        else:
            ex.shutdown(wait=True)

    if early_kw is not None:
        frags_e: List[str] = []
        for vi in range(n):
            tt = (out.get(vi) or "").strip()
            if tt:
                frags_e.append(tt)
        return early_kw, _KW_FOUND, frags_e

    frags: List[str] = []
    for vi in range(n):
        t = (out.get(vi) or "").strip()
        if t:
            frags.append(t)

    if seek_keywords is not None:
        blob = " ".join(frags).lower()
        if frags and any(k in blob for k in seek_keywords):
            return " ".join(frags), _KW_FOUND, frags

    best, best_sc = "", 0
    for vi in range(n):
        s = out.get(vi) or ""
        if not s:
            continue
        sc = _ocr_text_score(s)
        if sc > best_sc:
            best_sc = sc
            best = s
            if best_sc >= _OCR_STRING_SCORE_OK and _hangul_count(s) >= 3:
                if seek_keywords is None:
                    return best, best_sc, frags
                if any(k in best.lower() for k in seek_keywords):
                    return best, _KW_FOUND, frags
    return best, best_sc, frags


def ocr_runtime_ok(engine: str = DEFAULT_OCR_ENGINE) -> Tuple[bool, str]:
    return ocr_engine_runtime_ok(normalize_ocr_engine(engine))


def _odd_k(n: int, lo: int = 3, hi: int = 51) -> int:
    n = max(lo, min(hi, int(n)))
    return n if n % 2 == 1 else n + 1


def _work_bgr(frame_bgr: np.ndarray, max_side: int = _OCR_MAX_SIDE) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame_bgr
    s = max_side / m
    return cv2.resize(
        frame_bgr,
        (max(1, int(w * s)), max(1, int(h * s))),
        interpolation=cv2.INTER_AREA,
    )


def _prepare_bgr_for_ocr(frame_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """기존 호환: 축소 + CLAHE 그레이 → RGB."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    work = _work_bgr(frame_bgr, max_side=max_side)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)


def _iter_game_ocr_variants_rgb_with_scale(
    frame_bgr: np.ndarray,
    *,
    variant_groups: Tuple[str, ...] = (),
) -> List[Tuple[np.ndarray, int, int, str]]:
    """
    배경·글자색이 바뀌어도 글자 에지/대비를 살리는 여러 RGB 입력.
    반환: (rgb, ws, hs, log_label) — log_label 은 OCR 로그·설정 필터용.
    Tesseract·뉴럴 OCR 모두 동일 변형 집합( variant_groups 필터 적용).

    variant_groups 비어 있으면 전 변형 사용. OCR_VARIANT_GROUPS_DISABLED 이면 빈 목록.
    그 외에는 OCR_VARIANT_UI_CHOICES id 만 포함.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    enabled = _enabled_variant_frozen(variant_groups)
    h0, w0 = frame_bgr.shape[:2]
    work = _work_bgr(frame_bgr, _OCR_MAX_SIDE)
    hw, ww = work.shape[:2]
    out: List[Tuple[np.ndarray, int, int, str]] = []

    def push_l(rgb: np.ndarray, log_label: str) -> None:
        if not _include_variant_log_label(log_label, enabled):
            return
        hs, ws = rgb.shape[:2]
        if ws > 0 and hs > 0:
            out.append((rgb, ws, hs, log_label))

    # 캡처 BGR을 긴 변 _OCR_MAX_SIDE 기준으로만 줄인 뒤 RGB (대비·이진 없음)
    push_l(cv2.cvtColor(work, cv2.COLOR_BGR2RGB), "raw_bgr")

    try:
        push_l(cv2.cvtColor(cv2.bitwise_not(work), cv2.COLOR_BGR2RGB), "bgr_invert")
    except cv2.error:
        pass

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # 그레이·적응형 전처리를 먼저 넣는다.
    push_l(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), "gray_clahe")

    # 경계 보존 + 배경 스무딩 (번쩍이는 배경 완화)
    try:
        bil = cv2.bilateralFilter(work, 7, 50, 50)
        g2 = clahe.apply(cv2.cvtColor(bil, cv2.COLOR_BGR2GRAY))
        push_l(cv2.cvtColor(g2, cv2.COLOR_GRAY2RGB), "bilateral_clahe")
    except cv2.error:
        pass

    # 언샤프 마스크 (픽셀 글자 선명화)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    sharp = cv2.addWeighted(g, 1.35, blur, -0.35, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    push_l(cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB), "unsharp")

    blk = _odd_k(min(ww, hw) // 28, 11, 45)
    try:
        at = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 6
        )
        push_l(cv2.cvtColor(at, cv2.COLOR_GRAY2RGB), "adaptive_bin")
        ati = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 6
        )
        push_l(cv2.cvtColor(ati, cv2.COLOR_GRAY2RGB), "adaptive_bin_inv")
    except cv2.error:
        pass

    try:
        _, ot = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        push_l(cv2.cvtColor(ot, cv2.COLOR_GRAY2RGB), "otsu")
        _, oti = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        push_l(cv2.cvtColor(oti, cv2.COLOR_GRAY2RGB), "otsu_inv")
    except cv2.error:
        pass

    # 밝은 필라멘트/글자 강조 (배경이 어두운 구간에서 유효)
    ksz = _odd_k(min(ww, hw) // 22, 9, 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    thn = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    _, thb = cv2.threshold(thn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    push_l(cv2.cvtColor(thb, cv2.COLOR_GRAY2RGB), "tophat_bin")

    # 픽셀 폰트용 2배 NEAREST 후 CLAHE (작은 글자)
    up = cv2.resize(work, (ww * 2, hw * 2), interpolation=cv2.INTER_NEAREST)
    gu = clahe.apply(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY))
    push_l(cv2.cvtColor(gu, cv2.COLOR_GRAY2RGB), "scale2x_clahe")

    # 부분 선택일 때는 필터에 맞는 변형이 0개면 그대로 둠(gray_clahe 를 끼워 넣지 않음).
    if not out and enabled is None:
        hs, ws = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).shape[:2]
        if ws > 0 and hs > 0:
            out.append((cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), ws, hs, "gray_clahe"))
    return out


def _hangul_count(s: str) -> int:
    return sum(1 for c in s if "\uac00" <= c <= "\ud7a3")


def _ocr_text_score(s: str) -> int:
    if not s:
        return 0
    return _hangul_count(s) * 4 + sum(1 for c in s if c.isalnum())


def _ocr_string_from_frame_neural(
    frame_bgr: np.ndarray,
    seek_keywords: Optional[Tuple[str, ...]],
    ocr_engine: str,
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
    log_alert_keywords: Tuple[str, ...] = (),
) -> str:
    """EasyOCR / RapidOCR: 변형별 joined_text 병렬 수집·키워드 조기 종료."""
    eng = normalize_ocr_engine(ocr_engine)
    variants = _iter_game_ocr_variants_rgb_with_scale(
        frame_bgr, variant_groups=variant_groups
    )
    if not variants:
        return ""
    _KW_FOUND = 10**9
    frags: List[str] = []

    def run_pass(
        rgb_list: List[Tuple[np.ndarray, int, int, str]],
    ) -> Tuple[str, int]:
        nonlocal frags
        best, sc, fr = _neural_joined_parallel_run_pass(
            rgb_list,
            eng,
            seek_keywords,
            stop_event,
            log_alert_keywords=log_alert_keywords,
        )
        frags.extend(fr)
        return best, sc

    n_fast = min(8, len(variants))
    best, sc = run_pass(variants[:n_fast])

    if seek_keywords is not None:
        if sc >= _KW_FOUND:
            return best
        blob = " ".join(frags).lower()
        if any(k in blob for k in seek_keywords):
            return " ".join(frags)
        if _stop_req(stop_event):
            return " ".join(frags) if frags else best
        best2, sc2 = run_pass(variants)
        if sc2 >= _KW_FOUND:
            return best2
        blob2 = " ".join(frags).lower()
        if any(k in blob2 for k in seek_keywords):
            return " ".join(frags)
        return " ".join(frags) if frags else (best2 if sc2 > sc else best)

    if sc >= _OCR_STRING_SCORE_OK:
        return best
    if _stop_req(stop_event):
        return best
    best2, sc2 = run_pass(variants)
    return best2 if sc2 > sc else best


def _ocr_string_from_frame_bgr(
    frame_bgr: np.ndarray,
    seek_keywords: Optional[Tuple[str, ...]] = None,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
    log_alert_keywords: Tuple[str, ...] = (),
) -> str:
    """
    게임 UI용: 전처리 × (Tesseract: PSM×언어 / 뉴럴: 변형별 문자열) 조합.
    """
    eng = normalize_ocr_engine(ocr_engine)
    ok, _ = ocr_engine_runtime_ok(eng)
    if not ok:
        return ""
    if eng != ENGINE_TESSERACT:
        return _ocr_string_from_frame_neural(
            frame_bgr,
            seek_keywords,
            eng,
            stop_event,
            variant_groups=variant_groups,
            log_alert_keywords=log_alert_keywords,
        )

    if pytesseract is None:
        return ""
    variants = _cap_tesseract_variants(
        _iter_game_ocr_variants_rgb_with_scale(
            frame_bgr, variant_groups=variant_groups
        )
    )
    if not variants:
        return ""

    _KW_FOUND = 10**9
    frags: List[str] = []

    _log_kw = log_alert_keywords if log_alert_keywords else None

    def run_pass(
        rgb_list: List[Tuple[np.ndarray, int, int, str]],
        psms: Tuple[str, ...],
        langs: Tuple[str, ...],
    ) -> Tuple[str, int]:
        nonlocal frags
        best = ""
        best_sc = 0
        for rgb, _ws, _hs, lab in rgb_list:
            if _stop_req(stop_event):
                return best, best_sc
            for cfg in psms:
                for lang in langs:
                    try:
                        s = tesseract_call_image_to_string(
                            rgb,
                            lang=lang,
                            config=cfg,
                            preprocess_label=lab,
                            alert_keywords=_log_kw,
                        )
                        if not s:
                            continue
                        t = s.strip()
                        if t:
                            frags.append(t)
                            if seek_keywords:
                                if any(k in t.lower() for k in seek_keywords):
                                    return t, _KW_FOUND
                                joined = " ".join(frags)
                                if any(k in joined.lower() for k in seek_keywords):
                                    return joined, _KW_FOUND
                        sc = _ocr_text_score(s)
                        if sc > best_sc:
                            best_sc = sc
                            best = s
                            if (
                                best_sc >= _OCR_STRING_SCORE_OK
                                and _hangul_count(s) >= 3
                            ):
                                if seek_keywords is None:
                                    return best, best_sc
                                if any(k in best.lower() for k in seek_keywords):
                                    return best, _KW_FOUND
                    except Exception:
                        continue
                try:
                    s = tesseract_call_image_to_string(
                        rgb,
                        config=cfg,
                        preprocess_label=lab,
                        alert_keywords=_log_kw,
                    )
                    if s:
                        t = s.strip()
                        if t:
                            frags.append(t)
                            if seek_keywords:
                                if any(k in t.lower() for k in seek_keywords):
                                    return t, _KW_FOUND
                                joined = " ".join(frags)
                                if any(k in joined.lower() for k in seek_keywords):
                                    return joined, _KW_FOUND
                        sc = _ocr_text_score(s)
                        if sc > best_sc:
                            best_sc = sc
                            best = s
                            if (
                                best_sc >= _OCR_STRING_SCORE_OK
                                and _hangul_count(s) >= 3
                            ):
                                if seek_keywords is None:
                                    return best, best_sc
                                if any(k in best.lower() for k in seek_keywords):
                                    return best, _KW_FOUND
                except Exception:
                    continue
        return best, best_sc

    n_fast = min(8, len(variants))
    best, sc = run_pass(variants[:n_fast], _OCR_PSM_FAST, _OCR_LANG_FAST)

    if _tesseract_exe_light_mode():
        if seek_keywords is not None:
            if sc >= _KW_FOUND:
                return best
            blob = " ".join(frags).lower()
            if any(k in blob for k in seek_keywords):
                return " ".join(frags)
            return " ".join(frags) if frags else best
        if sc >= _OCR_STRING_SCORE_OK:
            return best
        return best

    if seek_keywords is not None:
        if sc >= _KW_FOUND:
            return best
        blob = " ".join(frags).lower()
        if any(k in blob for k in seek_keywords):
            return " ".join(frags)
        if _stop_req(stop_event):
            return " ".join(frags) if frags else best
        best2, sc2 = run_pass(variants, _OCR_PSM_CONFIGS, _OCR_LANG_SEQUENCE)
        if sc2 >= _KW_FOUND:
            return best2
        blob2 = " ".join(frags).lower()
        if any(k in blob2 for k in seek_keywords):
            return " ".join(frags)
        return " ".join(frags) if frags else (best2 if sc2 > sc else best)

    if sc >= _OCR_STRING_SCORE_OK:
        return best
    if _stop_req(stop_event):
        return best
    best2, sc2 = run_pass(variants, _OCR_PSM_CONFIGS, _OCR_LANG_SEQUENCE)
    return best2 if sc2 > sc else best


def _ocr_boxes_best_from_frame(
    frame_bgr: np.ndarray,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
    log_alert_keywords: Tuple[str, ...] = (),
) -> Tuple[List[Tuple[str, Tuple[int, int, int, int]]], int, int]:
    """
    전처리·(Tesseract: PSM) / 뉴럴: 단일 패스) 중 단어 박스가 가장 많은 조합.
    """
    eng = normalize_ocr_engine(ocr_engine)
    variants = _iter_game_ocr_variants_rgb_with_scale(
        frame_bgr,
        variant_groups=variant_groups,
    )
    if eng == ENGINE_TESSERACT:
        variants = _cap_tesseract_variants(variants)
    _log_kw = log_alert_keywords if log_alert_keywords else None
    best: List[Tuple[str, Tuple[int, int, int, int]]] = []
    best_ws, best_hs = 1, 1
    best_key: Tuple[int, int, int] = (-1, 0, 0)

    def consider(
        boxes: List[Tuple[str, Tuple[int, int, int, int]]],
        ws: int,
        hs: int,
        vi: int,
        pi: int,
    ) -> bool:
        nonlocal best, best_ws, best_hs, best_key
        n = len(boxes)
        key = (n, -vi, -pi)
        if key > best_key:
            best_key = key
            best = boxes
            best_ws, best_hs = ws, hs
        return n >= _OCR_BOX_COUNT_OK

    if eng != ENGINE_TESSERACT:
        max_v = len(variants)

        def box_job(vi: int) -> Tuple[int, int, int, List[Tuple[str, Tuple[int, int, int, int]]]]:
            if _stop_req(stop_event):
                return vi, 1, 1, []
            rgb, ws, hs, lab = variants[vi]
            return vi, ws, hs, ocr_word_boxes(
                rgb,
                eng,
                preprocess_label=lab,
                alert_keywords=_log_kw,
            )

        indices = list(range(max_v))
        mw = _variant_parallel_workers(len(indices))
        ex = ThreadPoolExecutor(max_workers=mw)
        futs = [ex.submit(box_job, vi) for vi in indices]
        shutdown_cancel = False
        try:
            for fut in as_completed(futs):
                if _stop_req(stop_event):
                    shutdown_cancel = True
                    break
                vi, ws, hs, boxes = fut.result()
                if consider(boxes, ws, hs, vi, 0):
                    shutdown_cancel = True
                    break
        finally:
            if shutdown_cancel:
                _pool_shutdown_cancel(ex)
            else:
                ex.shutdown(wait=True)
        if best_key[0] < 0:
            return [], 1, 1
        return best, best_ws, best_hs

    n_fast = min(5, len(variants))
    done_early = False
    for vi, (rgb, ws, hs, lab) in enumerate(variants[:n_fast]):
        if _stop_req(stop_event):
            break
        for pi, cfg in enumerate(_OCR_PSM_FAST):
            data = tesseract_image_to_data(
                rgb, cfg, preprocess_label=lab, alert_keywords=_log_kw
            )
            if not data:
                continue
            boxes = boxes_from_tesseract_data(data)
            if consider(boxes, ws, hs, vi, pi):
                done_early = True
                break
        if done_early:
            break
    if not done_early and not _tesseract_exe_light_mode():
        for vi, (rgb, ws, hs, lab) in enumerate(variants):
            if _stop_req(stop_event):
                break
            for pi, cfg in enumerate(_OCR_PSM_CONFIGS):
                data = tesseract_image_to_data(
                    rgb, cfg, preprocess_label=lab, alert_keywords=_log_kw
                )
                if not data:
                    continue
                boxes = boxes_from_tesseract_data(data)
                if consider(boxes, ws, hs, vi, pi):
                    done_early = True
                    break
            if done_early:
                break

    if best_key[0] < 0:
        return [], 1, 1
    return best, best_ws, best_hs


def _map_ocr_rect_to_frame(
    x: int,
    y: int,
    w: int,
    h: int,
    w0: int,
    h0: int,
    ws: int,
    hs: int,
) -> Tuple[int, int, int, int]:
    sx = w0 / float(ws)
    sy = h0 / float(hs)
    x2 = max(0, int(round(x * sx)))
    y2 = max(0, int(round(y * sy)))
    x3 = min(w0, int(round((x + w) * sx)))
    y3 = min(h0, int(round((y + h) * sy)))
    return x2, y2, max(0, x3 - x2), max(0, y3 - y2)


def _keyword_hit_word_boxes(
    words: List[Tuple[str, Tuple[int, int, int, int]]],
    kws: Tuple[str, ...],
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    키워드가 OCR 단어 목록에 있을 때, 박스를 그릴 단어만 고른다.
    (이전에는 blob 에 키워드만 있어도 전체 단어 min/max 로 거대한 사각형이 나오던 문제 방지)
    """
    if not words or not kws:
        return []
    parts = [(b[0] or "").lower() for b in words]
    blob = " ".join(parts)
    per_word = [b for b in words if any(k in (b[0] or "").lower() for k in kws)]
    if per_word:
        return per_word
    n = len(words)
    best: Optional[Tuple[int, int, int]] = None  # (span 단어 수, i, j)
    for k in kws:
        if not k or k not in blob:
            continue
        for i in range(n):
            joined = ""
            for j in range(i, n):
                piece = parts[j]
                joined = piece if not joined else f"{joined} {piece}"
                if k in joined:
                    span = j - i + 1
                    if best is None or span < best[0]:
                        best = (span, i, j)
                    break
    if best is not None:
        _, i, j = best
        return words[i : j + 1]
    return []


def _tesseract_row_level(data: Dict[str, List], i: int) -> int:
    try:
        return int(data["level"][i])
    except (KeyError, ValueError, TypeError, IndexError):
        return -1


def _overlay_keyword_text_fallback(
    frame_bgr: np.ndarray,
    keywords: Tuple[str, ...],
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
) -> List[OverlayRect]:
    """
    단어 박스에 키워드가 안 잡혀도(분절·스타일 폰트) 전체 OCR·줄·블록 단위로 박스 표시.
    """
    if not keywords:
        return []
    eng = normalize_ocr_engine(ocr_engine)
    ok, _ = ocr_engine_runtime_ok(eng)
    if not ok:
        return []
    kws = tuple((k or "").strip().lower() for k in keywords if (k or "").strip())
    if not kws:
        return []
    if _stop_req(stop_event):
        return []
    h0, w0 = frame_bgr.shape[:2]
    variant_list = list(
        _iter_game_ocr_variants_rgb_with_scale(
            frame_bgr,
            variant_groups=variant_groups,
        )
    )
    if eng == ENGINE_TESSERACT:
        variant_list = _cap_tesseract_variants(variant_list)
    if not variant_list:
        return []
    _log_kw = tuple(keywords) if keywords else None
    seen: set[Tuple[int, int, int, int]] = set()

    for rgb, ws, hs, lab in variant_list:
        if _stop_req(stop_event):
            return []

        if eng != ENGINE_TESSERACT:
            wboxes = ocr_word_boxes(
                rgb,
                eng,
                preprocess_label=lab,
                alert_keywords=_log_kw,
            )
            if not wboxes:
                continue
            hit_boxes = _keyword_hit_word_boxes(wboxes, kws)
            if not hit_boxes:
                continue
            xs0 = min(b[1][0] for b in hit_boxes)
            ys0 = min(b[1][1] for b in hit_boxes)
            xe = max(b[1][0] + b[1][2] for b in hit_boxes)
            ye = max(b[1][1] + b[1][3] for b in hit_boxes)
            fx, fy, fw, fh = _map_ocr_rect_to_frame(
                xs0, ys0, xe - xs0, ye - ys0, w0, h0, ws, hs
            )
            if fw > 0 and fh > 0:
                return [
                    OverlayRect(
                        fx,
                        fy,
                        fw,
                        fh,
                        stable_overlay_bgr("kwf", fx, fy, fw, fh),
                        "키워드(영역)",
                    )
                ]
            continue

        if pytesseract is None:
            return []

        for psm in _tesseract_overlay_psms():
            if _stop_req(stop_event):
                return []
            data = tesseract_image_to_data(
                rgb,
                psm,
                preprocess_label=lab,
                alert_keywords=_log_kw,
            )
            if not data or not data.get("text"):
                continue
            n = len(data["text"])
            has_level = "level" in data

            if has_level:
                for L in (4, 3, 2):
                    chunk: List[OverlayRect] = []
                    for i in range(n):
                        if _tesseract_row_level(data, i) != L:
                            continue
                        t = (data["text"][i] or "").strip().lower()
                        if not t or not any(k in t for k in kws):
                            continue
                        try:
                            x, y, ww, hh = (
                                int(data["left"][i]),
                                int(data["top"][i]),
                                int(data["width"][i]),
                                int(data["height"][i]),
                            )
                        except (ValueError, TypeError, KeyError):
                            continue
                        if ww < 2 or hh < 2:
                            continue
                        fx, fy, fw, fh = _map_ocr_rect_to_frame(
                            x, y, ww, hh, w0, h0, ws, hs
                        )
                        if fw <= 0 or fh <= 0:
                            continue
                        key = (fx, fy, fw, fh)
                        if key in seen:
                            continue
                        seen.add(key)
                        chunk.append(
                            OverlayRect(
                                fx,
                                fy,
                                fw,
                                fh,
                                stable_overlay_bgr("kwl", fx, fy, fw, fh),
                                "키워드(줄)",
                            )
                        )
                    if chunk:
                        return chunk

            wboxes = boxes_from_tesseract_data(data)
            if not wboxes:
                continue
            hit_boxes = _keyword_hit_word_boxes(wboxes, kws)
            if not hit_boxes:
                continue
            xs0 = min(b[1][0] for b in hit_boxes)
            ys0 = min(b[1][1] for b in hit_boxes)
            xe = max(b[1][0] + b[1][2] for b in hit_boxes)
            ye = max(b[1][1] + b[1][3] for b in hit_boxes)
            fx, fy, fw, fh = _map_ocr_rect_to_frame(
                xs0, ys0, xe - xs0, ye - ys0, w0, h0, ws, hs
            )
            if fw > 0 and fh > 0:
                return [
                    OverlayRect(
                        fx,
                        fy,
                        fw,
                        fh,
                        stable_overlay_bgr("kwf", fx, fy, fw, fh),
                        "키워드(영역)",
                    )
                ]
    return []


def _overlay_keyword_union_any_variant(
    frame_bgr: np.ndarray,
    keywords: Tuple[str, ...],
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
) -> List[OverlayRect]:
    """
    문자열 OCR로만 키워드가 잡힐 때, 모든 전처리 변형에서 단어 박스를 모아
    키워드가 포함된 텍스트가 있으면 그 단어들의 합집합 박스 1개로 표시.
    """
    if not keywords:
        return []
    eng = normalize_ocr_engine(ocr_engine)
    ok, _ = ocr_engine_runtime_ok(eng)
    if not ok:
        return []
    kws = tuple((k or "").strip().lower() for k in keywords if (k or "").strip())
    if not kws:
        return []
    h0, w0 = frame_bgr.shape[:2]
    variant_list = list(
        _iter_game_ocr_variants_rgb_with_scale(
            frame_bgr,
            variant_groups=variant_groups,
        )
    )
    if eng == ENGINE_TESSERACT:
        variant_list = _cap_tesseract_variants(variant_list)
    _log_kw = tuple(keywords) if keywords else None
    if not variant_list:
        return []

    def _kwn_rect_from_words(
        words: List[Tuple[str, Tuple[int, int, int, int]]],
        ws: int,
        hs: int,
    ) -> Optional[List[OverlayRect]]:
        hit = _keyword_hit_word_boxes(words, kws)
        if not hit:
            return None
        xs0 = min(b[1][0] for b in hit)
        ys0 = min(b[1][1] for b in hit)
        xe = max(b[1][0] + b[1][2] for b in hit)
        ye = max(b[1][1] + b[1][3] for b in hit)
        fx, fy, fw, fh = _map_ocr_rect_to_frame(
            xs0, ys0, xe - xs0, ye - ys0, w0, h0, ws, hs
        )
        if fw <= 0 or fh <= 0:
            return None
        return [
            OverlayRect(
                fx,
                fy,
                fw,
                fh,
                stable_overlay_bgr("kwn", fx, fy, fw, fh),
                "키워드(통합)",
            )
        ]

    def _parallel_first_hit(
        job,
    ) -> List[OverlayRect]:
        mw = _variant_parallel_workers(len(variant_list))
        ex = ThreadPoolExecutor(max_workers=mw)
        futs = [ex.submit(job, tri) for tri in variant_list]
        found: Optional[List[OverlayRect]] = None
        shutdown_cancel = False
        try:
            for fut in as_completed(futs):
                if _stop_req(stop_event):
                    shutdown_cancel = True
                    break
                r = fut.result()
                if r is not None:
                    found = r
                    shutdown_cancel = True
                    break
        finally:
            if shutdown_cancel:
                _pool_shutdown_cancel(ex)
            else:
                ex.shutdown(wait=True)
        if _stop_req(stop_event):
            return []
        return found if found is not None else []

    if eng != ENGINE_TESSERACT:

        def nu_job(
            tri: Tuple[np.ndarray, int, int, str],
        ) -> Optional[List[OverlayRect]]:
            rgb, ws, hs, lab = tri
            if _stop_req(stop_event):
                return None
            words = ocr_word_boxes(
                rgb,
                eng,
                preprocess_label=lab,
                alert_keywords=_log_kw,
            )
            if not words:
                return None
            return _kwn_rect_from_words(words, ws, hs)

        return _parallel_first_hit(nu_job)

    def te_job(tri: Tuple[np.ndarray, int, int, str]) -> Optional[List[OverlayRect]]:
        rgb, ws, hs, lab = tri
        if _stop_req(stop_event):
            return None
        for psm in _tesseract_overlay_psms():
            if _stop_req(stop_event):
                return None
            data = tesseract_image_to_data(
                rgb, psm, preprocess_label=lab, alert_keywords=_log_kw
            )
            if not data:
                continue
            words = boxes_from_tesseract_data(data)
            if not words:
                continue
            hit = _kwn_rect_from_words(words, ws, hs)
            if hit is not None:
                return hit
        return None

    return _parallel_first_hit(te_job)


def _overlay_keyword_hits_with_boxes(
    frame_bgr: np.ndarray,
    keywords: Tuple[str, ...],
    ocr_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
    ws: int,
    hs: int,
) -> List[OverlayRect]:
    if not keywords or not ocr_boxes:
        return []
    h0, w0 = frame_bgr.shape[:2]
    seen: set[Tuple[int, int, int, int]] = set()
    out: List[OverlayRect] = []
    kws = tuple((k or "").strip().lower() for k in keywords if (k or "").strip())
    if not kws:
        return []
    for text, (x, y, w, h) in ocr_boxes:
        tl = text.lower()
        if not any(k in tl for k in kws):
            continue
        fx, fy, fw, fh = _map_ocr_rect_to_frame(x, y, w, h, w0, h0, ws, hs)
        if fw <= 0 or fh <= 0:
            continue
        key = (fx, fy, fw, fh)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            OverlayRect(fx, fy, fw, fh, stable_overlay_bgr("kw", fx, fy, fw, fh), "키워드")
        )
    return out


def _overlay_keyword_hits(
    frame_bgr: np.ndarray,
    keywords: Tuple[str, ...],
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
    log_alert_keywords: Tuple[str, ...] = (),
) -> List[OverlayRect]:
    if not keywords:
        return []
    ok, _ = ocr_engine_runtime_ok(normalize_ocr_engine(ocr_engine))
    if not ok:
        return []
    la = log_alert_keywords if log_alert_keywords else tuple(keywords)
    boxes, ws, hs = _ocr_boxes_best_from_frame(
        frame_bgr,
        ocr_engine,
        stop_event,
        variant_groups=variant_groups,
        log_alert_keywords=la,
    )
    return _overlay_keyword_hits_with_boxes(frame_bgr, keywords, boxes, ws, hs)


def check_plain_text(
    frame_bgr: np.ndarray,
    keywords: Tuple[str, ...],
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
) -> bool:
    if not keywords:
        return False
    if variant_groups == OCR_VARIANT_GROUPS_DISABLED:
        return False
    ok, _ = ocr_engine_runtime_ok(normalize_ocr_engine(ocr_engine))
    if not ok:
        return False
    kws = tuple(
        (kw or "").strip().lower() for kw in keywords if (kw or "").strip()
    )
    if not kws:
        return False
    if _stop_req(stop_event):
        return False
    eng = normalize_ocr_engine(ocr_engine)
    quick = _ocr_string_from_frame_bgr(
        frame_bgr,
        seek_keywords=None,
        ocr_engine=eng,
        stop_event=stop_event,
        variant_groups=variant_groups,
        log_alert_keywords=tuple(keywords),
    ).lower()
    if quick.strip() and any(k in quick for k in kws):
        return True
    if _stop_req(stop_event):
        return False
    merged = _ocr_string_from_frame_bgr(
        frame_bgr,
        seek_keywords=kws,
        ocr_engine=eng,
        stop_event=stop_event,
        variant_groups=variant_groups,
        log_alert_keywords=tuple(keywords),
    ).lower()
    return bool(merged.strip()) and any(k in merged for k in kws)


def _run_keyword_detection_for_engine(
    frame_bgr: np.ndarray,
    alert_keywords: Tuple[str, ...],
    eng: str,
    stop_event: Optional[threading.Event],
    *,
    variant_groups: Tuple[str, ...],
) -> Tuple[bool, List[OverlayRect]]:
    """단일 OCR 엔진으로 키워드 알림 여부 + 박스."""
    if _stop_req(stop_event):
        return False, []
    ocr_ok, _ = ocr_engine_runtime_ok(eng)
    log_kw = tuple(alert_keywords)
    ocr_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
    ocr_ws, ocr_hs = 1, 1
    if ocr_ok and alert_keywords:
        ocr_boxes, ocr_ws, ocr_hs = _ocr_boxes_best_from_frame(
            frame_bgr,
            eng,
            stop_event,
            variant_groups=variant_groups,
            log_alert_keywords=log_kw,
        )
    if _stop_req(stop_event):
        return False, []

    kw_ovs = _overlay_keyword_hits_with_boxes(
        frame_bgr, alert_keywords, ocr_boxes, ocr_ws, ocr_hs
    )
    if not kw_ovs and alert_keywords:
        kw_ovs = _overlay_keyword_text_fallback(
            frame_bgr,
            alert_keywords,
            eng,
            stop_event,
            variant_groups=variant_groups,
        )
    if _stop_req(stop_event):
        return False, []

    if kw_ovs:
        plain_hits = True
    elif alert_keywords:
        plain_hits = check_plain_text(
            frame_bgr,
            alert_keywords,
            eng,
            stop_event,
            variant_groups=variant_groups,
        )
    else:
        plain_hits = False
    if _stop_req(stop_event):
        return False, []

    if plain_hits and not kw_ovs and alert_keywords:
        kw_ovs = _overlay_keyword_union_any_variant(
            frame_bgr,
            alert_keywords,
            eng,
            stop_event,
            variant_groups=variant_groups,
        )

    return plain_hits, kw_ovs


def run_keyword_detection(
    frame_bgr: np.ndarray,
    alert_keywords: Tuple[str, ...],
    ocr_engines: Tuple[str, ...] = (DEFAULT_OCR_ENGINE,),
    stop_event: Optional[threading.Event] = None,
    *,
    variant_groups: Tuple[str, ...] = (),
    kw_abort: Optional[threading.Event] = None,
) -> Tuple[bool, List[OverlayRect]]:
    """키워드 알림 여부 + 미리보기용 키워드 박스. 엔진이 둘 이상이면 엔진마다 스레드로 병렬 실행."""
    global _kw_abort_token
    prev_token = _kw_abort_token
    _kw_abort_token = kw_abort
    try:
        if _stop_req(stop_event):
            return False, []
        if variant_groups == OCR_VARIANT_GROUPS_DISABLED:
            return False, []
        engines = _coerce_ocr_engines(ocr_engines)
        if not engines:
            return False, []

        def _run_one_engine(
            eng: str,
        ) -> Tuple[bool, List[OverlayRect]]:
            try:
                if _stop_req(stop_event):
                    return False, []
                ok, _ = ocr_engine_runtime_ok(eng)
                if not ok:
                    return False, []
                return _run_keyword_detection_for_engine(
                    frame_bgr,
                    alert_keywords,
                    eng,
                    stop_event,
                    variant_groups=variant_groups,
                )
            except Exception:
                return False, []

        any_hit = False
        merged: List[OverlayRect] = []
        seen_rect: set[Tuple[int, int, int, int]] = set()

        if len(engines) == 1:
            eng = engines[0]
            hit, ovs = _run_one_engine(eng)
            any_hit = hit
            merged = list(ovs)
        else:
            max_w = min(len(engines), 8)
            ex = ThreadPoolExecutor(max_workers=max_w)
            futs = [ex.submit(_run_one_engine, eng) for eng in engines]
            shutdown_cancel = False
            try:
                for fut in as_completed(futs):
                    if _stop_req(stop_event):
                        shutdown_cancel = True
                        break
                    try:
                        hit, ovs = fut.result()
                    except Exception:
                        continue
                    any_hit = any_hit or hit
                    for ov in ovs:
                        key = (ov.x, ov.y, ov.w, ov.h)
                        if key in seen_rect:
                            continue
                        seen_rect.add(key)
                        merged.append(ov)
            finally:
                if shutdown_cancel:
                    _pool_shutdown_cancel(ex)
                else:
                    ex.shutdown(wait=True)

        return any_hit, merged
    finally:
        _kw_abort_token = prev_token

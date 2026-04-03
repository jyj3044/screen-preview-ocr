"""
키워드 OCR 엔진 추상화: Tesseract / EasyOCR / RapidOCR.
각 백엔드는 RGB uint8 ndarray 입력을 가정한다 (OpenCV BGR 프레임은 호출부에서 변환).
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from .ocr_diag import begin_ocr_call, end_ocr_call


def _shape_prep_detail(
    rgb: np.ndarray, preprocess_label: str = "", extra: str = ""
) -> str:
    h, w = rgb.shape[:2]
    parts = [f"{w}x{h}"]
    p = (preprocess_label or "").strip()
    if p:
        parts.append(p)
    e = (extra or "").strip()
    if e:
        parts.append(e)
    return " ".join(parts)


def _keyword_alert_hit(
    text_blob: str, alert_keywords: Optional[Tuple[str, ...]]
) -> Optional[bool]:
    if not alert_keywords:
        return None
    kws = tuple((k or "").strip().lower() for k in alert_keywords if (k or "").strip())
    if not kws:
        return None
    b = (text_blob or "").lower()
    return any(k in b for k in kws)


def _boxes_text_blob(
    boxes: List[Tuple[str, Tuple[int, int, int, int]]],
) -> str:
    return " ".join((b[0] or "").lower() for b in boxes)

ENGINE_TESSERACT = "tesseract"
ENGINE_EASYOCR = "easyocr"
ENGINE_RAPIDOCR = "rapidocr"
DEFAULT_OCR_ENGINE = ENGINE_TESSERACT
ALL_OCR_ENGINES: Tuple[str, ...] = (
    ENGINE_TESSERACT,
    ENGINE_EASYOCR,
    ENGINE_RAPIDOCR,
)

_OCR_ENGINE_ALIASES: Dict[str, str] = {
    "tess": ENGINE_TESSERACT,
    "easy": ENGINE_EASYOCR,
    "rapid": ENGINE_RAPIDOCR,
    "rapid_ocr": ENGINE_RAPIDOCR,
    "rapid-ocr": ENGINE_RAPIDOCR,
}

_TESS_LANGS: Tuple[str, ...] = ("kor+eng", "kor", "eng+kor", "eng")

_easy_reader = None
_rapid_ocr = None
_rapid_ocr_korean_active = False
_RAPID_INIT_LOCK = threading.Lock()
_RAPID_INFER_LOCK = threading.Lock()

# rapidocr_onnxruntime 기본값은 중국어 인식(rec)만 있어 한글이 거의 안 나옴 → PP-OCR 한국어 rec + 사전 사용
_RAPID_KO_ONNX_URL = (
    "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv1/korean_mobile_v2.0_rec_infer.onnx"
)
_RAPID_KO_KEYS_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/korean_dict.txt"
)


def _rapid_korean_cache_dir() -> Path:
    p = Path.home() / ".cache" / "maplealert" / "rapidocr_korean"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_to_file(url: str, dest: Path, *, min_bytes: int, timeout_sec: float = 120.0) -> bool:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = Request(url, headers={"User-Agent": "MapleAlert/1.0 (RapidOCR Korean assets)"})
        with urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read()
        if len(data) < min_bytes:
            return False
        tmp.write_bytes(data)
        tmp.replace(dest)
        return True
    except (OSError, URLError, ValueError):
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False


def _ensure_rapid_korean_assets() -> Tuple[Optional[Path], Optional[Path]]:
    d = _rapid_korean_cache_dir()
    onnx_p = d / "korean_mobile_v2.0_rec_infer.onnx"
    keys_p = d / "korean_dict.txt"
    ok_o = _download_to_file(_RAPID_KO_ONNX_URL, onnx_p, min_bytes=500_000)
    ok_k = _download_to_file(_RAPID_KO_KEYS_URL, keys_p, min_bytes=10_000)
    if ok_o and ok_k:
        return onnx_p, keys_p
    return None, None


def normalize_ocr_engine(name: str) -> str:
    n = (name or "").strip().lower()
    if not n:
        return ""
    # 제거된 엔진(설정 JSON 호환): 무시
    if n in ("paddleocr", "paddle"):
        return ""
    if n in ALL_OCR_ENGINES:
        return n
    if n in _OCR_ENGINE_ALIASES:
        return _OCR_ENGINE_ALIASES[n]
    return DEFAULT_OCR_ENGINE


def _get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        import easyocr

        _easy_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _easy_reader


def _get_rapid():
    global _rapid_ocr, _rapid_ocr_korean_active
    if _rapid_ocr is not None:
        return _rapid_ocr
    with _RAPID_INIT_LOCK:
        if _rapid_ocr is not None:
            return _rapid_ocr
        from rapidocr_onnxruntime import RapidOCR

        inst = None
        korean_ok = False
        onnx_p, keys_p = _ensure_rapid_korean_assets()
        if onnx_p is not None and keys_p is not None:
            r_onnx = str(onnx_p.resolve())
            r_keys = str(keys_p.resolve())
            for rec_shape in ([3, 32, 320], [3, 48, 320]):
                try:
                    inst = RapidOCR(
                        rec_model_path=r_onnx,
                        rec_keys_path=r_keys,
                        rec_img_shape=rec_shape,
                    )
                    korean_ok = True
                    break
                except Exception:
                    inst = None
        if inst is None:
            inst = RapidOCR()
            korean_ok = False
        _rapid_ocr = inst
        _rapid_ocr_korean_active = korean_ok
        return _rapid_ocr


def _quad_to_xywh(box) -> Tuple[int, int, int, int]:
    arr = np.asarray(box, dtype=np.float64).reshape(-1, 2)
    x0, y0 = arr.min(axis=0)
    x1, y1 = arr.max(axis=0)
    w = max(1, int(round(x1 - x0)))
    h = max(1, int(round(y1 - y0)))
    return int(round(x0)), int(round(y0)), w, h


def _parse_tesseract_conf(raw) -> float:
    try:
        if raw is None or raw == "":
            return -1.0
        v = float(raw)
        if math.isnan(v):
            return -1.0
        return v
    except (ValueError, TypeError):
        return -1.0


def _boxes_from_tesseract_dict(
    data: Dict[str, List],
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        c = _parse_tesseract_conf(data.get("conf", [""])[i])
        has_hangul = any("\uac00" <= ch <= "\ud7a3" for ch in text)
        if c < 0:
            if not has_hangul and len(text) < 2:
                continue
        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        out.append((text, (x, y, w, h)))
    return out


def _tesseract_dict_text_blob(data: Optional[Dict[str, List]]) -> str:
    if not data:
        return ""
    parts = data.get("text") or []
    return " ".join((str(t) or "").lower() for t in parts)


def tesseract_image_to_data(
    rgb: np.ndarray,
    ocr_config: str,
    *,
    preprocess_label: str = "",
    alert_keywords: Optional[Tuple[str, ...]] = None,
) -> Optional[Dict[str, List]]:
    try:
        import pytesseract
    except ImportError:
        return None
    cfg_short = (ocr_config or "")[:40]
    for lang in _TESS_LANGS:
        try:
            d = _shape_prep_detail(rgb, preprocess_label, f"lang={lang} {cfg_short}")
            cid = begin_ocr_call("image_to_data", "tesseract", d)
            t0 = time.perf_counter()
            out: Optional[Dict[str, List]] = None
            try:
                out = pytesseract.image_to_data(
                    rgb,
                    output_type=pytesseract.Output.DICT,
                    lang=lang,
                    config=ocr_config,
                )
            finally:
                end_ocr_call(
                    cid,
                    "image_to_data",
                    "tesseract",
                    time.perf_counter() - t0,
                    d,
                    keyword_alert_hit=_keyword_alert_hit(
                        _tesseract_dict_text_blob(out), alert_keywords
                    ),
                )
            return out
        except Exception:
            continue
    d2 = _shape_prep_detail(rgb, preprocess_label, f"lang=- {cfg_short}")
    cid2 = begin_ocr_call("image_to_data", "tesseract", d2)
    t0 = time.perf_counter()
    out_fb: Optional[Dict[str, List]] = None
    try:
        out_fb = pytesseract.image_to_data(
            rgb, output_type=pytesseract.Output.DICT, config=ocr_config
        )
    except Exception:
        pass
    finally:
        end_ocr_call(
            cid2,
            "image_to_data",
            "tesseract",
            time.perf_counter() - t0,
            d2,
            keyword_alert_hit=_keyword_alert_hit(
                _tesseract_dict_text_blob(out_fb), alert_keywords
            ),
        )
    return out_fb


def tesseract_call_image_to_string(
    rgb: np.ndarray,
    *,
    lang: Optional[str] = None,
    config: str = "",
    preprocess_label: str = "",
    alert_keywords: Optional[Tuple[str, ...]] = None,
) -> str:
    """keywords.py용: image_to_string + 진단 로그."""
    try:
        import pytesseract
    except ImportError:
        return ""
    d = _shape_prep_detail(
        rgb, preprocess_label, f"{lang or '-'} {(config or '')[:40]}"
    )
    cid = begin_ocr_call("image_to_string", "tesseract", d)
    t0 = time.perf_counter()
    out = ""
    try:
        if lang:
            s = pytesseract.image_to_string(rgb, lang=lang, config=config)
        else:
            s = pytesseract.image_to_string(rgb, config=config)
        out = s if s else ""
    except Exception:
        pass
    finally:
        end_ocr_call(
            cid,
            "image_to_string",
            "tesseract",
            time.perf_counter() - t0,
            d,
            keyword_alert_hit=_keyword_alert_hit(out, alert_keywords),
        )
    return out


def boxes_from_tesseract_data(
    data: Dict[str, List],
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    return _boxes_from_tesseract_dict(data)


def ocr_word_boxes_tesseract(
    rgb: np.ndarray,
    tesseract_psm: str = "--oem 3 --psm 6",
    *,
    preprocess_label: str = "",
    alert_keywords: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    data = tesseract_image_to_data(
        rgb,
        tesseract_psm,
        preprocess_label=preprocess_label,
        alert_keywords=alert_keywords,
    )
    if not data:
        return []
    return _boxes_from_tesseract_dict(data)


def ocr_word_boxes_easyocr(
    rgb: np.ndarray,
    *,
    preprocess_label: str = "",
    alert_keywords: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    reader = _get_easy_reader()
    detail = _shape_prep_detail(rgb, preprocess_label)
    cid = begin_ocr_call("readtext", "easyocr", detail)
    t0 = time.perf_counter()
    rows = None
    # 탐지 임계값을 올려 배경·이진 노이즈에서 가짜 박스가 줄어듦. decoder=greedy 가 더 빠름.
    _rt_kw = dict(
        decoder="greedy",
        paragraph=False,
        text_threshold=0.65,
        low_text=0.48,
        link_threshold=0.35,
    )
    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    try:
        try:
            rows = reader.readtext(rgb, **_rt_kw)
        except TypeError:
            try:
                rows = reader.readtext(rgb, decoder="greedy", paragraph=False)
            except Exception:
                rows = None
        except Exception:
            rows = None
        if rows is not None:
            for row in rows:
                if len(row) < 2:
                    continue
                bbox, text = row[0], row[1]
                conf = float(row[2]) if len(row) > 2 else 1.0
                if conf < 0.22:
                    continue
                t = (text or "").strip()
                if not t:
                    continue
                x, y, w, h = _quad_to_xywh(bbox)
                out.append((t, (x, y, w, h)))
    finally:
        end_ocr_call(
            cid,
            "readtext",
            "easyocr",
            time.perf_counter() - t0,
            detail,
            keyword_alert_hit=_keyword_alert_hit(
                _boxes_text_blob(out), alert_keywords
            ),
        )
    return out


def ocr_word_boxes_rapidocr(
    rgb: np.ndarray,
    *,
    preprocess_label: str = "",
    alert_keywords: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    engine = _get_rapid()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detail = _shape_prep_detail(rgb, preprocess_label)
    cid = begin_ocr_call("infer", "rapidocr", detail)
    t0 = time.perf_counter()
    raw = None
    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    try:
        try:
            with _RAPID_INFER_LOCK:
                raw = engine(bgr)
        except Exception:
            raw = None
        if raw is not None:
            if isinstance(raw, tuple) and len(raw) >= 1:
                result = raw[0]
            else:
                result = raw
            if result:
                for item in result:
                    if item is None:
                        continue
                    if hasattr(item, "text") and hasattr(item, "box"):
                        box, text = item.box, item.text
                        score = float(getattr(item, "score", 1.0))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        box, text = item[0], item[1]
                        score = float(item[2]) if len(item) > 2 else 1.0
                    else:
                        continue
                    if score < 0.12:
                        continue
                    t = (str(text) or "").strip()
                    if not t:
                        continue
                    x, y, w, h = _quad_to_xywh(box)
                    out.append((t, (x, y, w, h)))
    finally:
        end_ocr_call(
            cid,
            "infer",
            "rapidocr",
            time.perf_counter() - t0,
            detail,
            keyword_alert_hit=_keyword_alert_hit(
                _boxes_text_blob(out), alert_keywords
            ),
        )
    return out


def ocr_word_boxes(
    rgb: np.ndarray,
    engine: str,
    *,
    tesseract_psm: str = "--oem 3 --psm 6",
    preprocess_label: str = "",
    alert_keywords: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    eng = normalize_ocr_engine(engine)
    if eng == ENGINE_TESSERACT:
        return ocr_word_boxes_tesseract(
            rgb,
            tesseract_psm,
            preprocess_label=preprocess_label,
            alert_keywords=alert_keywords,
        )
    if eng == ENGINE_EASYOCR:
        return ocr_word_boxes_easyocr(
            rgb, preprocess_label=preprocess_label, alert_keywords=alert_keywords
        )
    if eng == ENGINE_RAPIDOCR:
        return ocr_word_boxes_rapidocr(
            rgb, preprocess_label=preprocess_label, alert_keywords=alert_keywords
        )
    return []


def joined_text_from_rgb(
    rgb: np.ndarray,
    engine: str,
    *,
    preprocess_label: str = "",
    alert_keywords: Optional[Tuple[str, ...]] = None,
) -> str:
    """단일 RGB 이미지에서 한 줄로 이은 전체 문자열 (뉴럴 엔진용)."""
    eng = normalize_ocr_engine(engine)
    if eng == ENGINE_EASYOCR:
        boxes = ocr_word_boxes_easyocr(
            rgb, preprocess_label=preprocess_label, alert_keywords=alert_keywords
        )
        return " ".join(t for t, _ in boxes)
    if eng == ENGINE_RAPIDOCR:
        boxes = ocr_word_boxes_rapidocr(
            rgb, preprocess_label=preprocess_label, alert_keywords=alert_keywords
        )
        return " ".join(t for t, _ in boxes)
    return ""


def ocr_engine_runtime_ok(engine: str) -> Tuple[bool, str]:
    eng = normalize_ocr_engine(engine)
    if not eng:
        return False, "알 수 없는 OCR 엔진"
    if eng == ENGINE_TESSERACT:
        try:
            import pytesseract
            import pytesseract.pytesseract as pt

            pt.get_tesseract_version()
            return True, ""
        except ImportError:
            return False, "pytesseract 미설치. pip install pytesseract (+ Tesseract 본체)"
        except Exception as e:
            return False, str(e)
    if eng == ENGINE_EASYOCR:
        try:
            import easyocr  # noqa: F401

            return True, ""
        except ImportError:
            return False, "easyocr 미설치. pip install easyocr"
        except Exception as e:
            return False, str(e)
    if eng == ENGINE_RAPIDOCR:
        try:
            import rapidocr_onnxruntime  # noqa: F401

            _get_rapid()
            if _rapid_ocr_korean_active:
                return True, ""
            return (
                True,
                "한글 인식용 모델·사전을 받지 못했습니다. "
                "중국어 기본 인식만 사용되어 한글이 거의 안 나올 수 있습니다. "
                "네트워크 확인 후 "
                f"{_rapid_korean_cache_dir()} 에 ONNX·korean_dict.txt 가 생기는지 보세요.",
            )
        except ImportError:
            return False, "rapidocr-onnxruntime 미설치. pip install rapidocr-onnxruntime"
        except Exception as e:
            return False, str(e)
    return False, "알 수 없는 OCR 엔진"

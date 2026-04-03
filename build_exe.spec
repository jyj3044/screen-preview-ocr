# -*- mode: python ; coding: utf-8 -*-
# 프로젝트 루트에서: pyinstaller --noconfirm build_exe.spec
# 산출물: dist/cyj.exe  (PyInstaller 가 주입하는 SPEC = 이 spec 파일 경로)

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

ROOT = Path(SPEC).resolve().parent

block_cipher = None

# RapidOCR: ONNX·yaml·서브모듈 전부 (미포함 시 exe 에서 import 실패)
_rapid_datas, _rapid_bins, _rapid_hidden = collect_all("rapidocr_onnxruntime")

datas = list(_rapid_datas)
binaries = list(_rapid_bins)
try:
    binaries += collect_dynamic_libs("cv2")
except Exception:
    pass
try:
    binaries += collect_dynamic_libs("onnxruntime")
except Exception:
    pass

# onnxruntime: Python 훅만으로는 data/서브모듈 누락 시 rapidocr import 단계에서 실패할 수 있음
_onnx_hidden: list = []
try:
    _onnx_d, _onnx_b, _onnx_h = collect_all("onnxruntime")
    datas += list(_onnx_d)
    binaries += list(_onnx_b)
    _onnx_hidden = list(_onnx_h)
except Exception:
    pass

_base_hidden = [
    "PIL._tkinter_finder",
    "detection",
    "detection.common",
    "detection.keywords",
    "detection.pipeline",
    "detection.ocr_backends",
    "detection.ocr_diag",
    "detection.templates",
    "detection.overlay_store",
    "windows_capture",
    "tesseract_win_console",
    "setproctitle",
    "mss",
    "cv2",
    "numpy",
    "pytesseract",
    "rapidocr_onnxruntime",
    "onnxruntime",
]
hiddenimports = list(
    dict.fromkeys(_base_hidden + list(_rapid_hidden) + list(_onnx_hidden))
)
# EasyOCR 을 exe 에 넣으려면: venv 에 easyocr 설치 후 아래 줄 주석 해제 (용량·빌드 시간 크게 증가)
# hiddenimports.append("easyocr")

# 앱에서 PaddleOCR 미사용 — 전역에 설치돼 있어도 번들에 넣지 않음
_paddle_excludes = (
    "paddleocr",
    "paddle",
    "paddlepaddle",
    "paddlex",
    "paddlenlp",
)

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "pyi_rthook_onnx.py")],
    excludes=list(_paddle_excludes),
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="cyj",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

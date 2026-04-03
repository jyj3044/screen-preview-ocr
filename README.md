# Alert (cyj)

Windows에서 **특정 창 화면을 캡처**하고, **키워드 OCR**과 **템플릿 이미지 매칭**으로 이벤트를 감지하면 **알림음**을 재생하는 데스크톱 앱입니다.  
미리보기·설정은 **Tkinter** GUI로 제공합니다.

- **OCR 엔진**: Tesseract, EasyOCR, RapidOCR 중 UI에서 선택(복수 선택 가능).  
  한글 인식 보조를 위해 RapidOCR 사용 시 ONNX·사전 파일을 최초에 자동으로 내려받을 수 있습니다.
- **실행 파일 이름·창 제목**: 기본 `cyj` (`main.py`의 `APP_NAME`).
- **설정 저장**: `alert_settings.json` (소스 실행 시 프로젝트 루트, **exe 실행 시 exe와 같은 폴더**).

---

## 필요 환경

- **OS**: Windows (캡처·알림은 Win32 기준으로 동작)
- **Python**: 3.10 이상 권장 (개발 시 3.12 등에서 검증)

---

## 저장소를 처음 받았을 때 (개발 실행)

### 1) 가상환경 만들기 (권장)

```powershell
cd alert
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Python 패키지 설치

**일반 개발·실행용** (EasyOCR 포함, 용량 큼):

```powershell
pip install -r requirements.txt
```

`requirements.txt`에 포함되는 주요 항목은 다음과 같습니다.

- `mss`, `numpy`, `opencv-python`, `Pillow`, `setproctitle`, `pytesseract`
- `easyocr`, `rapidocr-onnxruntime` (OCR 엔진)

### 3) Tesseract 본체 (Tesseract 엔진을 쓸 때만)

`pytesseract`는 Python 래퍼이며, **Tesseract 실행 파일은 별도 설치**가 필요합니다.

- 설치 가이드: [UB-Mannheim Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- 기본적으로 `C:\Program Files\Tesseract-OCR\tesseract.exe` 가 있으면 앱에서 자동으로 가리킵니다. 없으면 **PATH**에 `tesseract`가 잡혀 있어야 합니다.

### 4) 실행

```powershell
python main.py
```

---

## 실행 파일(exe) 만들기

### 권장: 가벼운 exe (EasyOCR·PyTorch 제외)

전역 Python에 `easyocr`가 있으면 PyInstaller가 **torch까지 끌어와 exe가 수백 MB**가 될 수 있으므로, **새 venv**에서 빌드하는 것을 권장합니다.

```powershell
cd D:\경로\alert
python -m venv .venv-build
.\.venv-build\Scripts\activate.bat
python -m pip install --upgrade pip
```

(`activate` 대신 `activate.bat` — PowerShell에서 스크립트 실행 정책 경고를 피하려면 CMD에서 실행하거나 `activate.bat` 사용.)

빌드 전용 의존성 (`requirements-build.txt` → `requirements-runtime.txt` + PyInstaller):

```powershell
pip install -r requirements-build.txt
```

**방법 A — 배치 파일**

```powershell
build_exe.bat
```

**방법 B — 직접 PyInstaller**

```powershell
pyinstaller --noconfirm build_exe.spec
```

완료 후 산출물:

- **`dist\cyj\` 폴더 전체** + 그 안의 `cyj.exe` (기본). 단일 exe가 아니라 **폴더째** 배포·실행합니다. RapidOCR/ONNX는 이 방식에서 훨씬 잘 동작합니다.
- 예전처럼 **파일 하나만** 쓰려면 `build_exe.spec` 맨 위 `ONEFILE = True` 로 바꿔 빌드하세요 (같은 PC에서 `python main.py`는 되는데 exe만 깨질 때는 `False` 권장).

### exe 배포 시 참고

- **Tesseract**: exe에 포함되지 않습니다. 사용 PC에 Tesseract 설치 또는 PATH 설정이 필요합니다.
- **기본 OCR**: exe는 기본으로 **RapidOCR**을 쓰도록 되어 있으며, RapidOCR 관련 파일은 spec에서 번들됩니다.
- **EasyOCR를 exe에 넣으려면**: `build_exe.spec` 에서 `INCLUDE_EASYOCR = True`, venv에 `pip install easyocr`, `hiddenimports.append("easyocr")` 주석 해제 후 빌드 (용량·빌드 시간 크게 증가, RapidOCR과 DLL 충돌 가능).
- **RapidOCR / `onnxruntime_pybind11_state` DLL 초기화 실패 (`python main.py` 는 되는데 exe만 안 될 때)**:
  - PyInstaller가 **`import easyocr` 를 따라가며 torch 전체를 번들**에 넣으면, torch·onnxruntime OpenMP DLL이 겹쳐 exe에서만 깨질 수 있습니다. 기본 spec은 **`INCLUDE_EASYOCR = False`** 로 **easyocr·torch를 제외**합니다. **최신 spec으로 다시 빌드**한 뒤 `dist\cyj\_internal` 안에 **`torch` 폴더가 없는지** 확인하세요.
  - 그 외: `KMP_DUPLICATE_LIB_OK`, `_internal` 안 **DLL이 있는 모든 폴더**를 `add_dll_directory`에 등록(`bootstrap_onnx`), cv2 전 `onnxruntime` 선로드, `collect_all("onnxruntime")`, **onedir** 배포.
  - 여전히 실패 시 `dist\cyj\` 옆에 생기는 **`pyi_rthook_onnx_error.txt`**(rthook 단계 예외) 내용을 확인하세요.
  - **`onnxruntime` 버전**: exe 빌드용 `requirements-runtime.txt` 에서 **1.21.1 로 고정**해 두었습니다. 최신(1.24 등)은 PyInstaller 번들과 VC 런타임 조합에서 위 오류가 나는 보고가 있습니다. venv에서 `pip install -r requirements-build.txt` 후 다시 빌드하세요. venv에 옛 `msvc-runtime` 패키지가 깔려 있으면 제거하거나 최신으로 맞추는 것도 도움이 됩니다.

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `requirements.txt` | 개발 실행용 전체 의존성 |
| `requirements-runtime.txt` | exe 번들용 최소 의존성 (EasyOCR 제외) |
| `requirements-build.txt` | exe 빌드 시 pip 설치 목록 |
| `build_exe.spec` | PyInstaller 설정 |
| `build_exe.bat` | 빌드 자동화 배치 |

---

## 라이선스

저장소에 별도 라이선스 파일이 없다면, 프로젝트 관리자에게 문의하세요.

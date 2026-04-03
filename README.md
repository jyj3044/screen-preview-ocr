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
cd alert
python -m venv .venv-build
.\.venv-build\Scripts\activate
python -m pip install --upgrade pip
```

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

- `dist\cyj.exe` (단일 exe, 콘솔 없음)

### exe 배포 시 참고

- **Tesseract**: exe에 포함되지 않습니다. 사용 PC에 Tesseract 설치 또는 PATH 설정이 필요합니다.
- **기본 OCR**: exe는 기본으로 **RapidOCR**을 쓰도록 되어 있으며, RapidOCR 관련 파일은 spec에서 번들됩니다.
- **EasyOCR를 exe에 넣으려면**: venv에 `pip install easyocr` 후 `build_exe.spec` 안의 `hiddenimports.append("easyocr")` 주석을 해제하고 다시 빌드합니다 (용량·빌드 시간 증가).

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

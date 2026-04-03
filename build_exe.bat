@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo 권장: python -m venv .venv ^& .venv\Scripts\activate 후 이 배치를 실행하면 exe 가 가벼워집니다.
echo.

echo [1/2] pip install (requirements-build.txt)...
python -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1

echo [2/2] PyInstaller...
pyinstaller --noconfirm build_exe.spec
if errorlevel 1 exit /b 1

echo.
echo 완료: dist\cyj.exe
echo - Tesseract 본체는 PC에 별도 설치 (PATH 또는 기본 경로).
echo - EasyOCR 은 requirements-runtime 에 없음. 쓰려면 venv 에 pip install easyocr 후 build_exe.spec 의 easyocr 주석 해제.

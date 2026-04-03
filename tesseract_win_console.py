"""
Windows: pytesseract 가 tesseract.exe 를 subprocess 로 띄울 때 콘솔 창이 깜빡이지 않도록 한다.

- image_to_* 는 subprocess_args → Popen 경로에 CREATE_NO_WINDOW 를 추가.
- get_tesseract_version / get_languages 는 원본이 콘솔 숨김 없이 subprocess 를 써 UI 폴링 시 창이 반복될 수 있음 → 동일하게 숨김 처리.
"""

from __future__ import annotations

import os
import string
import subprocess
import sys


def apply_pytesseract_windows_no_console() -> None:
    if sys.platform != "win32":
        return
    try:
        import pytesseract as pyt_root
        import pytesseract.pytesseract as pt
    except ImportError:
        return

    from packaging.version import InvalidVersion, parse

    cf = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def _win_extra_kwargs() -> dict:
        kw: dict = {}
        if cf:
            kw["creationflags"] = cf
        if hasattr(subprocess, "STARTUPINFO"):
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kw["startupinfo"] = si
        return kw

    _orig_subprocess_args = pt.subprocess_args

    def subprocess_args_with_no_window(include_stdout: bool = True):
        kwargs = _orig_subprocess_args(include_stdout)
        if cf:
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | cf
        return kwargs

    pt.subprocess_args = subprocess_args_with_no_window

    @pt.run_once
    def get_tesseract_version_no_console():
        kw = {
            "stderr": subprocess.STDOUT,
            "env": os.environ,
            "stdin": subprocess.DEVNULL,
            **_win_extra_kwargs(),
        }
        try:
            output = subprocess.check_output([pt.tesseract_cmd, "--version"], **kw)
        except OSError:
            raise pt.TesseractNotFoundError()

        raw_version = output.decode(pt.DEFAULT_ENCODING)
        str_version, *_ = raw_version.lstrip(string.printable[10:]).partition(" ")
        str_version, *_ = str_version.partition("-")

        try:
            version = parse(str_version)
            assert version >= pt.TESSERACT_MIN_VERSION
        except (AssertionError, InvalidVersion):
            raise SystemExit(f'Invalid tesseract version: "{raw_version}"')

        return version

    pt.get_tesseract_version = get_tesseract_version_no_console

    import shlex

    @pt.run_once
    def get_languages_no_console(config: str = ""):
        cmd_args = [pt.tesseract_cmd, "--list-langs"]
        if config:
            cmd_args += shlex.split(config)
        kw = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            **_win_extra_kwargs(),
        }
        try:
            result = subprocess.run(cmd_args, **kw)
        except OSError:
            raise pt.TesseractNotFoundError()

        if result.returncode not in (0, 1):
            raise pt.TesseractNotFoundError()

        languages = []
        if result.stdout:
            for line in result.stdout.decode(pt.DEFAULT_ENCODING).split(os.linesep):
                lang = line.strip()
                if pt.LANG_PATTERN.match(lang):
                    languages.append(lang)
        return languages

    pt.get_languages = get_languages_no_console

    pyt_root.get_tesseract_version = pt.get_tesseract_version
    pyt_root.get_languages = pt.get_languages

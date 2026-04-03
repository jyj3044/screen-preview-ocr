# PyInstaller runtime hook (build_exe.spec 의 runtime_hooks).
# onefile/onedir 에서 onnxruntime 등이 _MEIPASS 아래 DLL 을 찾지 못해 ImportError 나는 경우 완화.
import os
import sys


def _dll_dirs_for_meipass(base: str) -> None:
    if sys.platform != "win32":
        return
    try:
        add = os.add_dll_directory  # type: ignore[attr-defined]
    except AttributeError:
        return
    seen: set[str] = set()

    def one(p: str) -> None:
        ap = os.path.normpath(os.path.abspath(p))
        if ap in seen or not os.path.isdir(ap):
            return
        seen.add(ap)
        try:
            add(ap)
        except OSError:
            pass

    one(base)
    # onnxruntime / rapidocr 등이 흔히 두는 위치
    for sub in (
        "onnxruntime",
        os.path.join("onnxruntime", "capi"),
    ):
        one(os.path.join(base, sub))


def _prepend_path(base: str) -> None:
    p = os.environ.get("PATH", "")
    b = os.path.normpath(os.path.abspath(base))
    if b and b not in p.split(os.pathsep):
        os.environ["PATH"] = b + os.pathsep + p


if getattr(sys, "frozen", False):
    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass:
        _prepend_path(_meipass)
        _dll_dirs_for_meipass(_meipass)

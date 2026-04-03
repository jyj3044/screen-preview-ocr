"""
2단계만: 캡처 스레드에서 받은 화면을 Tkinter로 보여 주기 (감지 없음).

실행: python viewer_demo.py
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from capture import CaptureThread


class ViewerDemo(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("화면 미리보기 (2단계)")
        self.geometry("900x600")

        self._thread: CaptureThread | None = None
        self._photo: ImageTk.PhotoImage | None = None

        bar = ttk.Frame(self, padding=6)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="시작", command=self._start).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="중지", command=self._stop).pack(side=tk.LEFT, padx=2)

        self._canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(40, self._loop)

    def _start(self) -> None:
        self._stop()
        self._thread = CaptureThread(monitor_index=1, target_fps=25)
        self._thread.start()

    def _stop(self) -> None:
        if self._thread:
            self._thread.stop()
            self._thread.join(timeout=2.0)
            self._thread = None

    def _close(self) -> None:
        self._stop()
        self.destroy()

    def _loop(self) -> None:
        if self._thread:
            frame = self._thread.get_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                scale = min(900 / w, 550 / h, 1.0)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._canvas.delete("all")
                cw = max(self._canvas.winfo_width(), nw)
                ch = max(self._canvas.winfo_height(), nh)
                x = (cw - nw) // 2
                y = (ch - nh) // 2
                self._canvas.create_image(x, y, anchor=tk.NW, image=self._photo)
        self.after(40, self._loop)


if __name__ == "__main__":
    ViewerDemo().mainloop()

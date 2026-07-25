"""Dynamic Windows client-area capture for OpenCV applications."""

from __future__ import annotations

import ctypes
import platform
from typing import Optional, Tuple

import cv2
import numpy as np

if platform.system().lower() != "windows":
    raise RuntimeError("Window capture is currently supported on Windows only.")

try:
    import mss
    import win32gui
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise RuntimeError(
        "Window capture requires 'mss' and 'pywin32'. "
        "Install them with: pip install mss pywin32"
    ) from exc


class DynamicWindowCapture:
    """Capture a visible window's client area and follow moves/resizes automatically."""

    def __init__(self, title_keyword: str = "Render") -> None:
        title_keyword = str(title_keyword).strip()
        if not title_keyword:
            raise ValueError("title_keyword must not be empty")

        self.title_keyword = title_keyword
        self._title_lower = title_keyword.lower()
        self.hwnd: Optional[int] = None
        self.window_title = ""
        self.last_size: Optional[Tuple[int, int]] = None
        self.sct = mss.mss()
        self._enable_dpi_awareness()

    @staticmethod
    def _enable_dpi_awareness() -> None:
        """Keep Win32 coordinates aligned with physical pixels at 125%/150% scaling."""
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    def _is_valid_window(self, hwnd: Optional[int]) -> bool:
        return bool(
            hwnd
            and win32gui.IsWindow(hwnd)
            and win32gui.IsWindowVisible(hwnd)
        )

    def _find_window(self) -> Optional[int]:
        exact_matches = []
        partial_matches = []

        def callback(hwnd: int, _extra: object) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return
            title_lower = title.lower()
            if title_lower == self._title_lower:
                exact_matches.append((hwnd, title))
            elif self._title_lower in title_lower:
                partial_matches.append((hwnd, title))

        win32gui.EnumWindows(callback, None)
        matches = exact_matches or partial_matches
        if not matches:
            self.hwnd = None
            self.window_title = ""
            return None

        # Prefer the shortest matching title when several windows contain the keyword.
        hwnd, title = min(matches, key=lambda item: len(item[1]))
        self.hwnd = hwnd
        self.window_title = title
        return hwnd

    def _client_region(self) -> Optional[dict]:
        if not self._is_valid_window(self.hwnd):
            if self._find_window() is None:
                return None

        if win32gui.IsIconic(self.hwnd):
            return None

        try:
            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
            screen_left, screen_top = win32gui.ClientToScreen(
                self.hwnd, (left, top)
            )
            screen_right, screen_bottom = win32gui.ClientToScreen(
                self.hwnd, (right, bottom)
            )
        except Exception:
            self.hwnd = None
            return None

        width = int(screen_right - screen_left)
        height = int(screen_bottom - screen_top)
        if width < 16 or height < 16:
            return None

        return {
            "left": int(screen_left),
            "top": int(screen_top),
            "width": width,
            "height": height,
        }

    def read(self) -> tuple[bool, Optional[np.ndarray], bool]:
        """Return (ok, BGR frame, size_changed)."""
        region = self._client_region()
        if region is None:
            return False, None, False

        try:
            screenshot = np.asarray(self.sct.grab(region))
        except Exception:
            self.hwnd = None
            return False, None, False

        if screenshot.size == 0:
            return False, None, False

        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        current_size = (int(frame.shape[1]), int(frame.shape[0]))
        size_changed = current_size != self.last_size
        self.last_size = current_size
        return True, frame, size_changed

    def release(self) -> None:
        self.sct.close()

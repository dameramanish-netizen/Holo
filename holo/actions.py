"""Runs the side effect assigned to an accepted zone. Windows-first, but falls
back gracefully on macOS/Linux for local testing."""
from __future__ import annotations
import os
import platform
import shlex
import subprocess
import webbrowser

from .profile import ZoneActionConfiguration
from .zone import ZoneActionKind

IS_WINDOWS = platform.system() == "Windows"


def _clean_path(path: str) -> str:
    """Strips whitespace and surrounding quotes -- Windows' "Copy as path"
    wraps paths in literal double quotes, which os.startfile() then treats
    as part of the filename and fails to find."""
    return path.strip().strip('"').strip("'")


def run_action(action: ZoneActionConfiguration, on_status=None) -> tuple[bool, str]:
    """Executes the configured action. Returns (success, message). on_status(str),
    if given, also receives the same message for a UI log."""
    def status(msg):
        if on_status:
            on_status(msg)

    try:
        if action.kind == ZoneActionKind.NONE:
            msg = "Visual only -- no action configured for this zone."
            status(msg)
            return True, msg
        elif action.kind == ZoneActionKind.COPY_TEXT:
            if not action.text.strip():
                raise RuntimeError("No text configured to copy")
            _copy_text(action.text)
            msg = f"Copied: {action.text[:40]}"
        elif action.kind == ZoneActionKind.SPEAK_TEXT:
            if not action.text.strip():
                raise RuntimeError("No text configured to speak")
            _speak_text(action.text)
            msg = f"Spoke: {action.text[:40]}"
        elif action.kind == ZoneActionKind.OPEN_URL:
            url = action.text.strip()
            if not url:
                raise RuntimeError("No URL configured")
            if "://" not in url:
                url = "https://" + url
            opened = webbrowser.open(url)
            if not opened:
                raise RuntimeError(f"No browser handler available for {url}")
            msg = f"Opened {url}"
        elif action.kind == ZoneActionKind.RUN_SCRIPT:
            path = _clean_path(action.path or action.text)
            _run_script(path)
            msg = f"Ran script: {path}"
        elif action.kind == ZoneActionKind.OPEN_APPLICATION:
            path = _clean_path(action.path)
            _open_path(path)
            msg = f"Opened application: {path}"
        elif action.kind == ZoneActionKind.OPEN_ITEM:
            path = _clean_path(action.path)
            _open_path(path)
            msg = f"Opened: {path}"
        elif action.kind == ZoneActionKind.RUN_SHELL_COMMAND:
            if not action.text.strip():
                raise RuntimeError("No command configured")
            _run_shell(action.text)
            msg = f"Ran: {action.text[:40]}"
        elif action.kind == ZoneActionKind.SCREENSHOT_CLIPBOARD:
            _screenshot_to_clipboard()
            msg = "Screenshot copied to clipboard"
        else:
            msg = f"Unhandled action kind: {action.kind}"
            status(msg)
            return False, msg
        status(msg)
        return True, msg
    except Exception as exc:
        msg = f"Action failed: {exc}"
        status(msg)
        return False, msg


def _copy_text(text: str):
    try:
        import pyperclip
        pyperclip.copy(text)
    except ImportError:
        raise RuntimeError("pyperclip not installed (pip install pyperclip)")


def _speak_text(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        raise RuntimeError("pyttsx3 not installed (pip install pyttsx3)")


def _run_script(path: str):
    if not path:
        raise RuntimeError("No script path configured")
    if IS_WINDOWS:
        if path.lower().endswith(".ps1"):
            subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-File", path])
        else:
            subprocess.Popen(["cmd", "/c", "start", "", path], shell=False)
    else:
        subprocess.Popen(["/bin/sh", path])


def _open_path(path: str):
    if not path:
        raise RuntimeError("No path configured")
    if not os.path.exists(path):
        raise RuntimeError(f"Path does not exist: {path}")
    if IS_WINDOWS:
        os.startfile(path)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _run_shell(command: str):
    if not command.strip():
        raise RuntimeError("No command configured")
    if IS_WINDOWS:
        subprocess.Popen(["cmd", "/c", command], shell=False)
    else:
        subprocess.Popen(shlex.split(command))


def _screenshot_to_clipboard():
    from PIL import ImageGrab
    img = ImageGrab.grab()
    if IS_WINDOWS:
        import io
        import win32clipboard
        output = io.BytesIO()
        img.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]  # strip BMP file header for CF_DIB
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
    else:
        raise RuntimeError("Screenshot-to-clipboard currently requires Windows (pywin32)")

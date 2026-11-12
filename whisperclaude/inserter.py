"""Insert text into the focused app: clipboard save -> set -> Ctrl+V -> restore.

The clipboard is saved and restored with every format it holds (text, rich text, images,
copied files, an empty clipboard, ...), straight through the Win32 API. Saving only the text
(as pyperclip did) left the dictation in the clipboard whenever it held anything else, or
nothing at all.
"""
from __future__ import annotations

import contextlib
import ctypes
import logging
import time
from collections.abc import Callable, Iterator
from ctypes import wintypes

from pynput.keyboard import Controller, Key

log = logging.getLogger(__name__)

# Virtual-key codes: Shift, Ctrl, Alt, left/right Win
_MODIFIER_VKS = (0x10, 0x11, 0x12, 0x5B, 0x5C)

CF_UNICODETEXT = 13
# Formats whose data is a GDI handle, not memory we can copy. Windows synthesizes CF_BITMAP
# from the CF_DIB / CF_DIBV5 copy, so images still come back. 0x200-0x3FF: private formats.
_NOT_MEMORY = {2, 3, 9, 14, 0x80, 0x82, 0x83, 0x8E}
MAX_SAVE_BYTES = 1024 * 1024 * 1024 # formats beyond this aren't saved (memory), e.g. huge images
RESTORE_TIMEOUT_S = 3.0 # a busy clipboard is waited for longer when the old content goes back
GMEM_MOVEABLE = 0x0002
HWND_MESSAGE = wintypes.HWND(-3)

# Own DLL instances: argtypes set on ctypes.windll.user32 would be shared with other libraries.
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _api(dll: ctypes.WinDLL, name: str, restype: object, *argtypes: object) -> Callable[..., object]:
    fn = getattr(dll, name)
    fn.restype, fn.argtypes = restype, argtypes
    return fn


_OpenClipboard = _api(_user32, "OpenClipboard", wintypes.BOOL, wintypes.HWND)
_CloseClipboard = _api(_user32, "CloseClipboard", wintypes.BOOL)
_EmptyClipboard = _api(_user32, "EmptyClipboard", wintypes.BOOL)
_EnumClipboardFormats = _api(_user32, "EnumClipboardFormats", wintypes.UINT, wintypes.UINT)
_GetClipboardData = _api(_user32, "GetClipboardData", wintypes.HANDLE, wintypes.UINT)
_SetClipboardData = _api(_user32, "SetClipboardData", wintypes.HANDLE, wintypes.UINT,
                         wintypes.HANDLE)
_RegisterClipboardFormatW = _api(_user32, "RegisterClipboardFormatW", wintypes.UINT,
                                 wintypes.LPCWSTR)
_CreateWindowExW = _api(_user32, "CreateWindowExW", wintypes.HWND, wintypes.DWORD,
                        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                        wintypes.HINSTANCE, wintypes.LPVOID)
_DestroyWindow = _api(_user32, "DestroyWindow", wintypes.BOOL, wintypes.HWND)
_GetClipboardSequenceNumber = _api(_user32, "GetClipboardSequenceNumber", wintypes.DWORD)
_GlobalAlloc = _api(_kernel32, "GlobalAlloc", wintypes.HGLOBAL, wintypes.UINT, ctypes.c_size_t)
_GlobalLock = _api(_kernel32, "GlobalLock", wintypes.LPVOID, wintypes.HGLOBAL)
_GlobalUnlock = _api(_kernel32, "GlobalUnlock", wintypes.BOOL, wintypes.HGLOBAL)
_GlobalSize = _api(_kernel32, "GlobalSize", ctypes.c_size_t, wintypes.HGLOBAL)
_GlobalFree = _api(_kernel32, "GlobalFree", wintypes.HGLOBAL, wintypes.HGLOBAL)

# Dictations go into the local clipboard history (Win+V: a second safety net next to our own
# history file) but never into the cloud clipboard sync. The restored old clipboard is kept
# out of the history: it's in there already.
_NO_CLOUD = [(_RegisterClipboardFormatW("CanUploadToCloudClipboard"), (0).to_bytes(4, "little"))]
_NO_HISTORY = [(_RegisterClipboardFormatW("CanIncludeInClipboardHistory"),
                (0).to_bytes(4, "little"))] + _NO_CLOUD

Saved = list[tuple[int, bytes]] # (format, data) pairs; [] = the clipboard was empty


class ClipboardError(OSError):
    """The clipboard couldn't be opened (another app holds it) or written."""


@contextlib.contextmanager
def _opened(timeout: float = 1.0) -> Iterator[None]:
    """Open the clipboard with a hidden owner window (writing needs an owner), retrying
    while another app holds it."""
    hwnd = _CreateWindowExW(0, "STATIC", None, 0, 0, 0, 0, 0, HWND_MESSAGE, None, None, None)
    try:
        _open_clipboard(hwnd, timeout)
        try:
            yield
        finally:
            _CloseClipboard()
    finally:
        if hwnd:
            _DestroyWindow(hwnd)


def _open_clipboard(hwnd: int, timeout: float) -> None:
    """Opens the clipboard for hwnd, retrying until timeout; raises ClipboardError if busy."""
    deadline = time.monotonic() + timeout
    while not _OpenClipboard(hwnd):
        if time.monotonic() > deadline:
            raise ClipboardError(f"clipboard busy (error {ctypes.get_last_error()})")
        time.sleep(0.01)


def save_clipboard() -> Saved:
    """Copy every memory-based format currently in the clipboard."""
    saved: Saved = []
    total = 0
    with _opened():
        fmt = 0
        while fmt := _EnumClipboardFormats(fmt):
            if fmt in _NOT_MEMORY or 0x200 <= fmt <= 0x3FF:
                continue
            handle = _GetClipboardData(fmt)
            size = _GlobalSize(handle) if handle else 0
            if size and total + size > MAX_SAVE_BYTES:
                log.warning("clipboard format %d (%d MB) is too large to save, "
                            "it won't be restored", fmt, size >> 20)
            if not size or total + size > MAX_SAVE_BYTES:
                continue
            ptr = _GlobalLock(handle)
            if not ptr:
                continue
            try:
                saved.append((fmt, ctypes.string_at(ptr, size)))
            finally:
                _GlobalUnlock(handle)
            total += size
    return saved


def _write(items: Saved, timeout: float = 1.0) -> None:
    """Replace the clipboard with these formats (empty list = empty clipboard)."""
    with _opened(timeout):
        if not _EmptyClipboard():
            raise ClipboardError(f"EmptyClipboard failed (error {ctypes.get_last_error()})")
        for fmt, data in items:
            handle = _GlobalAlloc(GMEM_MOVEABLE, max(len(data), 1))
            ptr = _GlobalLock(handle) if handle else None
            if not ptr:
                raise ClipboardError("out of memory")
            ctypes.memmove(ptr, data, len(data))
            _GlobalUnlock(handle)
            if not _SetClipboardData(fmt, handle): # on success Windows owns the memory
                _GlobalFree(handle)


def restore_clipboard(saved: Saved) -> None:
    # the restore isn't a new history entry
    _write(saved + _NO_HISTORY if saved else [], RESTORE_TIMEOUT_S)


def _restore_quietly(saved: Saved) -> None:
    """Put the old clipboard back. A failure is logged, never raised: the caller has a result
    (or an error) of its own, and must not lose it over the restore."""
    try:
        restore_clipboard(saved)
    except ClipboardError:
        log.warning("could not restore the previous clipboard", exc_info=True)


def set_clipboard_text(text: str) -> None:
    _write([(CF_UNICODETEXT, text.encode("utf-16-le") + b"\0\0")] + _NO_CLOUD)


def clipboard_text() -> str | None:
    """The clipboard's text, or None if it holds none."""
    with _opened():
        handle = _GetClipboardData(CF_UNICODETEXT)
        ptr = _GlobalLock(handle) if handle else None
        if not ptr:
            return None
        try:
            return ctypes.wstring_at(ptr)
        finally:
            _GlobalUnlock(handle)


def _modifiers_down() -> bool:
    """True if the user is still physically holding a modifier key."""
    user32 = ctypes.windll.user32
    return any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in _MODIFIER_VKS)


def wait_for_modifiers_released(timeout: float = 2.0) -> None:
    """Wait until Ctrl/Alt/Shift/Win are up, so our Ctrl+V isn't Ctrl+Alt+V.

    We wait instead of sending fake key-ups: a synthetic Alt-up can open an
    app's menu bar.
    """
    deadline = time.monotonic() + timeout
    while _modifiers_down():
        if time.monotonic() > deadline:
            log.warning("modifier keys still held after %.1f s, pasting anyway", timeout)
            return
        time.sleep(0.02)


def to_clipboard_text(text: str) -> str:
    """Windows clipboard text uses CRLF; plain LF shows up as nothing or a box in some apps."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def copy_text(text: str) -> None:
    """Leave text in the clipboard for the user to paste (also in Win+V history, not cloud)."""
    set_clipboard_text(to_clipboard_text(text))


def _format_data(name: str) -> bytes | None:
    """Raw data of a registered clipboard format (clipboard must be open), None if absent."""
    handle = _GetClipboardData(_RegisterClipboardFormatW(name))
    size = _GlobalSize(handle) if handle else 0
    ptr = _GlobalLock(handle) if size else None
    if not ptr:
        return None
    try:
        return ctypes.string_at(ptr, size)
    finally:
        _GlobalUnlock(handle)


def _copy_selection(timeout: float) -> bool:
    """Send Ctrl+Insert. True once the clipboard changed, False after `timeout` without."""
    sequence = _GetClipboardSequenceNumber()
    wait_for_modifiers_released()
    kb = Controller()
    with kb.pressed(Key.ctrl):
        kb.tap(Key.insert)
    deadline = time.monotonic() + timeout
    while _GetClipboardSequenceNumber() == sequence:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    time.sleep(0.05) # apps may add more formats right after the first one
    return True


def _read_copied() -> tuple[str | None, bytes]:
    """The clipboard's text, and VS Code's data (it marks a copy without a selection)."""
    with _opened():
        handle = _GetClipboardData(CF_UNICODETEXT)
        ptr = _GlobalLock(handle) if handle else None
        text = None
        if ptr:
            try:
                text = ctypes.wstring_at(ptr)
            finally:
                _GlobalUnlock(handle)
        return text, _format_data("vscode-editor-data") or b""


def capture_selection(timeout: float = 0.6) -> str | None:
    """The text selected in the focused app, or None if nothing is selected.

    Saves the clipboard, sends Ctrl+Insert (copy, like Ctrl+C; but in a terminal without a
    selection Ctrl+C would interrupt the running program), reads the text and puts the old
    clipboard back, whatever happens in between. An app that copies nothing leaves the
    clipboard unchanged: no selection. If the old clipboard can't be saved (another app holds
    it), nothing is copied and ClipboardError is raised.
    VS Code copies the whole line when nothing is selected; it marks that, so it counts as no
    selection too (translating it would paste a translated copy of the line).
    """
    saved = save_clipboard()
    sequence = _GetClipboardSequenceNumber()
    try:
        if not _copy_selection(timeout):
            return None # nothing copied, the clipboard is untouched
        text, vscode = _read_copied()
    finally:
        if _GetClipboardSequenceNumber() != sequence: # the app copied, even if reading failed
            _restore_quietly(saved)
    if b'"isFromEmptySelection":true' in vscode.replace(b" ", b""):
        return None
    return text.replace("\r\n", "\n") if text else None


def _holds_other_text(ours: str) -> bool:
    """True if the clipboard holds anything but our text (or nothing). A clipboard that is
    busy right now counts as ours: a clipboard manager reading what we just put there."""
    try:
        return clipboard_text() != ours
    except ClipboardError:
        return False


def _restore_unless_copied(saved: Saved, ours: str, sequence: int | None) -> None:
    """Put the old clipboard back after a paste, unless the user copied something new since our
    text went in (sequence: the clipboard's change counter right after that; None: putting our
    text in failed, which may have emptied the clipboard). A failed restore isn't raised: the
    text is pasted already."""
    if (sequence is not None and _GetClipboardSequenceNumber() != sequence
            and _holds_other_text(ours)):
        log.info("clipboard changed since the paste (you copied something?), not restoring")
        return
    _restore_quietly(saved)


def paste_text(text: str, restore_delay: float = 0.3) -> None:
    """Paste text into whatever has focus, then put the previous clipboard back exactly, also
    if the paste fails. If the old clipboard can't be saved (another app holds it), nothing is
    pasted and ClipboardError is raised: what can't be put back is never overwritten."""
    if not text:
        return
    saved = save_clipboard()
    text = to_clipboard_text(text)
    sequence = None
    try:
        set_clipboard_text(text)
        sequence = _GetClipboardSequenceNumber()
        wait_for_modifiers_released()
        time.sleep(0.05)
        kb = Controller()
        with kb.pressed(Key.ctrl):
            kb.tap("v")
        # The target app reads the clipboard asynchronously; restoring too early
        # would paste the old content instead.
        time.sleep(restore_delay)
    finally:
        _restore_unless_copied(saved, text, sequence)

"""Read the text just before the caret of the focused control (Windows UI Automation).

Only the few characters before the caret are read, never the document, and they only go to the
caller's decision function: nothing is logged or stored here. No third-party COM library is
installed, so the handful of UIA methods needed are called through their vtables with ctypes.

UIA calls cross into the target app and can hang for seconds in a busy or odd app. So they run
on one long-lived daemon thread (COM initialised once, the IUIAutomation object created once),
and the caller waits at most `timeout` seconds. While a request is still unanswered, new ones
are refused (None) instead of queueing up behind it.

Result of `CaretReader.text_before(n)`:
    a str      the up to n characters before the caret ("" = the caret is at the very start)
    None       unknown: the app doesn't expose its text, it timed out, or anything went wrong
"""
from __future__ import annotations

import ctypes
import logging
import queue
import threading
from ctypes import wintypes

log = logging.getLogger(__name__)

# UIA ids and vtable slots (UIAutomationClient.h; IUnknown takes slots 0-2)
_TEXT_PATTERN, _TEXT_PATTERN2 = 10014, 10024
_GET_FOCUSED_ELEMENT = 8 # IUIAutomation
_GET_CURRENT_PATTERN = 16 # IUIAutomationElement
_TP_GET_SELECTION = 5 # IUIAutomationTextPattern
_TP2_GET_CARET_RANGE = 10 # IUIAutomationTextPattern2 (after the 7 slots of TextPattern + 1)
_ARRAY_LENGTH, _ARRAY_GET_ELEMENT = 3, 4 # IUIAutomationTextRangeArray
_RANGE_CLONE, _RANGE_GET_TEXT = 3, 12 # IUIAutomationTextRange
_RANGE_MOVE_ENDPOINT_BY_UNIT, _RANGE_MOVE_ENDPOINT_BY_RANGE = 14, 15
_ENDPOINT_START, _ENDPOINT_END, _UNIT_CHARACTER = 0, 1, 0
_RELEASE = 2

COINIT_MULTITHREADED = 0
CLSCTX_INPROC_SERVER = 1
_CLSID_UIAUTOMATION = "{ff48dba4-60ef-4201-aa87-54103eef594e}"
_IID_UIAUTOMATION = "{30cbe57d-d9d0-452a-ab13-7ac5ac4825ee}"


class _GUID(ctypes.Structure):
    _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD),
                ("d", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> _GUID:
        import uuid
        u = uuid.UUID(text)
        return cls.from_buffer_copy(u.bytes_le)


_ole32 = ctypes.WinDLL("ole32")
_oleaut32 = ctypes.WinDLL("oleaut32")
_ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
_ole32.CoInitializeEx.restype = ctypes.HRESULT
_ole32.CoCreateInstance.argtypes = [ctypes.POINTER(_GUID), ctypes.c_void_p, wintypes.DWORD,
                                    ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
_ole32.CoCreateInstance.restype = ctypes.HRESULT
_oleaut32.SysFreeString.argtypes = [ctypes.c_void_p]
_oleaut32.SysFreeString.restype = None


def _slot(obj: int, index: int, *argtypes: object):
    """The COM method at vtable slot `index` of `obj`, as a callable taking (*args)."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    fn = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[index])
    return lambda *args: fn(obj, *args)


def _release(obj: int | None) -> None:
    if obj:
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtable[_RELEASE])(obj)


def _out_ptr(call, *args) -> int | None:
    """Call a method whose last argument is an out pointer; return that pointer or None."""
    out = ctypes.c_void_p()
    call(*args, ctypes.byref(out))
    return out.value


def _range_before_caret(automation: int, n: int) -> str | None:
    """Worker-thread body: the up to n characters before the caret/selection start, or None."""
    owned: list[int] = []

    def keep(ptr: int | None) -> int | None:
        if ptr:
            owned.append(ptr)
        return ptr

    try:
        element = keep(_out_ptr(_slot(automation, _GET_FOCUSED_ELEMENT,
                                      ctypes.POINTER(ctypes.c_void_p))))
        if not element:
            return None
        get_pattern = _slot(element, _GET_CURRENT_PATTERN, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_void_p))
        caret = None
        pattern2 = keep(_out_ptr(get_pattern, _TEXT_PATTERN2))
        if pattern2: # TextPattern2: the caret itself (also in a terminal without a selection)
            get_caret = _slot(pattern2, _TP2_GET_CARET_RANGE, ctypes.POINTER(wintypes.BOOL),
                              ctypes.POINTER(ctypes.c_void_p))
            active, caret_out = wintypes.BOOL(), ctypes.c_void_p()
            try:
                get_caret(ctypes.byref(active), ctypes.byref(caret_out))
                caret = keep(caret_out.value)
            except OSError:
                caret = None
        if not caret:
            pattern = keep(_out_ptr(get_pattern, _TEXT_PATTERN))
            if not pattern:
                return None
            array = keep(_out_ptr(_slot(pattern, _TP_GET_SELECTION,
                                        ctypes.POINTER(ctypes.c_void_p))))
            length = ctypes.c_int()
            if not array:
                return None
            _slot(array, _ARRAY_LENGTH, ctypes.POINTER(ctypes.c_int))(ctypes.byref(length))
            if length.value < 1:
                return None
            caret = keep(_out_ptr(_slot(array, _ARRAY_GET_ELEMENT, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_void_p)), 0))
        if not caret:
            return None
        rng = keep(_out_ptr(_slot(caret, _RANGE_CLONE, ctypes.POINTER(ctypes.c_void_p))))
        if not rng:
            return None
        # collapse to the start (a selection is replaced by the paste), then extend n back
        _slot(rng, _RANGE_MOVE_ENDPOINT_BY_RANGE, ctypes.c_int, ctypes.c_void_p, ctypes.c_int)(
            _ENDPOINT_END, rng, _ENDPOINT_START)
        moved = ctypes.c_int()
        _slot(rng, _RANGE_MOVE_ENDPOINT_BY_UNIT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
              ctypes.POINTER(ctypes.c_int))(_ENDPOINT_START, _UNIT_CHARACTER, -n,
                                            ctypes.byref(moved))
        if moved.value == 0:
            return ""
        bstr = ctypes.c_void_p()
        _slot(rng, _RANGE_GET_TEXT, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(
            n, ctypes.byref(bstr))
        if not bstr.value:
            return ""
        try:
            return ctypes.wstring_at(bstr.value)
        finally:
            _oleaut32.SysFreeString(bstr.value)
    except OSError:
        return None # e.g. UIA_E_NOTSUPPORTED, or the element vanished
    finally:
        for ptr in reversed(owned):
            _release(ptr)


class CaretReader:
    """Asks UIA for the text before the caret on a private thread (see the module docstring)."""

    def __init__(self) -> None:
        self._requests: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._outstanding = False # a request is being served (maybe stuck in the app)
        self._failed = False

    def _run(self) -> None:
        try:
            _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
            automation = _out_ptr(
                lambda out: _ole32.CoCreateInstance(
                    ctypes.byref(_GUID.parse(_CLSID_UIAUTOMATION)), None, CLSCTX_INPROC_SERVER,
                    ctypes.byref(_GUID.parse(_IID_UIAUTOMATION)), out))
        except OSError:
            log.warning("UI Automation is not available, no text context for insertion")
            self._failed = True
            return
        while True:
            n, reply = self._requests.get()
            try:
                result = _range_before_caret(automation, n)
            except Exception: # never let the thread die: later requests would hang
                log.debug("caret read failed", exc_info=True)
                result = None
            reply.put(result)
            self._outstanding = False

    def start(self) -> None:
        """Start the thread now (loading UIA takes ~70 ms), so the first read isn't slow."""
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True, name="uia-caret")
                self._thread.start()

    def text_before(self, n: int = 1, timeout: float = 0.05) -> str | None:
        """Up to n characters before the caret (see module docstring). Never raises."""
        self.start()
        with self._lock:
            if self._failed or self._outstanding:
                return None
            self._outstanding = True
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._requests.put((n, reply))
        try:
            return reply.get(timeout=timeout)
        except queue.Empty:
            return None # the answer is dropped; _outstanding clears when the thread is free

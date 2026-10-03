"""Dictate into any window: press a key, talk, press it again, and the text is typed for you.

```mermaid
sequenceDiagram
    participant K as GNOME shortcut
    participant T as dictate toggle
    participant D as daemon (model loaded)
    participant M as parecord (mic)
    participant X as xdotool
    K->>T: key press
    T->>D: {"cmd": "toggle"} over a unix socket
    D->>M: start recording (16 kHz mono)
    K->>T: key press again
    T->>D: toggle
    D->>M: stop
    D->>D: hear → clean (dictate/pipeline.py)
    D->>X: type the text into the focused window
```

**Why a daemon.** Loading the recogniser, a 3.6 GB word LM and the tagger takes seconds; a
hotkey that took seconds to *start listening* would lose the first words of every sentence.
The daemon loads once and waits. `toggle` is a tiny client that only talks to the socket, so
the key press itself costs a Python start-up and nothing else.

**Why toggle, not hold-to-talk.** A GNOME custom shortcut runs a command on press and has no
release event. Toggle also means a long paragraph does not need a held key. `max_seconds`
stops a recording whose second press never came.

**The desktop pieces are other programs, deliberately:** `parecord` (PulseAudio/PipeWire) for
the microphone, `xdotool` to type, `xclip` for the clipboard, `notify-send` to say what is
happening. Each is optional except the microphone; whatever is missing is reported by
`dictate status` with the line that installs it, rather than discovered at the first key press.
X11 only — Wayland does not let one program type into another's window this way.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .pipeline import SAMPLE_RATE, Dictator, Settings

SOCKET_NAME = "daemon.sock"
PID_NAME = "daemon.pid"
#: The tools the desktop path uses, what each is for, and how to get it on Ubuntu.
TOOLS = {
    "parecord": ("record the microphone", "sudo apt install pulseaudio-utils"),
    "xdotool": ("type the text into the focused window", "sudo apt install xdotool"),
    "xclip": ("put the text on the clipboard", "sudo apt install xclip"),
    "notify-send": ("show 'listening…' and the result", "sudo apt install libnotify-bin"),
}


def tools() -> dict:
    return {name: {"found": shutil.which(name) is not None, "for": why, "install": how}
            for name, (why, how) in TOOLS.items()}


def state_dir(root: Path, settings: Settings) -> Path:
    d = root / settings.personal_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def notify(title: str, body: str = "") -> None:
    if shutil.which("notify-send"):
        # One notification that updates in place, not a stack of them.
        subprocess.run(["notify-send", "-a", "aksharallm dictation", "-t", "4000",
                        "-h", "string:x-canonical-private-synchronous:aksharallm-dictate",
                        title, body], check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def deliver(text: str, how: str) -> str:
    """Hand `text` to the desktop. Returns what actually happened, which is not always `how`."""
    if not text or how == "none":
        return "none"
    if how in ("paste", "clipboard") and shutil.which("xclip"):
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=False)
        if how == "clipboard":
            return "clipboard"
        if shutil.which("xdotool"):
            subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], check=False)
            return "paste"
        return "clipboard"
    if shutil.which("xdotool"):
        # --clearmodifiers: the shortcut's own Super/Alt may still be held as typing starts.
        subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "4", "--", text],
                       check=False)
        return "type"
    return "none (install xdotool to type, xclip for the clipboard)"


class Recorder:
    """parecord writing raw 16-bit mono to a file; stopped with SIGINT."""

    def __init__(self, path: Path):
        self.path = path
        self.proc: subprocess.Popen | None = None
        self.started = 0.0

    def start(self) -> None:
        if not shutil.which("parecord"):
            raise RuntimeError("parecord not found: " + TOOLS["parecord"][1])
        self.fh = open(self.path, "wb")
        self.proc = subprocess.Popen(
            ["parecord", "--raw", f"--rate={SAMPLE_RATE}", "--channels=1", "--format=s16le",
             "--latency-msec=50"], stdout=self.fh, stderr=subprocess.DEVNULL)
        self.started = time.time()

    @property
    def active(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> np.ndarray:
        if self.proc is not None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.fh.close()
        self.proc = None
        raw = self.path.read_bytes()
        return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


class Daemon:
    def __init__(self, root: Path, settings: Settings):
        self.root = root
        self.s = settings
        self.dir = state_dir(root, settings)
        self.dictator = Dictator(settings, root)
        self.rec = Recorder(self.dir / "recording.raw")
        self.lock = threading.Lock()
        self.state = "loading"
        self.last: dict | None = None
        self._timer: threading.Timer | None = None
        self._preview: threading.Thread | None = None

    def load(self) -> None:
        t0 = time.time()
        self.dictator.recognizer()
        self.dictator.lm()
        self.dictator.punctuator()
        # Build the beam search's word-prefix table now (~9 s for 200k words): otherwise the
        # first dictation after start-up pays for it and reads as a hung hotkey.
        self.dictator._decoder()
        self.state = "idle"
        print(f"loaded in {time.time() - t0:.1f}s: {json.dumps(self.dictator.describe())}", flush=True)

    # ---- commands -------------------------------------------------------------------------

    def start(self) -> dict:
        with self.lock:
            if self.state != "idle":
                return {"ok": False, "state": self.state}
            self.rec.start()
            self.state = "recording"
            self._timer = threading.Timer(self.s.max_seconds, self._timeout)
            self._timer.daemon = True
            self._timer.start()
            self._preview = threading.Thread(target=self._live, daemon=True)
            self._preview.start()
        notify("Listening…", "press the shortcut again to finish")
        return {"ok": True, "state": "recording"}

    def _live(self) -> None:
        """The live preview (dictate/stream.py) in the notification, while recording.

        Shown in the notification and NOT typed: typed text would have to be deleted and
        retyped when the final pass (beam, dictionary, cleanup) differs from the preview —
        and it does, by design. Reads parecord's file as it grows."""
        from .stream import LiveTranscript
        lt = LiveTranscript(self.dictator.recognizer(), self.s.device)
        shown = ""
        while self.state == "recording":
            time.sleep(0.7)
            try:
                raw = self.rec.path.read_bytes()
            except OSError:
                continue
            lt.audio = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
            r = lt.tick()
            text = (r["stable"] + (" " + r["tentative"] if r["tentative"] else "")).strip()
            if text and text != shown and self.state == "recording":
                notify("Listening…", text[-200:])
                shown = text

    def _timeout(self) -> None:
        if self.state == "recording":
            notify("Stopped", f"{self.s.max_seconds:.0f} s is the limit")
            self.stop()

    def stop(self, deliver_text: bool = True) -> dict:
        with self.lock:
            if self.state != "recording":
                return {"ok": False, "state": self.state}
            if self._timer:
                self._timer.cancel()
            x = self.rec.stop()
            self.state = "thinking"
        try:
            notify("Writing…")
            rec = self.dictator.dictate(x, source="hotkey")
            how = deliver(rec["text"], self.s.output) if deliver_text else "none"
            rec["delivered"] = how
            self.last = rec
            if rec["text"]:
                notify("Done" if how in ("type", "paste") else "Copied" if how == "clipboard" else "Heard",
                       rec["text"][:200])
            else:
                notify("Nothing heard", rec.get("note", ""))
            return {"ok": True, **rec}
        finally:
            self.state = "idle"

    def toggle(self) -> dict:
        return self.start() if self.state == "idle" else self.stop()

    def cancel(self) -> dict:
        with self.lock:
            if self.state == "recording":
                if self._timer:
                    self._timer.cancel()
                self.rec.stop()
                self.state = "idle"
                notify("Cancelled")
                return {"ok": True, "state": "idle"}
        return {"ok": False, "state": self.state}

    def status(self) -> dict:
        return {"ok": True, "state": self.state, "pid": os.getpid(),
                "recording_s": round(time.time() - self.rec.started, 1) if self.state == "recording" else None,
                "last": self.last, "settings": self.s.__dict__}

    # ---- the socket ---------------------------------------------------------------------

    def serve(self) -> None:
        sock_path = self.dir / SOCKET_NAME
        pid_path = self.dir / PID_NAME
        if running(self.root, self.s):
            raise SystemExit("a dictation daemon is already running (python -m aksharallm.dictate status)")
        sock_path.unlink(missing_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        os.chmod(sock_path, 0o600)    # only this user may make it listen
        srv.listen(4)
        pid_path.write_text(str(os.getpid()))
        threading.Thread(target=self.load, daemon=True).start()
        stop = threading.Event()

        def bye(*_):
            stop.set()
            srv.close()
        signal.signal(signal.SIGTERM, bye)
        signal.signal(signal.SIGINT, bye)
        print(f"listening on {sock_path}", flush=True)
        try:
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except OSError:
                    break
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            if self.rec.active:
                self.rec.stop()
            sock_path.unlink(missing_ok=True)
            if pid_path.is_file() and pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                req = json.loads(conn.makefile().readline() or "{}")
                cmd = req.get("cmd")
                if self.state == "loading" and cmd not in ("status",):
                    resp = {"ok": False, "state": "loading", "error": "still loading the models"}
                    notify("Still loading", "try again in a few seconds")
                else:
                    fn = {"toggle": self.toggle, "start": self.start, "stop": self.stop,
                          "cancel": self.cancel, "status": self.status}.get(cmd)
                    resp = fn() if fn else {"ok": False, "error": f"unknown command {cmd!r}"}
            except Exception as e:   # noqa: BLE001 -- a daemon must answer, not die
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.state = "idle" if self.state != "loading" else self.state
                notify("Dictation failed", str(e)[:200])
            conn.sendall((json.dumps(resp) + "\n").encode())


def running(root: Path, settings: Settings) -> int | None:
    pid_path = state_dir(root, settings) / PID_NAME
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError):
        return None
    return pid if "aksharallm.dictate" in cmd else None


def send(root: Path, settings: Settings, cmd: str, timeout: float = 180.0) -> dict:
    """One request to the daemon. The timeout covers a long recording being transcribed."""
    path = state_dir(root, settings) / SOCKET_NAME
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(path))
    except OSError:
        return {"ok": False, "error": "no dictation daemon is running — start one: "
                                      "python -m aksharallm.dictate daemon --bg"}
    with s:
        s.sendall((json.dumps({"cmd": cmd}) + "\n").encode())
        return json.loads(s.makefile().readline() or "{}")


# ---------------------------------------------------------------------------------------
# the GNOME shortcut
# ---------------------------------------------------------------------------------------

_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
_KEY_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
_PATH = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/aksharallm-dictate/"


def _gsettings_bin() -> str:
    """The SYSTEM gsettings. Conda (and some venvs) ship their own copy, built without the
    dconf backend: it falls back to an in-memory store, so every `set` succeeds and is
    forgotten when the process exits. With conda's `base` active that copy comes first on
    PATH — which is how this installer once reported "installed" for a shortcut GNOME never
    saw. Prefer the distribution's binary; never trust a write without reading it back."""
    for cand in ("/usr/bin/gsettings", "/bin/gsettings"):
        if Path(cand).is_file():
            return cand
    found = shutil.which("gsettings")
    if not found:
        raise RuntimeError("gsettings not found — is this GNOME?")
    return found


def _gsettings(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("GSETTINGS_BACKEND", "GIO_MODULE_DIR")}
    return subprocess.run([_gsettings_bin(), *args], capture_output=True, text=True, check=True,
                          env=env).stdout.strip()


def _check(schema: str, key: str, want: str) -> None:
    got = _gsettings("get", schema, key)
    if got.strip("'") != want.strip("'"):
        raise RuntimeError(f"GNOME did not keep {key} (wrote {want!r}, read back {got!r}) — "
                           f"{_gsettings_bin()} is not talking to your desktop's settings")


def _paths(text: str) -> list[str]:
    """gsettings prints an empty string list as `@as []`, a typed GVariant, not Python."""
    import ast
    text = text.strip()
    return [] if not text or text.startswith("@as") else list(ast.literal_eval(text))


def install_shortcut(command: str, binding: str) -> str:
    """Add (or update) one GNOME custom shortcut. Other custom shortcuts are left alone: the
    list is read, our path appended if missing, and written back."""
    cur = _gsettings("get", _SCHEMA, "custom-keybindings")
    paths = _paths(cur)
    if _PATH not in paths:
        paths.append(_PATH)
        _gsettings("set", _SCHEMA, "custom-keybindings", str(paths))
        if _PATH not in _paths(_gsettings("get", _SCHEMA, "custom-keybindings")):
            raise RuntimeError(f"GNOME did not keep the shortcut list — {_gsettings_bin()} is "
                               "not talking to your desktop's settings")
    rel = f"{_KEY_SCHEMA}:{_PATH}"
    for key, value in (("name", "aksharallm dictation"), ("command", command), ("binding", binding)):
        _gsettings("set", rel, key, value)
        _check(rel, key, value)
    return binding


def uninstall_shortcut() -> bool:
    cur = _gsettings("get", _SCHEMA, "custom-keybindings")
    paths = _paths(cur)
    if _PATH not in paths:
        return False
    paths.remove(_PATH)
    _gsettings("set", _SCHEMA, "custom-keybindings", str(paths) if paths else "@as []")
    _gsettings("reset-recursively", f"{_KEY_SCHEMA}:{_PATH}")
    return True


def shortcut() -> dict | None:
    try:
        cur = _gsettings("get", _SCHEMA, "custom-keybindings")
    except (OSError, subprocess.CalledProcessError):
        return None
    paths = _paths(cur)
    if _PATH not in paths:
        return None
    rel = f"{_KEY_SCHEMA}:{_PATH}"
    return {"binding": _gsettings("get", rel, "binding").strip("'"),
            "command": _gsettings("get", rel, "command").strip("'")}

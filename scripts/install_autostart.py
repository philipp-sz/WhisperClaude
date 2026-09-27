"""Start WhisperClaude silently at Windows login (shortcut in the Startup folder).

.venv\\Scripts\\python.exe scripts\\install_autostart.py            install (and start now)
.venv\\Scripts\\python.exe scripts\\install_autostart.py --remove   uninstall
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
STARTUP = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
LINK = STARTUP / "WhisperClaude.lnk"


def create_shortcut():
    # WScript.Shell is built into Windows; avoids a pywin32 dependency.
    ps = f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{LINK}')
$s.TargetPath = '{PYTHONW}'
$s.Arguments = '-m whisperclaude'
$s.WorkingDirectory = '{ROOT}'
$s.Description = 'WhisperClaude dictation'
$s.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--no-start", action="store_true", help="don't start the app now")
    args = ap.parse_args()

    if args.remove:
        if LINK.exists():
            LINK.unlink()
            print(f"Removed {LINK}")
        else:
            print("Autostart was not installed.")
        print("A running instance keeps running: tray icon -> Quit.")
        return

    if not PYTHONW.exists():
        sys.exit(f"{PYTHONW} not found. Create the venv first.")
    create_shortcut()
    print(f"Installed: {LINK}")
    print(f"  -> {PYTHONW} -m whisperclaude")
    if not args.no_start:
        # Detached from this console, so closing the terminal doesn't end the app.
        # (A second instance exits by itself if one is already running.)
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(PYTHONW), "-m", "whisperclaude"], cwd=ROOT, creationflags=flags,
                         close_fds=True)
        print("Started. Look for the green microphone in the tray.")


if __name__ == "__main__":
    main()

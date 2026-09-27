"""Start WhisperClaude silently at Windows login.

.venv\\Scripts\\python.exe scripts\\install_autostart.py            install (and start now)
.venv\\Scripts\\python.exe scripts\\install_autostart.py --remove   uninstall

Uses a Task Scheduler logon task: it starts right at login, while Startup-folder shortcuts
are held back by Windows until the desktop has settled (took ~3 min here). Falls back to
a Startup-folder shortcut if the task can't be registered.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
TASK = "WhisperClaude"
STARTUP = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
LINK = STARTUP / "WhisperClaude.lnk"


def powershell(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell", "-NoProfile", "-Command", script],
                          capture_output=True, text=True)


def register_task() -> bool:
    # Defaults that would break a tray app are overridden: tasks normally don't start on
    # battery, stop after 72 h, and run at below-normal priority (7).
    r = powershell(f"""
$user = "$env:USERDOMAIN\\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute '{PYTHONW}' -Argument '-m whisperclaude' -WorkingDirectory '{ROOT}'
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -Priority 5
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '{TASK}' -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description 'WhisperClaude dictation' -Force | Out-Null
""")
    if r.returncode != 0:
        print("Task Scheduler registration failed:", r.stderr.strip().splitlines()[0] if r.stderr else "")
    return r.returncode == 0


def create_shortcut() -> None:
    r = powershell(f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{LINK}')
$s.TargetPath = '{PYTHONW}'
$s.Arguments = '-m whisperclaude'
$s.WorkingDirectory = '{ROOT}'
$s.Save()
""")
    r.check_returncode()


def remove() -> None:
    r = powershell(f"Unregister-ScheduledTask -TaskName '{TASK}' -Confirm:$false")
    print("Removed logon task." if r.returncode == 0 else "No logon task installed.")
    if LINK.exists():
        LINK.unlink()
        print(f"Removed {LINK}")
    print("A running instance keeps running: tray icon -> Quit.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--no-start", action="store_true", help="don't start the app now")
    args = ap.parse_args()
    if args.remove:
        return remove()

    if not PYTHONW.exists():
        sys.exit(f"{PYTHONW} not found. Create the venv first.")
    if register_task():
        if LINK.exists():  # replaced by the task
            LINK.unlink()
        print(f"Installed logon task '{TASK}' -> {PYTHONW.name} -m whisperclaude")
    else:
        create_shortcut()
        print(f"Installed Startup-folder shortcut instead: {LINK}")
    if not args.no_start:
        # Detached from this console, so closing the terminal doesn't end the app.
        # (A second instance exits by itself if one is already running.)
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(PYTHONW), "-m", "whisperclaude"], cwd=ROOT, creationflags=flags,
                         close_fds=True)
        print("Started. Look for the green microphone in the tray.")


if __name__ == "__main__":
    main()

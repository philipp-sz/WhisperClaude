"""Start WhisperClaude silently at Windows login, and bring it back after sleep or a lock.

.venv\\Scripts\\python.exe scripts\\install_autostart.py            install (and start now)
.venv\\Scripts\\python.exe scripts\\install_autostart.py --remove   uninstall

Two Task Scheduler tasks (no admin rights needed):
- "WhisperClaude": at logon. Task Scheduler starts it right at login, while Startup-folder
  shortcuts are held back by Windows until the desktop has settled (took ~3 min here).
- "WhisperClaude Revive": when the PC wakes from sleep or the session is unlocked. Windows can
  close all user programs at a failed shutdown attempt without ending the session, so the
  logon task never fires again. This task starts the app again ("--revive": does nothing if it
  is already running, or if you chose Quit in the tray).
Falls back to a Startup-folder shortcut if the logon task can't be registered.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
TASK = "WhisperClaude"
REVIVE_TASK = "WhisperClaude Revive"
STARTUP = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
LINK = STARTUP / "WhisperClaude.lnk"

# Modern standby (this laptop) logs Kernel-Power 507 on wake; classic sleep logs 107 or
# Power-Troubleshooter 1.
WAKE_QUERY = (
    '<QueryList><Query Id="0" Path="System"><Select Path="System">'
    "*[System[(Provider[@Name='Microsoft-Windows-Kernel-Power'] and (EventID=107 or EventID=507))"
    " or (Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1)]]"
    "</Select></Query></QueryList>"
)


def powershell(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell", "-NoProfile", "-Command", script],
                          capture_output=True, text=True)


def task_xml(triggers: str, arguments: str, description: str) -> str:
    # Defaults that would break a tray app are overridden: tasks normally don't start on battery,
    # stop after 72 h, and run at below-normal priority (7).
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>{escape(description)}</Description></RegistrationInfo>
  <Triggers>{triggers}</Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{{USER}}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(PYTHONW))}</Command>
      <Arguments>{escape(arguments)}</Arguments>
      <WorkingDirectory>{escape(str(ROOT))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def logon_xml() -> str:
    return task_xml(
        "<LogonTrigger><Enabled>true</Enabled><UserId>{USER}</UserId></LogonTrigger>",
        "-m whisperclaude --autostart", "WhisperClaude dictation: start at login")


def revive_xml() -> str:
    return task_xml(
        "<SessionStateChangeTrigger><Enabled>true</Enabled><StateChange>SessionUnlock</StateChange>"
        "<UserId>{USER}</UserId></SessionStateChangeTrigger>"
        f"<EventTrigger><Enabled>true</Enabled><Delay>PT3S</Delay>"
        f"<Subscription>{escape(WAKE_QUERY)}</Subscription></EventTrigger>",
        "-m whisperclaude --revive", "WhisperClaude dictation: start again after sleep or unlock")


def register(name: str, xml: str) -> bool:
    xml_path = Path(os.environ.get("TEMP", ROOT)) / f"{name.replace(' ', '_')}.xml"
    xml_path.write_text(xml, encoding="utf-16")  # Task Scheduler wants UTF-16 with a BOM
    try:
        r = powershell(f"""
$user = "$env:USERDOMAIN\\$env:USERNAME"
$xml = (Get-Content -Raw -Encoding Unicode '{xml_path}').Replace('{{USER}}', $user)
Register-ScheduledTask -TaskName '{name}' -Xml $xml -Force | Out-Null
""")
    finally:
        xml_path.unlink(missing_ok=True)
    if r.returncode != 0:
        first = (r.stderr or "").strip().splitlines()
        print(f"Task Scheduler registration of '{name}' failed:", first[0] if first else "")
    return r.returncode == 0


def create_shortcut() -> None:
    r = powershell(f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{LINK}')
$s.TargetPath = '{PYTHONW}'
$s.Arguments = '-m whisperclaude --autostart'
$s.WorkingDirectory = '{ROOT}'
$s.Save()
""")
    r.check_returncode()


def remove() -> None:
    for name in (TASK, REVIVE_TASK):
        r = powershell(f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false")
        print(f"Removed task '{name}'." if r.returncode == 0 else f"Task '{name}' was not installed.")
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
    if register(TASK, logon_xml()):
        if LINK.exists():  # replaced by the task
            LINK.unlink()
        print(f"Installed logon task '{TASK}' -> {PYTHONW.name} -m whisperclaude --autostart")
    else:
        create_shortcut()
        print(f"Installed Startup-folder shortcut instead: {LINK}")
    if register(REVIVE_TASK, revive_xml()):
        print(f"Installed task '{REVIVE_TASK}' (after wake from sleep and unlock)")
    else:
        print("The app will not come back by itself after a failed shutdown; start it by hand "
              "or sign out and in again.")
    if not args.no_start:
        # Detached from this console, so closing the terminal doesn't end the app.
        # (A second instance exits by itself if one is already running.)
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(PYTHONW), "-m", "whisperclaude"], cwd=ROOT, creationflags=flags,
                         close_fds=True)
        print("Started. Look for the green microphone in the tray.")


if __name__ == "__main__":
    main()

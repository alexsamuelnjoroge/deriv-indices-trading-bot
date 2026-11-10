"""
Windowless launcher for Task Scheduler: pythonw.exe pro_bot/run_hidden.py <module> [args...]

Sends stdout/stderr to logs/<module>_console.log and refuses to start a second
copy of the same module (a duplicate incubator or pro_bot would double-trade).
"""

import ctypes
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]

(ROOT / "logs").mkdir(exist_ok=True)
log = open(ROOT / "logs" / f"{module.replace('.', '_')}_console.log", "a", buffering=1, encoding="utf-8")
sys.stdout = sys.stderr = log

ERROR_ALREADY_EXISTS = 183
mutex = ctypes.windll.kernel32.CreateMutexW(None, False, f"Local\\trading_bot_{module}")
if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
    print(f"{module} is already running — not starting a second copy", flush=True)
    sys.exit(0)

runpy.run_module(module, run_name="__main__", alter_sys=True)

"""Run every numbered pipeline script (src/0*_*.py) in sorted order."""
import runpy
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
for script in sorted(SRC.glob("0*_*.py")):
    print(f"\n===== {script.name} =====", flush=True)
    t0 = time.time()
    runpy.run_path(str(script), run_name="__main__")
    print(f"[{script.name} done in {time.time() - t0:.1f}s]", flush=True)

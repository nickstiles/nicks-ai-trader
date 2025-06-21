import subprocess
import json
import os
import sys

# compute your project root (one level up from models/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

hold_days_list = [1, 2, 3, 5, 7]
results = []

for hd in hold_days_list:
    print(f"\n▶️  Sweeping hold_days = {hd}")
    cmd = [
        sys.executable,
        "-m", "models.ml_model",
        f"--hold_days={hd}",
        "--retrain"
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    # 1) check for errors
    if completed.returncode != 0:
        print(f"[ERROR] hold_days={hd} failed:")
        print(completed.stderr.strip())
        continue

    # 2) pull out any non-blank lines of stdout
    lines = [l for l in completed.stdout.splitlines() if l.strip()]
    if not lines:
        print(f"[WARN] hold_days={hd} produced no stdout.")
        continue

    # 3) parse the last line as JSON
    last = lines[-1]
    try:
        summary = json.loads(last)
    except json.JSONDecodeError:
        print(f"[WARN] hold_days={hd} last line not JSON:\n  {last}")
        continue

    results.append(summary)

print("\n🏁 Sweep results:")
print(json.dumps(results, indent=2))
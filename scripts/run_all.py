"""Run the whole study: unit tests + all demo scripts.

    python3 scripts/run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = [
    "01_forward_model.py",
    "02_phaseshift_pipeline.py",
    "03_fourier_mode.py",
    "04_lm_inverse.py",
    "05_error_analysis.py",
    "06_freeform_wavefront.py",
]


def main() -> int:
    print("=" * 78)
    print("unit tests")
    print("=" * 78)
    rc = subprocess.call([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
    if rc != 0:
        print("tests failed")
        return rc
    for name in SCRIPTS:
        print("\n" + "=" * 78)
        print(f"script {name}")
        print("=" * 78)
        rc = subprocess.call([sys.executable, os.path.join("scripts", name)], cwd=ROOT)
        if rc != 0:
            print(f"{name} failed")
            return rc
    print("\nall good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Diffraction comparison; explicitly choose a sampling profile.

    uv run python experiment.py --profile test --output output/diffraction/smoke

The earlier five-beam demonstration is in experiments/legacy_five_beam.py.
"""
import sys
from lsi.study import main

if __name__ == '__main__':
    main(['compare', *sys.argv[1:]])

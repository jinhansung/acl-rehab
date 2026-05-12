"""
Run all persona traces as a standalone script.

Usage:
    python tests/persona_traces/run_all.py           # default: all three
    python tests/persona_traces/run_all.py --maya    # single persona
    python tests/persona_traces/run_all.py -v        # verbose pytest output

Exit code 0 = all passed, non-zero = failures (same as pytest).
"""
from __future__ import annotations

import sys
import os

# Allow running as `python tests/persona_traces/run_all.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import pytest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run persona trace tests.")
    parser.add_argument("--maya",   action="store_true", help="Run only Maya trace")
    parser.add_argument("--helen",  action="store_true", help="Run only Helen trace")
    parser.add_argument("--sophia", action="store_true", help="Run only Sophia trace")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    args, extra = parser.parse_known_args()

    here = os.path.dirname(__file__)

    if args.maya:
        targets = [os.path.join(here, "maya.py")]
    elif args.helen:
        targets = [os.path.join(here, "helen.py")]
    elif args.sophia:
        targets = [os.path.join(here, "sophia.py")]
    else:
        targets = [
            os.path.join(here, "maya.py"),
            os.path.join(here, "helen.py"),
            os.path.join(here, "sophia.py"),
        ]

    pytest_args = targets + (["-v"] if args.verbose else []) + extra
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())

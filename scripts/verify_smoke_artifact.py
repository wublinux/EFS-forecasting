"""Command-line wrapper for the synthetic smoke artifact verifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptforecast.verification import verify_smoke_artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    verified = verify_smoke_artifact(parser.parse_args().artifact_root)
    print(f"Verified smoke artifact: {verified}")

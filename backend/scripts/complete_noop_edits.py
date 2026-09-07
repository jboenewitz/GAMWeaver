#!/usr/bin/env python3
"""Create a completed IGANN export with explicit untouched-chart submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.noop_edits import complete_shape_function_edits_export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    artifact["shape_function_edits_export"] = complete_shape_function_edits_export(
        artifact.get("shape_function_edits_export", {}),
        artifact.get("shape_functions", []),
    )
    artifact["artifact_version"] = "1.2"
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Create a validated manifest for an extracted MSD task (no download happens here).

    python scripts/import_msd.py ../data/external/msd/Task06_Lung --manifest-id msd-task06-lung

The manifest is written to data/external/<manifest-id>/manifest.json and lists
every training case with its split, geometry and quality checks. See
docs/DATASETS.md for download steps and licence terms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova.config import get_settings  # noqa: E402
from onconova.data.msd import build_msd_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--manifest-id", default="msd-task06-lung")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_msd_manifest(args.task_dir, manifest_id=args.manifest_id, split_seed=args.split_seed)
    out = get_settings().data_dir / "external" / args.manifest_id / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"{manifest['n_patients']} cases ({manifest['scan_counts_by_split']}), {len(manifest['excluded'])} excluded -> {out}")


if __name__ == "__main__":
    main()

"""Evaluate every frozen v1.1 checkpoint on the predeclared real subset."""

import argparse

from robotics_vision.real_evaluation import run_real_study

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="docs/evidence/v1_1/study.json")
parser.add_argument("--manifest", default="data/external/coco_v1_2/manifest.json")
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
report = run_real_study(args.source, args.manifest, args.output_dir)
print({"evaluations": len(report["records"]), "failed": report["failed_evaluations"]})
if report["failed_evaluations"]:
    raise SystemExit(1)

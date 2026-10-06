"""Download the predeclared official COCO real-photo evaluation subset."""

import argparse

from robotics_vision.real_data import prepare_real_dataset, validate_real_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="data/external/coco_v1_2")
parser.add_argument("--reference-manifest", default="docs/evidence/v1_2/manifest.json")
args = parser.parse_args()
path = prepare_real_dataset(args.root)
print(path, validate_real_dataset(path, args.reference_manifest))

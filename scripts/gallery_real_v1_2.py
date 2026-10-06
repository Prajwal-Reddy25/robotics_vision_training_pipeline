"""Deterministic successes/failures; source photo attribution is retained separately."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from robotics_vision.benchmarking import _sha256
from robotics_vision.models import build_model
from robotics_vision.real_evaluation import (
    binary_metrics,
    letterbox,
    load_cases,
    native_confusions,
    restore_prediction,
)
from robotics_vision.runlog import write_json
from robotics_vision.training import load_checkpoint

# Correct a thumbnail ID transcription, not a tag or a model-dependent selection.
TAG_ID_CORRECTIONS = {343468: 343466}
PALETTE = np.array([[55, 170, 85], [100, 110, 120], [245, 195, 35], [150, 60, 185]], np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="runs/v1_2_real/study.json")
    parser.add_argument("--manifest", default="data/external/coco_v1_2/manifest.json")
    parser.add_argument("--tags", default="docs/evidence/v1_2/input_tags.json")
    parser.add_argument("--output", default="runs/v1_2_real/gallery")
    parser.add_argument("--credits", help="Reuse recorded public attribution; no network lookup")
    args = parser.parse_args()
    cached_credits = (
        {r["coco_id"]: r for r in json.loads(Path(args.credits).read_text())}
        if args.credits
        else {}
    )
    study, tags = [json.loads(Path(p).read_text()) for p in (args.study, args.tags)]
    if tags["manifest_sha256"] != _sha256(Path(args.manifest)):
        raise ValueError("Input tag source changed")
    manifest, cases = load_cases(Path(args.manifest))
    by_id = {c["id"]: c for c in cases}
    roles, models = {}, {}
    torch.set_num_threads(1)
    for arch in ("tiny_unet", "lite_fcn"):
        baseline = next(
            r
            for r in study["records"]
            if r["architecture"] == arch
            and r["arm"] == "baseline"
            and r["seed"] == 42
            and r["size"] == 64
        )
        ranked = sorted(baseline["measured"]["per_image"], key=lambda x: (x["floor_iou"], x["id"]))
        for role, index in [("worst", 0), ("lower_median", 39), ("relatively_best", 79)]:
            roles.setdefault(ranked[index]["id"], []).append(f"{arch}:{role}")
        if arch == "tiny_unet":
            lookup = {r["id"]: r for r in ranked}
            for tag, ids in tags["tags"].items():
                ids = [TAG_ID_CORRECTIONS.get(i, i) for i in ids]
                if not set(ids) <= by_id.keys():
                    raise ValueError("Qualitative tag refers to an absent source image")
                selected = min(ids, key=lambda i: (lookup[i]["floor_iou"], i))
                roles.setdefault(selected, []).append(f"pretag:{tag}:lower_floor_iou")
        for arm in ("baseline", "occlusion"):
            record = next(
                r
                for r in study["records"]
                if r["architecture"] == arch
                and r["arm"] == arm
                and r["seed"] == 42
                and r["size"] == 64
            )
            if _sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
                raise ValueError("Frozen gallery model changed")
            checkpoint = load_checkpoint(record["checkpoint"])
            model = build_model(checkpoint["config"]["model"]).eval()
            model.load_state_dict(checkpoint["model_state"])
            models[arch, arm] = (model, record)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    credits, panels = [], []
    for id in sorted(roles):
        case = by_id[id]
        if id in cached_credits:
            credit = cached_credits[id]
        else:
            record = next(r for r in manifest["records"] if r["image"]["id"] == id)
            photoid = record["image"]["flickr_url"].rsplit("/", 1)[1].split("_")[0]
            source = f"https://www.flickr.com/photos/_/{photoid}/"
            credit = {
                "coco_id": id,
                "source": source,
                "source_static_url": record["image"]["flickr_url"],
                "license": "https://creativecommons.org/licenses/by/2.0/",
                "modifications": "Photo resized into ground-truth/prediction panel.",
            }
            try:
                url = "https://www.flickr.com/services/oembed/?" + urllib.parse.urlencode(
                    {"format": "json", "url": source}
                )
                with urllib.request.urlopen(url, timeout=30) as response:
                    metadata = json.load(response)
                credit.update(
                    author_name=metadata.get("author_name"),
                    author_url=metadata.get("author_url"),
                    title=metadata.get("title"),
                )
            except Exception as exc:
                credit["lookup_failure"] = f"{type(exc).__name__}: {exc}"
        tensor, box = letterbox(case["image"], 64)
        truth = np.where(case["floor"][..., None], PALETTE[0], [25, 25, 25]).astype(np.uint8)
        truth[~case["valid"]] = [170, 170, 170]
        tiles = [("Input", case["image"]), ("GT floor / other / void", Image.fromarray(truth))]
        photo_included = bool(credit.get("author_name"))
        if not photo_included:
            placeholder = Image.new("RGB", (192, 160), (40, 40, 40))
            ImageDraw.Draw(placeholder).text(
                (8, 45), "Photo local-only:\nauthor unresolved\nSee original manifest", fill="white"
            )
            tiles[0] = ("Input photograph omitted", placeholder)
        measurements = {}
        with torch.inference_mode():
            for (arch, arm), (model, record) in models.items():
                pred = restore_prediction(model(tensor), box, case["image"].size)
                binary, _ = native_confusions(pred, case["categories"])
                reference = next(r for r in record["measured"]["per_image"] if r["id"] == id)
                if binary.tolist() != reference["binary_confusion"]:
                    raise ValueError("Gallery inference differs from measured evidence")
                score = binary_metrics(binary)["per_class"]["floor"]["iou"]
                measurements[f"{arch}:{arm}"] = score
                tiles.append((f"{arch} {arm} IoU {score:.3f}", Image.fromarray(PALETTE[pred])))
                if arm == "baseline":
                    errors = np.zeros((*pred.shape, 3), np.uint8)
                    errors[(pred == 0) & ~case["floor"]] = [245, 45, 45]
                    errors[(pred != 0) & case["floor"]] = [45, 95, 245]
                    errors[~case["valid"]] = [170, 170, 170]
                    tiles.append((f"{arch} floor errors", Image.fromarray(errors)))
        panel = Image.new("RGB", (len(tiles) * 192, 218), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((4, 3), f"COCO {id}: " + ", ".join(roles[id]), fill="black")
        for i, (label, image) in enumerate(tiles):
            panel.paste(ImageOps.pad(image, (192, 160), color=(127, 127, 127)), (i * 192, 40))
            draw.text((i * 192 + 2, 22), label, fill="black")
        panel.save(out / f"{id}.png")
        panels.append(
            {
                "id": id,
                "roles": roles[id],
                "floor_iou": measurements,
                "path": f"{id}.png",
                "photo_included": photo_included,
            }
        )
        credits.append(credit)
        print(id, credit.get("author_name", credit.get("lookup_failure")), flush=True)
    write_json(
        out / "selection.json",
        {
            "panels": panels,
            "input_tag_id_corrections": TAG_ID_CORRECTIONS,
            "models": "baseline/occlusion seed42 at64; no selection by arm.",
            "error_colors": "red false floor, blue missed floor, gray void",
            "prediction_palette": PALETTE.tolist(),
        },
    )
    write_json(out / "credits.json", credits)


if __name__ == "__main__":
    main()

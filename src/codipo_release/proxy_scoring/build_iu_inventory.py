"""Freeze Cookie Theft information-unit regions with SAM 3.

SAM 3 is loaded only at execution time because it is installed from its
upstream distribution rather than as a mandatory dependency of this package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COOKIE_THEFT_PROMPTS = [
    "boy",
    "girl",
    "floor",
    "mother",
    "stool",
    "cookie jar",
    "jar lid",
    "cookie",
    "sink",
    "faucet",
    "water",
    "dish",
    "plate",
    "cup",
    "closet",
    "cabinet",
    "saucer",
    "dishcloth",
    "towel",
    "window",
    "curtain",
    "garden",
    "tree",
    "grass",
    "house",
]


def build_inventory(
    *,
    image_path: Path,
    checkpoint_path: Path,
    confidence_threshold: float = 0.2,
    nms_iou_threshold: float = 0.5,
) -> tuple[list[dict[str, object]], object]:
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from torchvision.ops import nms

    image = Image.open(image_path).convert("RGB")
    model = build_sam3_image_model(checkpoint_path=str(checkpoint_path))
    processor = Sam3Processor(model)
    state = processor.set_image(image)

    all_boxes = []
    all_masks = []
    all_scores = []
    all_names: list[str] = []
    for prompt in COOKIE_THEFT_PROMPTS:
        output = processor.set_text_prompt(state=state, prompt=prompt)
        valid = output["scores"] > confidence_threshold
        if int(valid.sum()) == 0:
            continue
        boxes = output["boxes"][valid]
        masks = output["masks"][valid]
        scores = output["scores"][valid]
        all_boxes.append(boxes)
        all_masks.append(masks)
        all_scores.append(scores)
        all_names.extend([prompt] * len(boxes))
    if not all_boxes:
        raise RuntimeError("SAM 3 did not return any information-unit regions.")

    boxes = torch.cat(all_boxes, dim=0)
    masks = torch.cat(all_masks, dim=0)
    scores = torch.cat(all_scores, dim=0)
    kept = nms(boxes, scores, iou_threshold=nms_iou_threshold)
    inventory = [
        {
            "name": all_names[int(index)],
            "bbox": [float(value) for value in boxes[index].detach().cpu()],
            "sam_score": float(scores[index].detach().cpu()),
            "prompt": all_names[int(index)],
        }
        for index in kept
    ]
    kept_masks = np.asarray(
        [masks[index].detach().cpu().numpy() for index in kept]
    )
    return inventory, kept_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-masks", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--expected-count", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    import numpy as np

    args = parse_args()
    inventory, masks = build_inventory(
        image_path=args.image,
        checkpoint_path=args.sam_checkpoint,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )
    if args.expected_count is not None and len(inventory) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} frozen IU regions, found "
            f"{len(inventory)}. Do not continue with a changed inventory."
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "information_units": inventory,
                "prompts": COOKIE_THEFT_PROMPTS,
                "confidence_threshold": args.confidence_threshold,
                "nms_iou_threshold": args.nms_iou_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.output_masks:
        args.output_masks.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_masks, masks=masks)


if __name__ == "__main__":
    main()

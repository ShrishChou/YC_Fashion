import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor


# -----------------------------
# Helpers: image I/O
# -----------------------------
def load_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_mask_png(mask: np.ndarray, out_path: str) -> None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(out_path, mask_u8)


def save_overlay(image_rgb: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    overlay = image_rgb.copy()
    # green overlay on masked region
    overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_masked_image_whitebg(image_rgb: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    masked = image_rgb.copy()
    masked[~mask] = 255  # white background outside mask
    cv2.imwrite(out_path, cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))


# -----------------------------
# Category logic
# -----------------------------
def category_roi_box(category: str, w: int, h: int) -> np.ndarray:
    """
    Returns a default ROI box (x1,y1,x2,y2) in pixel coords.
    These heuristics reduce 'wrong object' selections for common listing compositions.
    """
    c = category.lower()

    if c in {"shirts", "jackets"}:
        # torso-ish, keep it broad
        x1, y1, x2, y2 = 0.12 * w, 0.05 * h, 0.88 * w, 0.95 * h
    elif c == "pants":
        # lower half dominates
        x1, y1, x2, y2 = 0.10 * w, 0.18 * h, 0.90 * w, 0.99 * h
    elif c == "shoes":
        # often near bottom; still keep wide
        x1, y1, x2, y2 = 0.05 * w, 0.45 * h, 0.95 * w, 0.99 * h
    else:
        # fallback
        x1, y1, x2, y2 = 0.10 * w, 0.10 * h, 0.90 * w, 0.90 * h

    return np.array([int(x1), int(y1), int(x2), int(y2)], dtype=np.int32)


def mask_metrics(mask: np.ndarray, roi: np.ndarray) -> dict:
    """
    Compute metrics used to select the best mask.
    """
    x1, y1, x2, y2 = roi.tolist()
    h, w = mask.shape[:2]

    # Clamp ROI to image bounds
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    area = int(mask.sum())
    if area == 0:
        return {"area": 0, "roi_overlap": 0.0, "bbox_area_ratio": 0.0}

    roi_mask = np.zeros_like(mask, dtype=bool)
    roi_mask[y1:y2, x1:x2] = True

    overlap = int((mask & roi_mask).sum())
    roi_overlap = overlap / float(area)

    # bounding box size ratio (helps reject tiny fragments)
    ys, xs = np.where(mask)
    bx1, bx2 = xs.min(), xs.max()
    by1, by2 = ys.min(), ys.max()
    bbox_area = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    bbox_area_ratio = bbox_area / float(h * w)

    return {
        "area": area,
        "roi_overlap": float(roi_overlap),
        "bbox_area_ratio": float(bbox_area_ratio),
    }


def is_plausible_for_category(category: str, metrics: dict, image_hw: tuple[int, int]) -> bool:
    """
    Simple sanity filters to reject obvious nonsense (too small, etc.).
    You can tune these as you see failures.
    """
    h, w = image_hw
    img_area = h * w
    area_ratio = metrics["area"] / float(img_area)

    c = category.lower()

    # Reject extremely tiny masks always
    if area_ratio < 0.01:
        return False

    if c == "shoes":
        # shoes often smaller than shirts/pants; allow smaller but not microscopic
        if area_ratio > 0.60:
            return False  # almost whole image is unlikely to be shoes
        return True

    if c == "pants":
        # pants often large-ish
        if area_ratio < 0.05:
            return False
        return True

    if c in {"shirts", "jackets"}:
        # medium to large
        if area_ratio < 0.03:
            return False
        return True

    return True


def select_best_mask(category: str, masks: np.ndarray, scores: np.ndarray, roi: np.ndarray, image_hw: tuple[int, int]) -> int:
    """
    Choose the best mask among SAM's candidates using a category-aware score.
    """
    best_idx = 0
    best_value = -1e18

    h, w = image_hw
    img_area = h * w

    for i in range(masks.shape[0]):
        m = masks[i]
        s = float(scores[i])
        met = mask_metrics(m, roi)

        if met["area"] == 0:
            continue
        if not is_plausible_for_category(category, met, (h, w)):
            continue

        area_ratio = met["area"] / float(img_area)

        # Composite scoring:
        # - prioritize ROI overlap a lot
        # - prefer larger masks (but not all-image)
        # - keep some SAM confidence
        value = (
            2.5 * met["roi_overlap"] +
            0.8 * np.sqrt(area_ratio) +
            0.3 * s
        )

        if value > best_value:
            best_value = value
            best_idx = i

    return best_idx


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=["shirts", "jackets", "pants", "shoes"],
                        help="Which category rules to use")
    parser.add_argument("--input", required=True, help="Path to an image OR a folder of images")
    parser.add_argument("--output", default="data/output", help="Output directory")
    parser.add_argument("--sam_checkpoint", default="checkpoints/sam/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu (default: auto)")
    parser.add_argument("--show_roi", action="store_true", help="Save an ROI visualization too")
    args = parser.parse_args()

    # Device selection
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Collect images
    in_path = Path(args.input)
    if in_path.is_dir():
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            img_paths.extend(in_path.glob(ext))
        img_paths = sorted(img_paths)
        if not img_paths:
            raise ValueError(f"No images found in directory: {in_path}")
    else:
        img_paths = [in_path]

    for p in img_paths:
        image_rgb = load_image_rgb(str(p))
        h, w = image_rgb.shape[:2]

        roi = category_roi_box(args.category, w, h)

        predictor.set_image(image_rgb)

        # Use ROI as a box prompt
        masks, scores, _ = predictor.predict(
            box=roi,
            multimask_output=True,
        )

        best = select_best_mask(args.category, masks, scores, roi, (h, w))
        mask = masks[best]

        stem = p.stem
        mask_path = out_dir / f"{stem}_{args.category}_mask.png"
        overlay_path = out_dir / f"{stem}_{args.category}_overlay.png"
        masked_path = out_dir / f"{stem}_{args.category}_masked.png"

        save_mask_png(mask, str(mask_path))
        save_overlay(image_rgb, mask, str(overlay_path))
        save_masked_image_whitebg(image_rgb, mask, str(masked_path))

        if args.show_roi:
            roi_vis = image_rgb.copy()
            x1, y1, x2, y2 = roi.tolist()
            cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (255, 0, 0), 4)
            roi_path = out_dir / f"{stem}_{args.category}_roi.png"
            cv2.imwrite(str(roi_path), cv2.cvtColor(roi_vis, cv2.COLOR_RGB2BGR))

        print(f"[OK] {p.name}  category={args.category}")
        print(f"     roi:      {roi.tolist()}")
        print(f"     selected: idx={best}  sam_score={float(scores[best]):.4f}")
        print(f"     mask:     {mask_path}")
        print(f"     overlay:  {overlay_path}")
        print(f"     masked:   {masked_path}")


if __name__ == "__main__":
    main()

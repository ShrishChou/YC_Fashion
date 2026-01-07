import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    return np.array(rgb)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Green overlay
    out = image_rgb.copy()
    out[mask] = (0.6 * out[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    return out


def draw_contour(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    m = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 0), 2)  # blue-ish contour
    return out


def save_outputs(out_dir: Path, stem: str, image_rgb: np.ndarray, final_mask: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / f"{stem}_mask.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    masked_path = out_dir / f"{stem}_masked.png"

    mask_u8 = (final_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_path), mask_u8)

    ov = overlay_mask(image_rgb, final_mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    masked = image_rgb.copy()
    masked[~final_mask] = 255
    cv2.imwrite(str(masked_path), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

    return mask_path, overlay_path, masked_path


@st.cache_resource
def load_mask_generator(checkpoint_path: str, model_type: str, device: str):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    # These settings control how many masks you get.
    # Start conservative; raise points_per_side if you want more candidates.
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500,  # filter tiny specks
    )
    return generator


def main():
    st.set_page_config(page_title="SAM Mask Selector", layout="wide")

    st.title("SAM Mask Selector (interactive)")
    st.write("Upload an image → generate candidate masks → select the ones you want → export a final union mask.")

    with st.sidebar:
        st.header("Model / Output")
        checkpoint = st.text_input(
            "SAM checkpoint",
            value="checkpoints/sam/sam_vit_h_4b8939.pth",
        )
        model_type = st.selectbox("SAM model type", ["vit_h", "vit_l", "vit_b"], index=0)

        device_auto = "cuda" if torch.cuda.is_available() else "cpu"

        device = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
        if device == "auto":
            device = device_auto

        out_dir = Path(st.text_input("Output directory", value="data/output_selected"))

        st.header("Candidate filtering")
        max_candidates = st.slider("Show top N candidates", 10, 120, 40, 5)
        min_area_pct = st.slider("Min area (% of image)", 0.0, 10.0, 0.3, 0.1)
        max_area_pct = st.slider("Max area (% of image)", 5.0, 100.0, 80.0, 1.0)

        st.caption("Tip: if you’re seeing too many tiny fragments, raise Min area or min_mask_region_area in code.")

    uploaded = st.file_uploader("Upload a listing photo (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.stop()

    pil_img = Image.open(io.BytesIO(uploaded.read()))
    image_rgb = pil_to_rgb_np(pil_img)
    h, w = image_rgb.shape[:2]
    img_area = h * w

    st.subheader("Input image")
    st.image(image_rgb, use_container_width=True)

    if not Path(checkpoint).exists():
        st.error(f"Checkpoint not found: {checkpoint}")
        st.stop()

    if st.button("Generate candidate masks"):
        with st.spinner("Running SAM AutomaticMaskGenerator..."):
            gen = load_mask_generator(str(checkpoint), model_type, device)
            masks = gen.generate(image_rgb)

        # sort larger masks first (usually helpful for picking clothing)
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)

        # filter by area percent
        filtered = []
        for m in masks:
            a = m["area"] / img_area * 100.0
            if a < min_area_pct or a > max_area_pct:
                continue
            filtered.append(m)

        st.session_state["candidates"] = filtered[:max_candidates]
        st.session_state["selected"] = [False] * len(st.session_state["candidates"])

    candidates = st.session_state.get("candidates", None)
    if not candidates:
        st.info("Click **Generate candidate masks** to see options.")
        st.stop()

    st.subheader("Candidates (check the ones to INCLUDE)")
    st.write("Each tile shows the mask overlay + contour. Check multiple to union them.")

    cols = st.columns(4)
    for i, cand in enumerate(candidates):
        mask = cand["segmentation"]
        area_pct = cand["area"] / img_area * 100.0

        tile = overlay_mask(image_rgb, mask)
        tile = draw_contour(tile, mask)

        with cols[i % 4]:
            st.image(tile, use_container_width=True)
            st.caption(f"#{i}  area={area_pct:.2f}%  iou={cand.get('predicted_iou', 0):.2f}")
            st.session_state["selected"][i] = st.checkbox(
                f"Include #{i}",
                value=st.session_state["selected"][i],
                key=f"inc_{i}",
            )

    # Build union mask
    sel = st.session_state["selected"]
    union = np.zeros((h, w), dtype=bool)
    selected_idxs = [i for i, v in enumerate(sel) if v]

    if selected_idxs:
        for i in selected_idxs:
            union |= candidates[i]["segmentation"]

        st.subheader("Final union mask preview")
        st.image(overlay_mask(image_rgb, union), use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            stem = st.text_input("Output name stem", value="selected")
        with colB:
            st.write(" ")

        if st.button("Export mask + overlay + masked image"):
            mask_path, overlay_path, masked_path = save_outputs(out_dir, stem, image_rgb, union)
            st.success("Exported!")
            st.write(f"Mask: {mask_path}")
            st.write(f"Overlay: {overlay_path}")
            st.write(f"Masked: {masked_path}")
    else:
        st.warning("No candidates selected yet. Check one or more masks to build the final output.")


if __name__ == "__main__":
    main()

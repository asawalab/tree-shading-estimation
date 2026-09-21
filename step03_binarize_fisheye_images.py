# ==============================================================
# Step 3 - Binarize Fisheye Images and Generate Diffuse Sky Mask
# ==============================================================
#
# Purpose
# -------
# This step performs interactive binarization on the sky and ground orthographic fisheye images generated in Step 2.
# It also generates the final sky mask used for diffuse-radiation calculation.
# (This extra diffuse mask is needed because the raw binary sky mask is not always appropriate for diffuse radiation.)
# (For example, when the weather-station is very close to the test building, the weather data already reflects the influence of local fixed shading (like far mountains).)
# (If the raw sky mask were used directly, part of that fixed shading could be counted again.)
# (To avoid this, the final diffuse sky mask keeps only the shading components that should remain in the diffuse calculation.)
#
# Contents
# --------
# 1. Load the sky and ground fisheye images from Step 2.
# 2. Perform interactive binary selection on the sky fisheye image.
# 3. Perform interactive binary selection on the ground fisheye image.
# 4. Save the raw binary masks:
#       - Sky_mask
#       - Ground_mask
# 5. Build the final diffuse sky mask according to:
#       - is_control_case
#       - station_has_fixed_shading
# 6. Only when fixed local shading is already included in the weather station data, perform an extra interactive selection of the nearby building region to be kept in the diffuse mask.
#
# Inputs
# ------
# From config.py:
# - image_path
# - is_control_case
# - station_has_fixed_shading
# - control_case_sky_mask_path : required only when
#       is_control_case=False and station_has_fixed_shading=True
#
# From Step 2 outputs:
# - *_SkyFisheye.png
# - *_GroundFisheye.png
#
# From user interaction:
# - raw sky-region selection
# - raw ground-region selection
# - optional nearby-building selection for diffuse mask generation
#
# Outputs
# -------
# Raw binary masks:
# - *_Sky_mask.npy
# - *_Sky_mask.png
# - *_Ground_mask.npy
# - *_Ground_mask.png
#
# Diffuse sky mask:
# - *_Sky_mask_diffuse.npy
# - *_Sky_mask_diffuse.png
#
# Notes
# -----
# - Raw mask convention:
#       1 = sky / visible / unshaded
#       0 = occluder
#
# Diffuse-sky logic
# -----------------
# Case A: is_control_case=True, station_has_fixed_shading=False
#         Sky_mask_diffuse = Sky_mask
#
# Case B: is_control_case=True, station_has_fixed_shading=True
#         Keep only the manually selected nearby test building occlusion in the diffuse sky mask.
#
# Case C: is_control_case=False, station_has_fixed_shading=False
#         Sky_mask_diffuse = Sky_mask
#
# Case D: is_control_case=False, station_has_fixed_shading=True
#         Diffuse sky mask keeps:
#             - tree-added occlusion
#             - manually selected nearby test building occlusion
#         while removing far fixed occlusion.
#
# ==============================================================


from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image

import config as cfg
from region_selector import select_region_mask


# ============================================================
# Data container
# ============================================================

@dataclass
class Step3Paths:
    image_path: str
    basename: str
    step2_dir: str
    step3_dir: str
    sky_fisheye_path: str
    ground_fisheye_path: str


# ============================================================
# Path helpers
# ============================================================

def resolve_paths() -> Step3Paths:
    """
    Resolve Step2 input paths and Step3 output folder from image_path.
    """
    image_path = getattr(cfg, "image_path", None)
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_path = os.path.abspath(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    case_dir = os.path.dirname(image_path)

    step2_folder_name = "02 Fisheye images"
    step3_folder_name = "03 Binarized images"

    step2_dir = os.path.join(case_dir, step2_folder_name)
    step3_dir = os.path.join(case_dir, step3_folder_name)
    os.makedirs(step3_dir, exist_ok=True)

    sky_fisheye_path = os.path.join(step2_dir, f"{basename}_SkyFisheye.png")
    ground_fisheye_path = os.path.join(step2_dir, f"{basename}_GroundFisheye.png")

    if not os.path.exists(sky_fisheye_path):
        raise FileNotFoundError(f"Step2 sky fisheye not found: {sky_fisheye_path}")
    if not os.path.exists(ground_fisheye_path):
        raise FileNotFoundError(f"Step2 ground fisheye not found: {ground_fisheye_path}")

    return Step3Paths(
        image_path=image_path,
        basename=basename,
        step2_dir=step2_dir,
        step3_dir=step3_dir,
        sky_fisheye_path=sky_fisheye_path,
        ground_fisheye_path=ground_fisheye_path,
    )


# ============================================================
# Save helpers
# ============================================================

def save_mask_png(mask01: np.ndarray, valid_mask: np.ndarray, out_png: str) -> None:
    """
    Save binary mask as RGBA PNG:
        white = 1, black = 0, outside valid circle = transparent
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    valid_mask = (valid_mask > 0).astype(np.uint8)

    rgb = (mask01 * 255).astype(np.uint8)
    rgba = np.stack([rgb, rgb, rgb, valid_mask * 255], axis=-1)
    Image.fromarray(rgba).save(out_png)


def save_raw_mask(mask01: np.ndarray, valid_mask: np.ndarray, out_npy: str, out_png: str) -> None:
    """
    Save raw binary mask to NPY and PNG.
    """
    np.save(out_npy, mask01.astype(np.uint8))
    save_mask_png(mask01, valid_mask, out_png)


# ============================================================
# Interactive selection
# ============================================================

def select_raw_mask(image_path: str, target_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interactively select the visible unshaded region.
    """
    instruction = f"Select the visible {target_name} region, then click Confirm"
    mask_bool, valid_mask = select_region_mask(
        image_path,
        instruction=instruction,
    )

    mask01 = mask_bool.astype(np.uint8)
    mask01[~valid_mask] = 0

    if not np.any(mask01):
        raise RuntimeError(f"No visible {target_name} region was selected.")

    return mask01, valid_mask


def select_building_keep_mask(
    image_path: str,
    valid_mask: np.ndarray,
    current_occluder_mask: np.ndarray,
) -> np.ndarray:
    """
    Interactively select the nearby building occluder region to be kept
    in the diffuse sky mask.

    """
    mask_bool, _ = select_region_mask(
        image_path,
        instruction="Select nearby building occlusion, then click Confirm",
    )

    building_keep = mask_bool.astype(np.uint8)
    building_keep[~valid_mask] = 0

    # Restrict to current occluders only
    building_keep[(current_occluder_mask == 0)] = 0
    return building_keep.astype(np.uint8)


# ============================================================
# Diffuse mask logic
# ============================================================

def build_diffuse_mask(
    sky_mask_case: np.ndarray,
    sky_valid: np.ndarray,
    sky_fisheye_path: str,
) -> np.ndarray:
    """
    Build the final Sky_mask_diffuse according to the two switches:
      - is_control_case
      - station_has_fixed_shading
    """
    is_control_case = bool(getattr(cfg, "is_control_case", False))
    station_has_fixed_shading = bool(getattr(cfg, "station_has_fixed_shading", False))

    sky_mask_case = (sky_mask_case > 0).astype(np.uint8)
    sky_valid = sky_valid.astype(bool)

    # Current occluder mask: 1 = occluder
    case_occ = np.zeros_like(sky_mask_case, dtype=np.uint8)
    case_occ[(sky_mask_case == 0) & sky_valid] = 1

    # --------------------------------------------------------
    # Case A / C:
    # station_has_fixed_shading = False
    # diffuse = raw sky mask
    # --------------------------------------------------------
    if not station_has_fixed_shading:
        return sky_mask_case.copy()

    # --------------------------------------------------------
    # Case B:
    # control case + fixed shading
    # diffuse keeps only manually selected nearby building occlusion
    # --------------------------------------------------------
    if is_control_case:
        building_keep = select_building_keep_mask(
            image_path=sky_fisheye_path,
            valid_mask=sky_valid,
            current_occluder_mask=case_occ,
        )

        diffuse = np.ones_like(sky_mask_case, dtype=np.uint8)
        diffuse[sky_valid] = 1
        diffuse[(building_keep == 1) & sky_valid] = 0
        diffuse[~sky_valid] = 0
        return diffuse

    # --------------------------------------------------------
    # Case D:
    # tree case + fixed shading
    # diffuse keeps:
    #   - tree-added occlusion
    #   - manually selected nearby building occlusion
    # and removes far fixed occlusion
    # --------------------------------------------------------
    control_case_sky_mask_path = getattr(cfg, "control_case_sky_mask_path", None)
    if not control_case_sky_mask_path or not os.path.exists(control_case_sky_mask_path):
        raise FileNotFoundError(
            "control_case_sky_mask_path is required when "
            "is_control_case=False and station_has_fixed_shading=True"
        )

    control_sky = np.load(control_case_sky_mask_path)
    control_sky = (control_sky > 0).astype(np.uint8)

    if control_sky.shape != sky_mask_case.shape:
        raise ValueError(
            f"Shape mismatch: current case sky mask {sky_mask_case.shape} vs control case sky mask {control_sky.shape}"
        )

    # tree_occ = newly added occlusion relative to control case
    tree_occ = np.zeros_like(sky_mask_case, dtype=np.uint8)
    tree_occ[(control_sky == 1) & (sky_mask_case == 0) & sky_valid] = 1

    building_keep = select_building_keep_mask(
        image_path=sky_fisheye_path,
        valid_mask=sky_valid,
        current_occluder_mask=case_occ,
    )

    diffuse = np.ones_like(sky_mask_case, dtype=np.uint8)
    diffuse[sky_valid] = 1

    # Keep tree shading
    diffuse[(tree_occ == 1) & sky_valid] = 0

    # Add back nearby building shading
    diffuse[(building_keep == 1) & sky_valid] = 0

    diffuse[~sky_valid] = 0
    return diffuse


# ============================================================
# Main
# ============================================================

def run() -> Dict[str, Any]:
    """
    Main entry of Step3.
    """
    print("\n[Step 3/6] Creating binary masks")

    paths = resolve_paths()

    # --------------------------------------------------------
    # SKY raw mask
    # --------------------------------------------------------
    print("Action required: Select the visible sky region, then click Confirm.")
    sky_mask, sky_valid = select_raw_mask(
        image_path=paths.sky_fisheye_path,
        target_name="sky",
    )

    sky_mask_npy = os.path.join(paths.step3_dir, f"{paths.basename}_Sky_mask.npy")
    sky_mask_png = os.path.join(paths.step3_dir, f"{paths.basename}_Sky_mask.png")
    save_raw_mask(sky_mask, sky_valid, sky_mask_npy, sky_mask_png)

    # --------------------------------------------------------
    # GROUND raw mask
    # --------------------------------------------------------
    print("Action required: Select the visible ground region, then click Confirm.")
    ground_mask, ground_valid = select_raw_mask(
        image_path=paths.ground_fisheye_path,
        target_name="ground",
    )

    ground_mask_npy = os.path.join(paths.step3_dir, f"{paths.basename}_Ground_mask.npy")
    ground_mask_png = os.path.join(paths.step3_dir, f"{paths.basename}_Ground_mask.png")
    save_raw_mask(ground_mask, ground_valid, ground_mask_npy, ground_mask_png)

    # --------------------------------------------------------
    # SKY diffuse mask
    # --------------------------------------------------------
    if bool(getattr(cfg, "station_has_fixed_shading", False)):
        print("Action required: Select nearby building occlusion if present, then click Confirm.")
    sky_mask_diffuse = build_diffuse_mask(
        sky_mask_case=sky_mask,
        sky_valid=sky_valid,
        sky_fisheye_path=paths.sky_fisheye_path,
    )

    sky_diff_npy = os.path.join(paths.step3_dir, f"{paths.basename}_Sky_mask_diffuse.npy")
    sky_diff_png = os.path.join(paths.step3_dir, f"{paths.basename}_Sky_mask_diffuse.png")
    save_raw_mask(sky_mask_diffuse, sky_valid, sky_diff_npy, sky_diff_png)
    print(f"Saved to: {paths.step3_dir}")
    print("[Step 3/6] Completed")

    return {
        "step3_dir": paths.step3_dir,
        "basename": paths.basename,
        "sky_mask_path": sky_mask_npy,
        "ground_mask_path": ground_mask_npy,
        "sky_mask_diffuse_path": sky_diff_npy,
    }


if __name__ == "__main__":
    run()

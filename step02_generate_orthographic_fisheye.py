# ========================================================================
# Step 2 - Generate Orthographic Fisheye Images from the Original Panorama
# ========================================================================
#
# Purpose
# -------
# This step converts the calibrated original equirectangular panorama into orthographic fisheye images.
# In the current version, the sky hemisphere and the ground hemisphere are generated.
#
# Contents
# --------
# 1. Load the original panorama image.
# 2. Read the calibration metadata generated in Step 1.
# 3. Build a normalized circular output grid for orthographic fisheye projection.
# 4. Generate the sky orthographic fisheye image.
# 5. Generate the ground orthographic fisheye image.
# 6. Generate annotated info images with azimuth and zenith-angle guides.
# 7. Compute and save the remapping grids used for the sky and ground projections:
#       - map_x : source x coordinates in the original panorama
#       - map_y : source y coordinates in the original panorama
# 8. Save the fisheye images and remapping data for later processing.
#
# Inputs
# ------
# From config.py:
# - image_path : path to the original equirectangular panorama
# - step2_output_folder_name : output folder name (optional)
# - step2_output_size : output image size ("auto" or integer, optional)
#
# From Step 1 outputs:
# - *_panorama_metadata.json : contains calibration results, especially az_0
#
# Outputs
# -------
# 1. Orthographic fisheye image of the sky hemisphere:
#       *_SkyFisheye.png
# 2. Orthographic fisheye image of the ground hemisphere:
#       *_GroundFisheye.png
# 3. Annotated info image of the sky hemisphere:
#       *_SkyFisheye_Info.png
# 4. Annotated info image of the ground hemisphere:
#       *_GroundFisheye_Info.png
# 5. Remapping grid for the sky projection:
#       *_Sky_mapx.npy
#       *_Sky_mapy.npy
# 6. Remapping grid for the ground projection:
#       *_Ground_mapx.npy
#       *_Ground_mapy.npy
#
# Notes
# -----
# - This step works directly on the ORIGINAL panorama.
# - No additional panorama shifting or recentering is performed.
# - The left-edge azimuth az_0 obtained in Step 1 is used to correctly map absolute azimuth angles back to the original panorama coordinates.
# - Output direction convention of the fisheye images:
#       top    -> North
#       right  -> East
#       bottom -> South
#       left   -> West
# - Orthographic projection relation:
#       r = sin(theta)
#   where theta is the zenith angle.
#
# ========================================================================

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw

import config as cfg


# ============================================================
# Data containers
# ============================================================

@dataclass
class Step2Input:
    image_path: str
    image_np: np.ndarray
    width: int
    height: int
    basename: str
    output_folder: str
    metadata: Dict[str, Any]
    output_size: int
    interpolation: int


# ============================================================
# Path helpers
# ============================================================

def resolve_step1_folder_and_basename(image_path: str) -> Tuple[str, str]:
    """
    Resolve Step1 output folder and basename from original image path.
    """
    image_path = os.path.abspath(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    step1_folder_name = "01 Panorama calibration"
    step1_folder = os.path.join(os.path.dirname(image_path), step1_folder_name)

    return step1_folder, basename


def resolve_step2_output_folder(image_path: str) -> str:
    """
    Resolve Step2 output folder.
    """
    image_path = os.path.abspath(image_path)
    folder_name = getattr(cfg, "step2_output_folder_name", "02 Fisheye images")
    output_folder = os.path.join(os.path.dirname(image_path), folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


# ============================================================
# Load inputs
# ============================================================

def load_step1_results() -> Step2Input:
    """
    Load original panorama and Step1 metadata.
    """
    image_path = getattr(cfg, "image_path", None)
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    step1_folder, basename = resolve_step1_folder_and_basename(image_path)
    metadata_path = os.path.join(step1_folder, f"{basename}_panorama_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Step1 metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    height, width = image_np.shape[:2]

    cfg_size = getattr(cfg, "step2_output_size", "auto")
    if cfg_size == "auto" or cfg_size is None:
        output_size = int(height)
    else:
        output_size = int(cfg_size)

    output_size = max(100, output_size)

    # Keep nearest interpolation for cleaner edge preservation.
    interpolation = cv2.INTER_NEAREST

    output_folder = resolve_step2_output_folder(image_path)

    return Step2Input(
        image_path=image_path,
        image_np=image_np,
        width=width,
        height=height,
        basename=basename,
        output_folder=output_folder,
        metadata=metadata,
        output_size=output_size,
        interpolation=interpolation,
    )


# ============================================================
# Text / font helpers
# ============================================================

def load_fonts_by_height(
    img_height: int,
    font_name: str = "arial.ttf",
    size_ratio: float = 0.032,
    size_large_ratio: float = 0.06,
    min_size: int = 20,
    min_large_size: int = 28,
):
    """
    Load fonts scaled by image height; fallback to default if unavailable.
    """
    from PIL import ImageFont

    size = max(min_size, int(img_height * size_ratio))
    size_large = max(min_large_size, int(img_height * size_large_ratio))

    try:
        font = ImageFont.truetype(font_name, size=size)
        font_large = ImageFont.truetype(font_name, size=size_large)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=size)
            font_large = ImageFont.truetype("DejaVuSans.ttf", size=size_large)
        except OSError:
            font = ImageFont.load_default()
            font_large = font

    return font, font_large


# ============================================================
# Grid helpers
# ============================================================

def build_output_grid(output_size: int):
    """
    Build normalized unit-circle grid for fisheye output.

    Coordinate convention:
    - x: right positive
    - y: up positive
    """
    f = output_size / 2.0
    j_grid, i_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))

    x = (j_grid - f) / f
    y = -(i_grid - f) / f
    r = np.sqrt(x * x + y * y)
    mask = r <= 1.0

    return x, y, r, mask



# ============================================================
# Panorama remap helper
# ============================================================

def build_panorama_remap(
    phi_deg: np.ndarray,
    theta_deg: np.ndarray,
    mask: np.ndarray,
    az_0: float,
    w_equi: int,
    h_equi: int,
):

    map_x = np.full(phi_deg.shape, -1.0, dtype=np.float32)
    map_y = np.full(theta_deg.shape, -1.0, dtype=np.float32)

    # Horizontal direction: periodic wrap.
    delta_phi = (phi_deg[mask].astype(np.float64) - float(az_0)) % 360.0
    mx = delta_phi / 360.0 * float(w_equi)
    mx = np.mod(mx, float(w_equi))

    # Vertical direction: not periodic, so clip safely.
    my = theta_deg[mask].astype(np.float64) / 180.0 * float(h_equi - 1)
    my = np.clip(my, 0.0, float(h_equi - 1))

    map_x[mask] = mx.astype(np.float32)
    map_y[mask] = my.astype(np.float32)

    return map_x, map_y


# ============================================================
# Projection functions
# ============================================================

def generate_sky_fisheye(step2_input: Step2Input):
    """
    Generate orthographic SKY fisheye.

    Output direction convention:
    - top    = North
    - right  = East
    - bottom = South
    - left   = West
    """
    image_np = step2_input.image_np
    h_equi, w_equi = image_np.shape[:2]
    output_size = step2_input.output_size
    interpolation = step2_input.interpolation
    az_0 = float(step2_input.metadata["az_0"])

    x, y, r, mask = build_output_grid(output_size)

    theta = np.zeros((output_size, output_size), dtype=np.float32)
    phi = np.zeros((output_size, output_size), dtype=np.float32)

    # Orthographic projection for sky:
    # r = sin(theta), theta in [0, pi/2]
    theta[mask] = np.arcsin(r[mask]).astype(np.float32)

    # Direction convention:
    # top = North, right = East, bottom = South, left = West
    phi_raw = np.arctan2(x[mask], y[mask])
    phi[mask] = (phi_raw % (2.0 * np.pi)).astype(np.float32)

    phi_deg = np.degrees(phi).astype(np.float32)
    theta_deg = np.degrees(theta).astype(np.float32)

    map_x, map_y = build_panorama_remap(
        phi_deg=phi_deg,
        theta_deg=theta_deg,
        mask=mask,
        az_0=az_0,
        w_equi=w_equi,
        h_equi=h_equi,
    )

    fisheye_rgb = cv2.remap(
        image_np,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_WRAP,
    )

    fisheye_rgb[~mask] = 0

    alpha = np.zeros((output_size, output_size), dtype=np.uint8)
    alpha[mask] = 255
    fisheye_rgba = np.dstack([fisheye_rgb, alpha]).astype(np.uint8)

    return fisheye_rgba, map_x, map_y


def generate_ground_fisheye(step2_input: Step2Input):
    """
    Generate orthographic GROUND fisheye.

    Output direction convention:
    - top    = North
    - right  = East
    - bottom = South
    - left   = West
    """
    image_np = step2_input.image_np
    h_equi, w_equi = image_np.shape[:2]
    output_size = step2_input.output_size
    interpolation = step2_input.interpolation
    az_0 = float(step2_input.metadata["az_0"])

    x, y, r, mask = build_output_grid(output_size)

    theta = np.zeros((output_size, output_size), dtype=np.float32)
    phi = np.zeros((output_size, output_size), dtype=np.float32)

    # Ground hemisphere:
    # theta = 180° at center, theta = 90° at boundary
    theta_sky = np.arcsin(r[mask])
    theta_ground = np.pi - theta_sky
    theta[mask] = theta_ground.astype(np.float32)

    phi_raw = np.arctan2(x[mask], y[mask])
    phi[mask] = (phi_raw % (2.0 * np.pi)).astype(np.float32)

    phi_deg = np.degrees(phi).astype(np.float32)
    theta_deg = np.degrees(theta).astype(np.float32)

    map_x, map_y = build_panorama_remap(
        phi_deg=phi_deg,
        theta_deg=theta_deg,
        mask=mask,
        az_0=az_0,
        w_equi=w_equi,
        h_equi=h_equi,
    )

    fisheye_rgb = cv2.remap(
        image_np,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_WRAP,
    )

    # The outside of the circular fisheye region is controlled by alpha.
    # Set RGB outside the valid circle to zero to keep the saved PNG clean.
    fisheye_rgb[~mask] = 0

    alpha = np.zeros((output_size, output_size), dtype=np.uint8)
    alpha[mask] = 255
    fisheye_rgba = np.dstack([fisheye_rgb, alpha]).astype(np.uint8)

    return fisheye_rgba, map_x, map_y


# ============================================================
# Info-image annotation helpers
# ============================================================

def annotate_fisheye_info(
    rgba_array: np.ndarray,
    hemisphere: str = "sky",   # "sky" or "ground"
    azimuth_step_deg: int = 20,
    angle_circle_step_deg: int = 10,
    azimuth_color=(13, 149, 206),   # blue
    angle_color=(255, 210, 0),      # yellow
    cardinal_color=(237, 64, 67),   # red
) -> np.ndarray:

    img = Image.fromarray(rgba_array.copy())
    draw = ImageDraw.Draw(img)

    w, h = img.size
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    R = min(cx, cy)

    font, font_large = load_fonts_by_height(
        h,
        size_ratio=0.038,
        size_large_ratio=0.065,
        min_size=24,
        min_large_size=32,
    )

    grid_w = max(1, int(h * 0.0025))
    cardinal_w = max(2, int(h * 0.0035))

    tick_len = max(20, int(R * 0.08))
    center_r = max(5, int(h * 0.008))

    hemisphere = hemisphere.lower()

    # --------------------------------------------------------
    # 1) Draw zenith-angle circles
    # --------------------------------------------------------
    if hemisphere == "sky":
        # Sky hemisphere: theta = 0° at center, theta = 90° at boundary.
        theta_values = range(angle_circle_step_deg, 90, angle_circle_step_deg)
    else:
        # Ground hemisphere: theta = 90° at boundary, theta = 180° at center.
        theta_values = range(90 + angle_circle_step_deg, 180, angle_circle_step_deg)

    for theta in theta_values:
        rr = R * np.sin(np.deg2rad(float(theta)))

        x0 = cx - rr
        y0 = cy - rr
        x1 = cx + rr
        y1 = cy + rr

        draw.ellipse((x0, y0, x1, y1), outline=angle_color, width=grid_w)

        label = f"{theta}°"

        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Put label near the upper-right side of each circle.
        tx = min(w - tw - 4, int(cx + 10))
        ty = max(0, int(cy - rr - th - 2))
        draw.text((tx, ty), label, fill=angle_color, font=font)

    # --------------------------------------------------------
    # 2) Draw non-cardinal azimuth ticks
    # --------------------------------------------------------
    cardinal_azimuths = {0, 90, 180, 270}

    for az in range(0, 360, azimuth_step_deg):
        if az in cardinal_azimuths:
            continue

        phi = np.deg2rad(float(az))

        # Outer short tick only
        x_outer = cx + R * np.sin(phi)
        y_outer = cy - R * np.cos(phi)

        r_inner = R - tick_len
        x_inner = cx + r_inner * np.sin(phi)
        y_inner = cy - r_inner * np.cos(phi)

        draw.line(
            (x_inner, y_inner, x_outer, y_outer),
            fill=azimuth_color,
            width=grid_w
        )

        # Non-cardinal azimuth labels
        label = f"{az}°"
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        label_r = R - tick_len - max(20, int(h * 0.035))
        lx = cx + label_r * np.sin(phi)
        ly = cy - label_r * np.cos(phi)

        tx = int(lx - tw / 2)
        ty = int(ly - th / 2)

        tx = max(0, min(w - tw, tx))
        ty = max(0, min(h - th, ty))

        draw.text((tx, ty), label, fill=azimuth_color, font=font)

    # --------------------------------------------------------
    # 3) Draw cardinal direction full lines
    # --------------------------------------------------------
    for az in [0, 90, 180, 270]:
        phi = np.deg2rad(float(az))

        x_outer = cx + R * np.sin(phi)
        y_outer = cy - R * np.cos(phi)

        draw.line(
            (cx, cy, x_outer, y_outer),
            fill=cardinal_color,
            width=cardinal_w
        )

    # --------------------------------------------------------
    # 4) Draw cardinal labels N/E/S/W
    # --------------------------------------------------------
    cardinal_r = R - tick_len - max(25, int(h * 0.04))

    for az, label in {0: "N", 90: "E", 180: "S", 270: "W"}.items():
        phi = np.deg2rad(float(az))

        lx = cx + cardinal_r * np.sin(phi)
        ly = cy - cardinal_r * np.cos(phi)

        bbox = font_large.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tx = int(lx - tw / 2)
        ty = int(ly - th / 2)

        tx = max(0, min(w - tw, tx))
        ty = max(0, min(h - th, ty))

        draw.text((tx, ty), label, fill=cardinal_color, font=font_large)

    # --------------------------------------------------------
    # 5) Draw center point and center angle label
    # --------------------------------------------------------
    draw.ellipse(
        (cx - center_r, cy - center_r, cx + center_r, cy + center_r),
        fill=cardinal_color
    )

    center_label = "0°" if hemisphere == "sky" else "180°"

    bbox = font.getbbox(center_label)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = int(cx + center_r + 6)
    ty = int(cy - th - 6)

    tx = max(0, min(w - tw, tx))
    ty = max(0, min(h - th, ty))

    draw.text((tx, ty), center_label, fill=angle_color, font=font)

    return np.array(img)


# ============================================================
# Save outputs
# ============================================================

def save_rgba_image(rgba_array: np.ndarray, output_path: str) -> None:
    """
    Save RGBA image using PIL.
    """
    Image.fromarray(rgba_array).save(output_path)


def save_step2_outputs(
    step2_input: Step2Input,
    sky_rgba: np.ndarray,
    sky_info_rgba: np.ndarray,
    sky_mapx: np.ndarray,
    sky_mapy: np.ndarray,
    ground_rgba: np.ndarray,
    ground_info_rgba: np.ndarray,
    ground_mapx: np.ndarray,
    ground_mapy: np.ndarray,
) -> Dict[str, str]:
    """
    Save Step2 outputs.
    """
    basename = step2_input.basename
    output_folder = step2_input.output_folder

    sky_img_path = os.path.join(output_folder, f"{basename}_SkyFisheye.png")
    sky_info_img_path = os.path.join(output_folder, f"{basename}_SkyFisheye_Info.png")

    ground_img_path = os.path.join(output_folder, f"{basename}_GroundFisheye.png")
    ground_info_img_path = os.path.join(output_folder, f"{basename}_GroundFisheye_Info.png")

    sky_mapx_path = os.path.join(output_folder, f"{basename}_Sky_mapx.npy")
    sky_mapy_path = os.path.join(output_folder, f"{basename}_Sky_mapy.npy")
    ground_mapx_path = os.path.join(output_folder, f"{basename}_Ground_mapx.npy")
    ground_mapy_path = os.path.join(output_folder, f"{basename}_Ground_mapy.npy")

    save_rgba_image(sky_rgba, sky_img_path)
    save_rgba_image(sky_info_rgba, sky_info_img_path)

    save_rgba_image(ground_rgba, ground_img_path)
    save_rgba_image(ground_info_rgba, ground_info_img_path)

    np.save(sky_mapx_path, sky_mapx.astype(np.float32))
    np.save(sky_mapy_path, sky_mapy.astype(np.float32))
    np.save(ground_mapx_path, ground_mapx.astype(np.float32))
    np.save(ground_mapy_path, ground_mapy.astype(np.float32))

    return {
        "sky_img_path": sky_img_path,
        "sky_info_img_path": sky_info_img_path,
        "ground_img_path": ground_img_path,
        "ground_info_img_path": ground_info_img_path,
        "sky_mapx_path": sky_mapx_path,
        "sky_mapy_path": sky_mapy_path,
        "ground_mapx_path": ground_mapx_path,
        "ground_mapy_path": ground_mapy_path,
    }


# ============================================================
# Main
# ============================================================

def run() -> Dict[str, Any]:
    """
    Main entry of Step2.
    """
    print("\n[Step 2/6] Generating orthographic fisheye images")

    step2_input = load_step1_results()

    sky_rgba, sky_mapx, sky_mapy = generate_sky_fisheye(step2_input)
    ground_rgba, ground_mapx, ground_mapy = generate_ground_fisheye(step2_input)

    sky_info_rgba = annotate_fisheye_info(
        sky_rgba,
        hemisphere="sky",
        azimuth_step_deg=20,
        angle_circle_step_deg=10,
    )

    ground_info_rgba = annotate_fisheye_info(
        ground_rgba,
        hemisphere="ground",
        azimuth_step_deg=20,
        angle_circle_step_deg=10,
    )

    saved_paths = save_step2_outputs(
        step2_input=step2_input,
        sky_rgba=sky_rgba,
        sky_info_rgba=sky_info_rgba,
        sky_mapx=sky_mapx,
        sky_mapy=sky_mapy,
        ground_rgba=ground_rgba,
        ground_info_rgba=ground_info_rgba,
        ground_mapx=ground_mapx,
        ground_mapy=ground_mapy,
    )

    print(f"Saved to: {step2_input.output_folder}")
    print("[Step 2/6] Completed")

    return {
        "image_path": step2_input.image_path,
        "output_folder": step2_input.output_folder,
        "output_size": step2_input.output_size,
        **saved_paths,
    }


if __name__ == "__main__":
    run()

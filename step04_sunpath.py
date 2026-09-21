# =======================================================================
# Step 4 - Compute Sunpath and Overlay It on Sky Fisheye Images and Masks
# =======================================================================
#
# Purpose
# -------
# This step computes the solar position throughout the target day.
# And then, this step projects the sunpath onto the sky orthographic fisheye image generated in Step 2 and the raw sky binary mask generated in Step 3.
# It also evaluates whether each sun position falls on visible sky or on an occluded region.
#
# Contents
# --------
# 1. Load the sky fisheye image from Step 2.
# 2. Load the raw sky mask from Step 3.
# 3. Compute solar azimuth and elevation throughout the day using pvlib.
# 4. Project the sun positions onto the orthographic sky fisheye image.
# 5. Draw the sunpath on:
#       - the sky fisheye image
#       - the raw sky mask image
# 6. Evaluate shading status at each timestamp using the raw sky mask.
# 7. Save overlay images and a table of solar positions and shading results.
#
# Inputs
# ------
# From config.py:
# - image_path
# - latitude
# - longitude
# - altitude
# - timezone
# - date_str
#
# From Step 2 outputs:
# - *_SkyFisheye.png
#
# From Step 3 outputs:
# - *_Sky_mask.npy
# - *_Sky_mask.png (optional, for visualization)
#
# Outputs
# -------
# 1. Sunpath overlay on sky fisheye image:
#       *_SkyFisheye_Sunpath.png
# 2. Sunpath overlay on raw sky mask:
#       *_SkyMask_Sunpath.png
# 3. Table of solar position and shading status:
#       *_Sunpath_Shading.xlsx
#
# Notes
# -----
# - Only daylight timestamps (solar elevation > 0°) are treated as valid.
#
# =======================================================================


from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pvlib.location import Location

import config as cfg


# ============================================================
# Data container
# ============================================================

@dataclass
class Step4Paths:
    image_path: str
    basename: str
    case_dir: str
    step2_dir: str
    step3_dir: str
    step4_dir: str
    sky_fisheye_path: str
    sky_mask_npy_path: str
    sky_mask_png_path: str


# ============================================================
# Path helpers
# ============================================================

def resolve_paths() -> Step4Paths:
    """
    Resolve all Step4 paths from image_path.
    """
    image_path = getattr(cfg, "image_path", None)
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_path = os.path.abspath(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    case_dir = os.path.dirname(image_path)

    step2_dir = os.path.join(case_dir, "02 Fisheye images")
    step3_dir = os.path.join(case_dir, "03 Binarized images")
    step4_dir = os.path.join(case_dir, "04 Sun path")
    os.makedirs(step4_dir, exist_ok=True)

    sky_fisheye_path = os.path.join(step2_dir, f"{basename}_SkyFisheye.png")
    sky_mask_npy_path = os.path.join(step3_dir, f"{basename}_Sky_mask.npy")
    sky_mask_png_path = os.path.join(step3_dir, f"{basename}_Sky_mask.png")

    if not os.path.exists(sky_fisheye_path):
        raise FileNotFoundError(f"Missing Step2 sky fisheye: {sky_fisheye_path}")
    if not os.path.exists(sky_mask_npy_path):
        raise FileNotFoundError(f"Missing Step3 sky mask: {sky_mask_npy_path}")

    return Step4Paths(
        image_path=image_path,
        basename=basename,
        case_dir=case_dir,
        step2_dir=step2_dir,
        step3_dir=step3_dir,
        step4_dir=step4_dir,
        sky_fisheye_path=sky_fisheye_path,
        sky_mask_npy_path=sky_mask_npy_path,
        sky_mask_png_path=sky_mask_png_path,
    )


# ============================================================
# Date
# ============================================================

def get_configured_date() -> str:
    """
    Return cfg.date_str after validating the YYYY-MM-DD format.
    """
    ds = getattr(cfg, "date_str", None)
    try:
        parsed = pd.Timestamp(ds)
    except (TypeError, ValueError):
        raise ValueError("Set cfg.date_str in YYYY-MM-DD format.") from None
    if not isinstance(ds, str) or parsed.strftime("%Y-%m-%d") != ds:
        raise ValueError("Set cfg.date_str in YYYY-MM-DD format.")
    return ds


def build_minute_times(date_str: str, tz: str):
    """
    Build one-minute timestamps from 00:00 to 23:59.
    """
    start = f"{date_str} 00:00:00"
    end = f"{date_str} 23:59:00"
    return pd.date_range(start=start, end=end, freq="1min", tz=tz)


# ============================================================
# Mask
# ============================================================

def load_mask01(mask_npy_path: str) -> np.ndarray:
    m = np.load(mask_npy_path)
    m = np.asarray(m)
    if m.max() > 1.5:
        return (m > 127).astype(np.uint8)
    return (m > 0.5).astype(np.uint8)


def mask01_to_rgba(mask01: np.ndarray) -> Image.Image:
    h, w = mask01.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    R = min(cx, cy)

    jj, ii = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    inside = ((jj - cx) ** 2 + (ii - cy) ** 2) <= (R + 0.5) ** 2

    img8 = (mask01.astype(np.uint8) * 255)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = img8
    rgba[..., 1] = img8
    rgba[..., 2] = img8
    rgba[..., 3] = 0
    rgba[inside, 3] = 255

    return Image.fromarray(rgba)


def mask_value_with_radius(mask01: np.ndarray, x: int, y: int, r: int = 1) -> int:
    h, w = mask01.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0

    if r <= 0:
        return int(mask01[y, x])

    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)

    return int(mask01[y0:y1, x0:x1].max())


# ============================================================
# Projection
# ============================================================

def project_sun_to_sky_orthographic(az_deg: float, el_deg: float, size: int) -> tuple[float, float]:
    """
    Project sun position to orthographic sky fisheye.
    Direction convention:
      top    = North
      right  = East
      bottom = South
      left   = West

    """
    radius = size / 2.0
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0

    theta_deg = 90.0 - el_deg  # zenith angle
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(az_deg)

    r = radius * np.sin(theta_rad)
    x = cx + r * np.sin(phi_rad)
    y = cy - r * np.cos(phi_rad)

    return float(x), float(y)


# ============================================================
# Drawing
# ============================================================

def draw_sunpath_overlay(base_img_rgba: Image.Image, df: pd.DataFrame, out_path: str) -> None:

    img = base_img_rgba.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)

    w, h = img.size

    line_width = max(4, int(round(w / 250)))
    minute_dot_radius = max(2, int(round(w / 500)))
    hour_dot_radius = max(8, int(round(w / 120)))
    label_dx = max(12, int(round(w / 90)))
    label_dy = -max(14, int(round(w / 80)))

    try:
        font = ImageFont.truetype("arial.ttf", size=max(20, int(round(w / 45))))
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(20, int(round(w / 45))))
        except:
            font = ImageFont.load_default()

    pts = []
    for _, row in df.iterrows():
        if int(row["valid"]) != 1:
            continue
        pts.append((float(row["x"]), float(row["y"])))

    if len(pts) >= 2:
        draw.line(pts, fill=(255, 140, 0, 255), width=line_width)

    for _, row in df.iterrows():
        if int(row["valid"]) != 1:
            continue

        x = int(round(row["x"]))
        y = int(round(row["y"]))

        unshaded = int(row["unshaded"])
        col = (0, 255, 0, 255) if unshaded == 1 else (255, 0, 0, 255)

        r_small = minute_dot_radius
        draw.ellipse(
            (x - r_small, y - r_small, x + r_small, y + r_small),
            fill=col
        )

    for _, row in df.iterrows():
        if int(row["valid"]) != 1:
            continue

        x = int(round(row["x"]))
        y = int(round(row["y"]))
        hhmm = str(row["time"])

        if not hhmm.endswith(":00"):
            continue

        unshaded = int(row["unshaded"])
        col = (0, 255, 0, 255) if unshaded == 1 else (255, 0, 0, 255)

        r_big = hour_dot_radius

        draw.ellipse(
            (x - r_big, y - r_big, x + r_big, y + r_big),
            fill=col
        )

        draw.text((x + label_dx, y + label_dy), hhmm, fill=(0, 0, 0, 255), font=font)

    img.save(out_path)


# ============================================================
# Main logic
# ============================================================

def build_sunpath_dataframe(paths: Step4Paths) -> pd.DataFrame:
    """
    Compute per-minute solar positions and shading judgment on raw sky mask.

    """
    tz = getattr(cfg, "timezone", "Asia/Tokyo")
    date_str = get_configured_date()

    lat = float(getattr(cfg, "latitude"))
    lon = float(getattr(cfg, "longitude"))
    alt = float(getattr(cfg, "altitude", 0.0))

    times = build_minute_times(date_str, tz)
    loc = Location(lat, lon, tz=tz, altitude=alt)
    solpos = loc.get_solarposition(times)

    az_all = solpos["azimuth"].values.astype(float)
    el_all = solpos["apparent_elevation"].values.astype(float)

    sky_img = Image.open(paths.sky_fisheye_path).convert("RGBA")
    w, h = sky_img.size
    if w != h:
        raise ValueError(f"Sky fisheye must be square, got {w}x{h}")

    sky_mask = load_mask01(paths.sky_mask_npy_path)
    if sky_mask.shape != (h, w):
        raise ValueError(
            f"Sky mask shape mismatch: mask={sky_mask.shape}, image={(h, w)}"
        )

    rows = []
    sample_r = 1

    for t, az, el in zip(times, az_all, el_all):
        hhmm = t.strftime("%H:%M")
        valid = 1 if el > 0.0 else 0

        if valid == 0:
            rows.append({
                "time": hhmm,
                "azimuth_deg": float(az),
                "elevation_deg": float(el),
                "x": np.nan,
                "y": np.nan,
                "x_int": -1,
                "y_int": -1,
                "mask_value": 0,
                "unshaded": 0,
                "valid": 0,
            })
            continue

        x, y = project_sun_to_sky_orthographic(
            az_deg=float(az),
            el_deg=float(el),
            size=w,
        )
        x_int = int(round(x))
        y_int = int(round(y))

        mask_val = mask_value_with_radius(sky_mask, x_int, y_int, r=sample_r)
        unshaded = 1 if mask_val > 0 else 0

        rows.append({
            "time": hhmm,
            "azimuth_deg": float(az),
            "elevation_deg": float(el),
            "x": float(x),
            "y": float(y),
            "x_int": x_int,
            "y_int": y_int,
            "mask_value": int(mask_val),
            "unshaded": int(unshaded),
            "valid": 1,
        })

    return pd.DataFrame(rows)


def run() -> Dict[str, Any]:
    """
    Main entry of Step4.
    """
    print("\n[Step 4/6] Computing the sun path")

    paths = resolve_paths()

    # --------------------------------------------------------
    # Build sunpath dataframe
    # --------------------------------------------------------
    df = build_sunpath_dataframe(paths)

    # --------------------------------------------------------
    # Save Excel
    # --------------------------------------------------------
    excel_path = os.path.join(paths.step4_dir, f"{paths.basename}_Sunpath_Shading.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="sunpath", index=False)

    # --------------------------------------------------------
    # Draw overlays
    # --------------------------------------------------------
    # Draw on sky fisheye image
    sky_img = Image.open(paths.sky_fisheye_path).convert("RGBA")
    out_sky = os.path.join(paths.step4_dir, f"{paths.basename}_SkyFisheye_Sunpath.png")
    draw_sunpath_overlay(
        base_img_rgba=sky_img,
        df=df,
        out_path=out_sky,
    )

    # Draw on sky mask image
    if os.path.exists(paths.sky_mask_png_path):
        sky_mask_img = Image.open(paths.sky_mask_png_path).convert("RGBA")
    else:
        sky_mask = load_mask01(paths.sky_mask_npy_path)
        sky_mask_img = mask01_to_rgba(sky_mask)

    out_mask = os.path.join(paths.step4_dir, f"{paths.basename}_SkyMask_Sunpath.png")
    draw_sunpath_overlay(
        base_img_rgba=sky_mask_img,
        df=df,
        out_path=out_mask,
    )
    print(f"Saved to: {paths.step4_dir}")
    print("[Step 4/6] Completed")

    return {
        "step4_dir": paths.step4_dir,
        "basename": paths.basename,
        "sunpath_excel_path": excel_path,
        "sky_sunpath_image_path": out_sky,
        "mask_sunpath_image_path": out_mask,
    }


if __name__ == "__main__":
    run()

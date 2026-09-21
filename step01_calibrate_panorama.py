# ====================================================================
# Step 1 - Calibrate Original Panorama and Build Pixel-wise Angle Maps
# ====================================================================
#
# Purpose
# -------
# This step calibrates the spatial geometry of the original equirectangular panorama using the sun as a reference point.
# The main goal is to determine the azimuth angle at the left edge of the panorama (az_0), and then assign direction information to every pixel in the image.
#
# Contents
# -------------------
# 1. Load the original panorama image.
# 2. Build a double-width grayscale preview.
# 3. Let the user manually select the sun region.
# 4. Compute the luminance-weighted centroid of the selected region as the sun position.
# 5. Compute the real solar azimuth and elevation from the shooting time and location.
# 6. Estimate az_0, i.e. the real-world azimuth corresponding to the left edge of the panorama.
# 7. Build two pixel-wise angle maps for the original panorama:
#       - phi_map   : azimuth angle of each pixel (degrees)
#       - theta_map : zenith angle of each pixel (degrees)
# 8. Save reference images and metadata for later processing.
#
# Inputs
# ------
# From config.py:
# - image_path : path to the original equirectangular panorama
# - latitude, longitude, altitude, timezone : site information
# - date_str, time_str : image capture date and time
#
# From user interaction:
# - manual selection of the sun region in the preview window
#
# Outputs
# -------
# 1. A panorama image with direction grid and sun marker:
#       *_Panorama_Direction.png
# 2. A preview image showing the manually selected sun region:
#       *_SunSelectionPreview.png
# 3. Pixel-wise azimuth map of the original panorama:
#       *_phi_map.npy
# 4. Pixel-wise zenith-angle map of the original panorama:
#       *_theta_map.npy
# 5. Metadata JSON containing key calibration results:
#       *_panorama_metadata.json
#
# Notes
# -----
# - This step works on the ORIGINAL panorama only.
# - Angle definitions:
#       theta = 0°   -> zenith (top of panorama)
#       theta = 90°  -> horizon (middle of panorama)
#       theta = 180° -> nadir / ground direction (bottom of panorama)
#
# ====================================================================

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from PIL import Image, ImageDraw
import pandas as pd
from pvlib.location import Location

import config as cfg


# ============================================================
# Local helper functions
# ============================================================

def normalize_azimuth(deg: float) -> float:
    """Normalize azimuth into [0, 360)."""
    return float(deg) % 360.0


def azimuth_to_pixel(azimuth_deg: float, az0_deg: float, width: int) -> int:
    """
    Convert azimuth (deg) to x pixel given left-edge azimuth az0_deg.
    Always clamp result into [0, width-1].
    """
    delta = (normalize_azimuth(azimuth_deg) - normalize_azimuth(az0_deg) + 360.0) % 360.0
    x = int((delta / 360.0) * width)
    return max(0, min(width - 1, x))


def build_double_width_panorama(img: Image.Image) -> Image.Image:
    """
    Concatenate the panorama to itself (double width)
    to avoid the 0°/360° seam problem.
    """
    w, h = img.size
    img_double = Image.new(img.mode, (w * 2, h))
    img_double.paste(img, (0, 0))
    img_double.paste(img, (w, 0))
    return img_double


def load_fonts_by_height(
    img_height: int,
    font_name: str = "arial.ttf",
    size_ratio: float = 0.025,
    size_large_ratio: float = 0.05,
    min_size: int = 16,
    min_large_size: int = 24,
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


def annotate_direction_overlay(
    img: Image.Image,
    az0_deg: float,
    sun_x: float,
    sun_y: float,
    phi_grid_step_deg: int = 20,
    theta_grid_step_deg: int = 30,
    sun_color=(237, 64, 67),
    phi_grid_color=(13, 149, 206),
    theta_grid_color=(255, 210, 0),   # yellow
    cardinal_color=(237, 64, 67),
    theta_text_color=(255, 210, 0),
) -> None:
    """
    Draw azimuth grid, zenith-angle grid, N/E/S/W labels, and the sun marker.
    """
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Fonts
    font, font_large = load_fonts_by_height(
        h,
        size_ratio=0.042,
        size_large_ratio=0.075,
        min_size=26,
        min_large_size=36,
    )

    # Graphic sizes
    sun_r = max(8, int(h * 0.014))
    grid_w_phi = max(1, int(h * 0.0025))
    grid_w_theta = max(1, int(h * 0.0025))
    grid_w_cardinal = max(2, int(h * 0.0035))

    angle_text_y = max(12, int(h * 0.018))
    left_margin = max(12, int(w * 0.012))
    bottom_margin = max(12, int(h * 0.02))

    # Theta label position
    theta_label_offset_y = max(10, int(h * 0.014))

    # ------------------------
    # 1) Regular vertical phi lines and labels (draw all non-cardinal 20° lines)
    # ------------------------
    cardinal_azimuths = {0, 90, 180, 270}

    for az in range(0, 360, int(phi_grid_step_deg)):
        if az in cardinal_azimuths:
            continue  # cardinal lines will be drawn separately

        x = azimuth_to_pixel(float(az), float(az0_deg), w)

        draw.line([(x, 0), (x, h)], fill=phi_grid_color, width=grid_w_phi)

        draw.text(
            (x + 8, angle_text_y),
            f"{az}°",
            fill=phi_grid_color,
            font=font
        )

    # ------------------------
    # 2) Horizontal theta lines and labels
    # ------------------------
    for theta in range(int(theta_grid_step_deg), 180, int(theta_grid_step_deg)):
        y = int((theta / 180.0) * (h - 1))

        draw.line([(0, y), (w, y)], fill=theta_grid_color, width=grid_w_theta)

        # Put theta label slightly above the line
        bbox = font.getbbox(f"{theta}°")
        text_h = bbox[3] - bbox[1]
        label_y = max(0, y - theta_label_offset_y - text_h)

        draw.text(
            (left_margin, label_y),
            f"{theta}°",
            fill=theta_text_color,
            font=font
        )

    # ------------------------
    # 3) Cardinal red vertical lines
    # ------------------------
    for az in [0, 90, 180, 270]:
        x = azimuth_to_pixel(float(az), float(az0_deg), w)
        draw.line([(x, 0), (x, h)], fill=cardinal_color, width=grid_w_cardinal)

    # ------------------------
    # 4) Cardinal direction labels N/E/S/W
    # ------------------------
    for az, label in {0: "N", 90: "E", 180: "S", 270: "W"}.items():
        x = azimuth_to_pixel(float(az), float(az0_deg), w)
        bbox = font_large.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        text_x = max(0, min(w - tw, x - tw // 2))
        text_y = max(0, h - th - bottom_margin)

        draw.text(
            (text_x, text_y),
            label,
            fill=cardinal_color,
            font=font_large
        )

    # ------------------------
    # 5) Sun marker and label
    # ------------------------
    draw.ellipse(
        (sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r),
        fill=sun_color
    )

    draw.text(
        (sun_x + 2 * sun_r, sun_y - sun_r),
        "Sun",
        fill=sun_color,
        font=font_large
    )


# ============================================================
# Data containers
# ============================================================
@dataclass
class PanoramaImage:
    img_pil: Image.Image
    width: int
    height: int


@dataclass
class PreviewBundle:
    """Prepared data for interactive sun selection."""
    gray_img_double: Image.Image
    gray_np_double: np.ndarray
    scaled_img: Image.Image
    scale_ratio: float


@dataclass
class SunSelectionResult:
    """Sun position obtained from manual selection."""
    sun_x_double: float
    sun_x: float
    sun_y: float
    selection_bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


@dataclass
class SolarReference:
    """Real solar position computed from date/time/location."""
    sun_azimuth: float
    sun_elevation: float


@dataclass
class PanoramaGeometry:
    """Spatial geometry of the panorama."""
    az_0: float
    phi_map: np.ndarray     # degrees, shape=(H, W)
    theta_map: np.ndarray   # degrees, shape=(H, W), zenith angle


# ============================================================
# Path helpers
# ============================================================

def resolve_output_folder(image_path: str) -> Tuple[str, str]:
    """
    Create output folder for Step1 and return (output_folder, basename).
    """
    image_path = os.path.abspath(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    folder_name = "01 Panorama calibration"
    output_folder = os.path.join(os.path.dirname(image_path), folder_name)
    os.makedirs(output_folder, exist_ok=True)

    return output_folder, basename


# ============================================================
# 1) Load panorama
# ============================================================

def load_panorama(image_path: str) -> PanoramaImage:
    img_pil = Image.open(image_path).convert("RGB")
    width, height = img_pil.size

    return PanoramaImage(
        img_pil=img_pil,
        width=width,
        height=height,
    )


# ============================================================
# 2) Build preview
# ============================================================

def build_selection_preview(img_pil: Image.Image) -> PreviewBundle:
    """
    Build double-width grayscale preview for sun selection.
    """
    img_double = build_double_width_panorama(img_pil)
    gray_img_double = img_double.convert("L")
    gray_np_double = np.array(gray_img_double)

    preview_width = 1000
    preview_width = max(200, preview_width)

    scale_ratio = preview_width / gray_img_double.width
    scaled_img = gray_img_double.resize(
        (
            int(gray_img_double.width * scale_ratio),
            int(gray_img_double.height * scale_ratio),
        )
    )

    return PreviewBundle(
        gray_img_double=gray_img_double,
        gray_np_double=gray_np_double,
        scaled_img=scaled_img,
        scale_ratio=scale_ratio,
    )


# ============================================================
# 3) Manual sun selection
# ============================================================

def select_sun_centroid(
    preview: PreviewBundle,
    img_width: int,
) -> SunSelectionResult:
    """
    Let user manually select a sun region, then compute luminance-weighted centroid.
    """
    gray_img_double = preview.gray_img_double
    gray_np_double = preview.gray_np_double
    scaled_img = preview.scaled_img
    scale_ratio = preview.scale_ratio

    fig, ax = plt.subplots(figsize=(16, 4))
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(scaled_img, cmap="gray")
    ax.set_title("Select the sun region, then click Confirm")

    selection: Dict[str, Optional[Tuple[int, int, int, int]]] = {"extent": None}
    result: Dict[str, Optional[SunSelectionResult]] = {"value": None}

    def onselect(eclick, erelease):
        if (
            eclick.xdata is None or eclick.ydata is None
            or erelease.xdata is None or erelease.ydata is None
        ):
            return

        x1 = int(eclick.xdata / scale_ratio)
        y1 = int(eclick.ydata / scale_ratio)
        x2 = int(erelease.xdata / scale_ratio)
        y2 = int(erelease.ydata / scale_ratio)

        x1 = max(0, min(gray_np_double.shape[1] - 1, x1))
        x2 = max(0, min(gray_np_double.shape[1] - 1, x2))
        y1 = max(0, min(gray_np_double.shape[0] - 1, y1))
        y2 = max(0, min(gray_np_double.shape[0] - 1, y2))

        selection["extent"] = (x1, y1, x2, y2)

    def on_confirm(_event):
        extent = selection.get("extent")
        if extent is None:
            print("Selection error: No valid region selected.")
            return

        x1, y1, x2, y2 = extent
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        if x_max - x_min < 2 or y_max - y_min < 2:
            print("Selection error: The selected region is too small.")
            return

        sub_area = gray_np_double[y_min:y_max, x_min:x_max]

        yy, xx = np.indices(sub_area.shape)
        weights = sub_area.astype(np.float32)
        w_sum = float(np.sum(weights))

        if w_sum <= 1e-6:
            print("Selection error: The selected region is too dark or empty.")
            return

        centroid_x = float(np.sum(xx * weights) / w_sum)
        centroid_y = float(np.sum(yy * weights) / w_sum)

        sun_x_double = x_min + centroid_x
        sun_y = y_min + centroid_y
        sun_x = float(sun_x_double) % img_width

        result["value"] = SunSelectionResult(
            sun_x_double=float(sun_x_double),
            sun_x=float(sun_x),
            sun_y=float(sun_y),
            selection_bbox=(x_min, y_min, x_max, y_max),
        )

        plt.close(fig)

    _selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props=dict(facecolor="white", edgecolor="black", alpha=0.05, fill=True),
    )

    confirm_ax = plt.axes((0.45, 0.05, 0.1, 0.075))
    button = Button(confirm_ax, "Confirm", color="lightgreen", hovercolor="green")
    button.on_clicked(on_confirm)

    plt.show()

    if result["value"] is None:
        raise RuntimeError("Sun selection was not completed.")

    return result["value"]


# ============================================================
# 4) Compute real solar position
# ============================================================

def compute_solar_reference() -> SolarReference:
    """
    Compute real solar azimuth/elevation using pvlib.
    """
    loc = Location(cfg.latitude, cfg.longitude, cfg.timezone, cfg.altitude)
    dt = pd.Timestamp(f"{cfg.date_str} {cfg.time_str}", tz=cfg.timezone)
    times = pd.DatetimeIndex([dt])
    solar_pos = loc.get_solarposition(times)

    sun_azimuth = float(solar_pos["azimuth"].iloc[0])
    sun_elevation = float(solar_pos["apparent_elevation"].iloc[0])

    return SolarReference(
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
    )


def check_sun_theta_consistency(
    img_height: int,
    sun_y: float,
    sun_elevation: float,
) -> Dict[str, float]:
    """
    Check whether the selected sun y-position is consistent with the
    solar elevation computed by pvlib.

    Returns:
        sun_theta_from_image : zenith angle derived from image y
        sun_theta_from_pvlib : zenith angle derived from solar elevation
        sun_theta_error      : image-derived minus pvlib-derived
    """
    if img_height <= 1:
        sun_theta_from_image = 0.0
    else:
        sun_theta_from_image = (float(sun_y) / float(img_height - 1)) * 180.0

    sun_theta_from_pvlib = 90.0 - float(sun_elevation)
    sun_theta_error = sun_theta_from_image - sun_theta_from_pvlib

    return {
        "sun_theta_from_image": float(sun_theta_from_image),
        "sun_theta_from_pvlib": float(sun_theta_from_pvlib),
        "sun_theta_error": float(sun_theta_error),
    }


# ============================================================
# 5) Compute panorama geometry
# ============================================================

def compute_panorama_geometry(
    img_width: int,
    img_height: int,
    sun_x: float,
    sun_azimuth: float,
) -> PanoramaGeometry:
    """
    Compute az_0 and pixel-wise angle maps (phi_map, theta_map).

    Definitions:
    - phi_map: azimuth angle in degrees
    - theta_map: zenith angle in degrees
        top    -> 0°
        middle -> 90°
        bottom -> 180°
    """
    # 1) az_0 = azimuth at the LEFT edge of the original panorama
    pixel_per_deg = img_width / 360.0
    az_0 = (sun_azimuth - sun_x / pixel_per_deg) % 360.0

    # 2) phi_map: each column corresponds to one azimuth
    phi_per_col = (az_0 + np.arange(img_width, dtype=np.float32) / img_width * 360.0) % 360.0
    phi_map = np.tile(phi_per_col, (img_height, 1)).astype(np.float32)

    # 3) theta_map: each row corresponds to one zenith angle
    if img_height <= 1:
        theta_per_row = np.zeros((img_height,), dtype=np.float32)
    else:
        theta_per_row = (np.arange(img_height, dtype=np.float32) / (img_height - 1)) * 180.0

    theta_map = np.repeat(theta_per_row[:, None], img_width, axis=1).astype(np.float32)

    return PanoramaGeometry(
        az_0=float(az_0),
        phi_map=phi_map,
        theta_map=theta_map,
    )


# ============================================================
# 6) Save outputs
# ============================================================

def save_direction_image(
    img_pil: Image.Image,
    az_0: float,
    sun_x: float,
    sun_y: float,
    output_path: str,
) -> None:
    """
    Save panorama with direction overlay and sun marker.
    """
    img_out = img_pil.copy()

    annotate_direction_overlay(
        img=img_out,
        az0_deg=float(az_0),
        sun_x=float(sun_x),
        sun_y=float(sun_y),
        phi_grid_step_deg=20,
        theta_grid_step_deg=30,
    )

    img_out.save(output_path)


def save_selection_preview(
    gray_img_double: Image.Image,
    selection_bbox: Tuple[int, int, int, int],
    output_path: str,
) -> None:
    """
    Save selected sun region preview on double-width grayscale image.
    The selected region is shown as an ellipse.
    """
    preview = gray_img_double.convert("RGB")
    draw = ImageDraw.Draw(preview)

    x_min, y_min, x_max, y_max = selection_bbox
    pad = 2
    draw.ellipse((x_min - pad, y_min - pad, x_max + pad, y_max + pad), outline=(255, 0, 0), width=4)

    preview.save(output_path)


def save_metadata(
    output_path: str,
    pano: PanoramaImage,
    sun: SunSelectionResult,
    solar_ref: SolarReference,
    geom: PanoramaGeometry,
    theta_check: Dict[str, float],
) -> None:
    """
    Save small metadata JSON for later steps.
    """
    metadata = {
        "image_width": pano.width,
        "image_height": pano.height,
        "sun_x": float(sun.sun_x),
        "sun_y": float(sun.sun_y),
        "sun_x_double": float(sun.sun_x_double),
        "sun_azimuth": float(solar_ref.sun_azimuth),
        "sun_elevation": float(solar_ref.sun_elevation),
        "az_0": float(geom.az_0),

        "sun_theta_from_image": float(theta_check["sun_theta_from_image"]),
        "sun_theta_from_pvlib": float(theta_check["sun_theta_from_pvlib"]),
        "sun_theta_error": float(theta_check["sun_theta_error"]),

        "theta_definition": "zenith angle in degrees: top=0, middle=90, bottom=180",
        "phi_definition": "azimuth in degrees: left edge starts at az_0 and increases to the right",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_step1_outputs(
    pano: PanoramaImage,
    preview: PreviewBundle,
    sun: SunSelectionResult,
    solar_ref: SolarReference,
    geom: PanoramaGeometry,
    theta_check: Dict[str, float],
    output_folder: str,
    basename: str,
) -> Dict[str, str]:
    """
    Save all Step1 outputs and return a dictionary of saved paths.
    """
    direction_image_path = os.path.join(output_folder, f"{basename}_Panorama_Direction.png")
    selection_preview_path = os.path.join(output_folder, f"{basename}_SunSelectionPreview.png")
    phi_map_path = os.path.join(output_folder, f"{basename}_phi_map.npy")
    theta_map_path = os.path.join(output_folder, f"{basename}_theta_map.npy")
    metadata_path = os.path.join(output_folder, f"{basename}_panorama_metadata.json")

    save_direction_image(
        img_pil=pano.img_pil,
        az_0=geom.az_0,
        sun_x=sun.sun_x,
        sun_y=sun.sun_y,
        output_path=direction_image_path,
    )

    save_selection_preview(
        gray_img_double=preview.gray_img_double,
        selection_bbox=sun.selection_bbox,
        output_path=selection_preview_path,
    )

    np.save(phi_map_path, geom.phi_map.astype(np.float32))
    np.save(theta_map_path, geom.theta_map.astype(np.float32))

    save_metadata(
        output_path=metadata_path,
        pano=pano,
        sun=sun,
        solar_ref=solar_ref,
        geom=geom,
        theta_check=theta_check,
    )

    return {
        "direction_image_path": direction_image_path,
        "selection_preview_path": selection_preview_path,
        "phi_map_path": phi_map_path,
        "theta_map_path": theta_map_path,
        "metadata_path": metadata_path,
    }


# ============================================================
# Main
# ============================================================

def run() -> Dict[str, Any]:
    """
    Main entry of Step1.
    """
    print("\n[Step 1/6] Calibrating panorama")

    image_path = getattr(cfg, "image_path", None)
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_folder, basename = resolve_output_folder(image_path)

    # 1) Load image
    pano = load_panorama(image_path)

    # 2) Build preview
    preview = build_selection_preview(pano.img_pil)

    # 3) Manual sun selection
    print("Action required: Select the sun region, then click Confirm.")
    sun = select_sun_centroid(preview, pano.width)

    # 4) Compute solar reference
    solar_ref = compute_solar_reference()

    # 4.5) Check sun theta consistency
    theta_check = check_sun_theta_consistency(
        img_height=pano.height,
        sun_y=sun.sun_y,
        sun_elevation=solar_ref.sun_elevation,
    )

    # 5) Compute panorama geometry
    geom = compute_panorama_geometry(
        img_width=pano.width,
        img_height=pano.height,
        sun_x=sun.sun_x,
        sun_azimuth=solar_ref.sun_azimuth,
    )

    # 6) Save outputs
    saved_paths = save_step1_outputs(
        pano=pano,
        preview=preview,
        sun=sun,
        solar_ref=solar_ref,
        geom=geom,
        theta_check=theta_check,
        output_folder=output_folder,
        basename=basename,
    )

    results = {
        "image_path": image_path,
        "output_folder": output_folder,
        "basename": basename,
        "image_width": pano.width,
        "image_height": pano.height,
        "sun_x": sun.sun_x,
        "sun_y": sun.sun_y,
        "sun_azimuth": solar_ref.sun_azimuth,
        "sun_elevation": solar_ref.sun_elevation,
        "sun_theta_from_image": theta_check["sun_theta_from_image"],
        "sun_theta_from_pvlib": theta_check["sun_theta_from_pvlib"],
        "sun_theta_error": theta_check["sun_theta_error"],
        "az_0": geom.az_0,
        **saved_paths,
    }

    print(f"Calibration azimuth (az_0): {geom.az_0:.3f}°")
    print(f"Saved to: {output_folder}")
    print("[Step 1/6] Completed")

    return results


if __name__ == "__main__":
    run()

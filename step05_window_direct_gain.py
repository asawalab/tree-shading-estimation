# ====================================================================
# Step 5 - Compute Direct Window Gain Ratio (Tree Case Only)
# ====================================================================
#
# Purpose
# -------
# This step computes the direct window gain ratio (0~1) by projecting the window onto the raw sky binary mask and sampling the visible-sky fraction for each timestamp.
# In this workflow, Step 5 is used only for tree cases.
# For control cases, the direct component can be handled directly from the Step 4 sunpath shading judgment, so Step 5 is skipped.
#
# Contents
# --------
# 1. Load the raw sky mask from Step 3.
# 2. Load the sunpath table from Step 4.
# 3. Rasterize the projected window on the z = D_eff plane.
# 4. Map the rasterized window points to the orthographic sky fisheye mask.
# 5. Sample the sky mask and compute the mean visible fraction as Window Gain (direct).
# 6. Save the gain table.
# 7. Save hourly preview images for daylight hours.
#
# Inputs
# ------
# From config.py:
# - image_path
# - is_control_case
# - window_width / window_height  (meters)
# - surface_azimuth_deg           (degrees)
# - tree_range_m                  (meters, horizontal center-to-tree distance in plan view)
# - tree_bearing_az_deg           (degrees, azimuth from window center toward the tree)
# - camera_height_m
# - camera_outward_m
# - camera_dx_m
# - window_center_height
#
# Step 5 will automatically compute:
# - D_effective_m from tree_range_m, tree_bearing_az_deg, and surface_azimuth_deg
#
# Outputs
# -------
# 1. Direct window gain table:
#       *_window_direct_gain.xlsx
# 2. Hourly preview images (daylight hours only):
#       previews/window_hour_XX.png
#
# Notes
# -----
# - This step uses the RAW sky mask, not the diffuse sky mask.
# - The gain ratio ranges from 0 to 1.
# - If is_control_case=True, this step is skipped.
#
# ====================================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config as cfg


# ============================================================
# Data container
# ============================================================

@dataclass
class Step5Paths:
    image_path: str
    basename: str
    case_dir: str
    step3_dir: str
    step4_dir: str
    step5_dir: str
    sky_mask_npy_path: str
    sky_mask_png_path: str
    sunpath_xlsx_path: str
    out_xlsx_path: str
    preview_dir: str


@dataclass
class Step5Params:
    window_width_m: float
    window_height_m: float
    D_effective_m: float
    cam_offset_xyz: Tuple[float, float, float]
    phi_offset_deg: float
    horizon_tol_deg: float
    raster_nx: int
    raster_ny: int
    full_day_fill: bool
    save_hourly_previews: bool
    preview_point_radius: int
    preview_point_alpha: int
    time_col: str
    gain_col: str


def _cosd(deg: float) -> float:
    return float(np.cos(np.deg2rad(float(deg))))


# ============================================================
# Path helpers
# ============================================================

def resolve_paths() -> Step5Paths:
    """
    Resolve all paths from image_path.
    """
    image_path = getattr(cfg, "image_path", None)
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_path = os.path.abspath(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    case_dir = os.path.dirname(image_path)

    step3_dir = os.path.join(case_dir, "03 Binarized images")
    step4_dir = os.path.join(case_dir, "04 Sun path")
    step5_dir = os.path.join(case_dir, "05 Window Direct Gain")
    preview_dir = os.path.join(step5_dir, "previews")

    os.makedirs(step5_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    sky_mask_npy_path = os.path.join(step3_dir, f"{basename}_Sky_mask.npy")
    sky_mask_png_path = os.path.join(step3_dir, f"{basename}_Sky_mask.png")
    sunpath_xlsx_path = os.path.join(step4_dir, f"{basename}_Sunpath_Shading.xlsx")
    out_xlsx_path = os.path.join(step5_dir, f"{basename}_window_direct_gain.xlsx")

    if not os.path.exists(sky_mask_npy_path):
        raise FileNotFoundError(f"Missing Step3 sky mask: {sky_mask_npy_path}")
    if not os.path.exists(sunpath_xlsx_path):
        raise FileNotFoundError(f"Missing Step4 sunpath excel: {sunpath_xlsx_path}")

    return Step5Paths(
        image_path=image_path,
        basename=basename,
        case_dir=case_dir,
        step3_dir=step3_dir,
        step4_dir=step4_dir,
        step5_dir=step5_dir,
        sky_mask_npy_path=sky_mask_npy_path,
        sky_mask_png_path=sky_mask_png_path,
        sunpath_xlsx_path=sunpath_xlsx_path,
        out_xlsx_path=out_xlsx_path,
        preview_dir=preview_dir,
    )


# ============================================================
# Param helpers
# ============================================================

def params_from_cfg() -> Step5Params:
    """
    Read Step5 parameters from config and compute D_effective automatically.
    """
    required = [
        "window_width",
        "window_height",
        "camera_height_m",
        "camera_outward_m",
        "window_center_height",
        "tree_range_m",
        "tree_bearing_az_deg",
        "surface_azimuth_deg",
    ]
    missing = [name for name in required if not hasattr(cfg, name)]
    if missing:
        raise AttributeError(
            "Step 5 is missing required geometry settings in config.py: "
            f"{', '.join(missing)}"
        )

    window_width_m = float(cfg.window_width)
    window_height_m = float(cfg.window_height)

    dx = float(getattr(cfg, "camera_dx_m", 0.0))
    dy = float(cfg.camera_height_m) - float(cfg.window_center_height)
    dz = float(cfg.camera_outward_m)
    cam_offset_xyz = (dx, dy, dz)

    tree_range_m = float(getattr(cfg, "tree_range_m"))
    tree_bearing_az_deg = float(getattr(cfg, "tree_bearing_az_deg"))
    surface_azimuth_deg = float(getattr(cfg, "surface_azimuth_deg"))

    D_effective_m = tree_range_m * _cosd(tree_bearing_az_deg - surface_azimuth_deg)

    if D_effective_m <= 0:
        raise ValueError(
            f"Computed D_effective_m = {D_effective_m:.4f} m <= 0.\n"
            "Please check tree_range_m / tree_bearing_az_deg / surface_azimuth_deg.\n"
            "The dominant tree shading layer should be located in front of the window."
        )

    return Step5Params(
        window_width_m=window_width_m,
        window_height_m=window_height_m,
        D_effective_m=D_effective_m,
        cam_offset_xyz=cam_offset_xyz,
        phi_offset_deg=float(getattr(cfg, "phi_offset_deg", 0.0)),
        horizon_tol_deg=float(getattr(cfg, "horizon_tol_deg", 0.3)),
        raster_nx=int(getattr(cfg, "step5_raster_nx", 200)),
        raster_ny=int(getattr(cfg, "step5_raster_ny", 211)),
        full_day_fill=bool(getattr(cfg, "step5_full_day_fill", True)),
        save_hourly_previews=bool(getattr(cfg, "step5_save_hourly_previews", True)),
        preview_point_radius=int(getattr(cfg, "step5_preview_point_radius", 2)),
        preview_point_alpha=int(getattr(cfg, "step5_preview_point_alpha", 120)),
        time_col=str(getattr(cfg, "step5_time_col", "time")),
        gain_col=str(getattr(cfg, "step5_gain_col", "Window Gain (direct)")),
    )


# ============================================================
# Small helpers
# ============================================================

def normalize_time_to_HHMM(series: pd.Series) -> pd.Series:
    """
    Normalize time strings to HH:MM.
    """
    s = series.astype(str).str.strip()
    m = s.str.extract(r'^\s*(\d{1,2}):(\d{1,2})(?::\d{1,2})?\s*$')
    ok = m.notna().all(axis=1)
    out = s.copy()
    out.loc[ok] = (
        m.loc[ok, 0].astype(int).astype(str).str.zfill(2) + ":" +
        m.loc[ok, 1].astype(int).astype(str).str.zfill(2)
    )
    out.loc[~ok] = np.nan
    return out


def pad_full_day_minutes(df: pd.DataFrame, time_col: str, cols_to_fill_zero: List[str]) -> pd.DataFrame:
    """
    Pad to full 00:00~23:59 minutes.
    """
    base = df.copy()
    base[time_col] = normalize_time_to_HHMM(base[time_col])
    base = base.dropna(subset=[time_col]).drop_duplicates(subset=[time_col], keep="first")

    full = pd.DataFrame({time_col: pd.date_range("00:00", "23:59", freq="1min").strftime("%H:%M")})
    merged = full.merge(base, on=time_col, how="left")
    for c in cols_to_fill_zero:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0).astype(float)
    return merged


def get_configured_date() -> str:
    """
    Return the configured date for preview labels.
    """
    ds = getattr(cfg, "date_str", None)
    if not isinstance(ds, str) or not ds.strip():
        raise ValueError("Set cfg.date_str in YYYY-MM-DD format.")
    return ds.strip()


# ============================================================
# Mask loading
# ============================================================

def load_sky_mask(mask_npy_path: str) -> Tuple[np.ndarray, Tuple[float, float, float, int, int]]:
    """
    Load raw sky mask and projection info.
    V=1 means sky/unshaded.

    """
    arr = np.load(mask_npy_path)
    arr = np.asarray(arr)
    if arr.max() > 1.5:
        V = (arr / 255.0).astype(np.float32)
    else:
        V = arr.astype(np.float32)

    H, W = V.shape[:2]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    R = min(cx, cy, (W - 1) - cx, (H - 1) - cy)
    return V, (cx, cy, R, H, W)


def bilinear_sample(V: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling on V for fractional pixel coordinates.
    """
    H, W = V.shape
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    fx = px - x0
    fy = py - y0

    Ia = V[y0, x0]
    Ib = V[y0, x1]
    Ic = V[y1, x0]
    Id = V[y1, x1]

    top = Ia * (1 - fx) + Ib * fx
    bot = Ic * (1 - fx) + Id * fx
    return top * (1 - fy) + bot * fy


# ============================================================
# Geometry
# ============================================================

def wrap_0_2pi(phi: np.ndarray) -> np.ndarray:
    return (phi + 2 * np.pi) % (2 * np.pi)


def sun_vec_local(theta_s: float, phi_s: float) -> np.ndarray:
    """
    Convert (theta_s, phi_s) to local unit vector:
      x = East(+), y = Up(+), z = South(+)

    """
    e = np.pi / 2 - float(theta_s)  # elevation
    phi = float(phi_s)
    sx = np.cos(e) * np.sin(phi)
    sy = np.sin(e)
    sz = -np.cos(e) * np.cos(phi)
    s = np.array([sx, sy, sz], dtype=float)
    n = np.linalg.norm(s)
    return s / (n if n > 0 else 1.0)


def center_shift_on_zD(theta_s: float, phi_s: float, D: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Intersection between center ray and plane z=D.
    """
    sx, sy, sz = sun_vec_local(theta_s, phi_s)
    if sz <= 0:
        return None, None
    t = D / sz
    return t * sx, t * sy


def xy_to_angles_with_viewpoint(
    X: np.ndarray,
    Y: np.ndarray,
    D_eff: float,
    viewpoint_xyz: Tuple[float, float, float],
    phi_offset_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map points on plane z=D_eff into (theta, phi) seen from viewpoint.

    """
    dx, dy, dz = viewpoint_xyz
    Dc = D_eff - dz
    Xc = X - dx
    Yc = Y - dy

    phi = np.mod(np.arctan2(Xc, -Dc) + np.deg2rad(phi_offset_deg), 2 * np.pi)
    e = np.arctan2(Yc, np.hypot(Dc, Xc))
    theta = (np.pi / 2) - e
    return theta, phi


def window_gain_raster(
    theta_s: float,
    phi_s: float,
    V: np.ndarray,
    proj: Tuple[float, float, float, int, int],
    window_w: float,
    window_h: float,
    D_eff: float,
    viewpoint_xyz: Tuple[float, float, float],
    phi_offset_deg: float,
    horizon_tol_deg: float,
    nx: int,
    ny: int,
) -> float:
    """
    Rasterize the window on plane z=D_eff, map it to the sky mask, and
    compute the average visible fraction.

    """
    if theta_s is None or phi_s is None or np.isnan(theta_s) or np.isnan(phi_s):
        return 0.0

    X0, Y0 = center_shift_on_zD(theta_s, phi_s, D_eff)
    if X0 is None:
        return 0.0

    cx, cy, R, H, W = proj
    a, b = window_w / 2.0, window_h / 2.0

    xs = np.linspace(-a, a, nx)
    ys = np.linspace(-b, b, ny)
    Xg, Yg = np.meshgrid(xs, ys)

    X = X0 + Xg
    Y = Y0 + Yg

    theta_t, phi_t = xy_to_angles_with_viewpoint(X, Y, D_eff, viewpoint_xyz, phi_offset_deg)

    r = R * np.sin(theta_t)
    px = cx + r * np.sin(phi_t)
    py = cy - r * np.cos(phi_t)

    tol = np.deg2rad(float(horizon_tol_deg))
    valid = (
        (theta_t >= 0.0) &
        (theta_t <= (np.pi / 2 + tol)) &
        (px >= 0) & (px <= W - 2) &
        (py >= 0) & (py <= H - 2) &
        ((px - cx) ** 2 + (py - cy) ** 2 <= (R + 0.5) ** 2)
    )

    if not np.any(valid):
        return 0.0

    vals = np.zeros_like(px, dtype=np.float32)
    vals[valid] = bilinear_sample(V, px[valid], py[valid])
    return float(vals[valid].mean())


# ============================================================
# Read Step4 sunpath excel
# ============================================================

def read_step4_sunpath_excel(xlsx_path: str) -> pd.DataFrame:
    """
    Read Step4 sunpath excel and return:
      time, theta_s_rad, phi_s_rad, valid

    Required columns:
      - time
      - azimuth_deg
      - elevation_deg

    """
    df = pd.read_excel(xlsx_path)
    low = {c.lower(): c for c in df.columns}

    def _pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None

    tcol = _pick("time")
    azcol = _pick("azimuth_deg")
    elcol = _pick("elevation_deg")
    vcol = _pick("valid")

    if tcol is None or azcol is None or elcol is None:
        raise ValueError(
            f"Missing required columns in Step4 excel: {list(df.columns)}\n"
            f"Need at least: time, azimuth_deg, elevation_deg"
        )

    out = pd.DataFrame()
    out["time"] = normalize_time_to_HHMM(df[tcol])

    az = df[azcol].astype(float).to_numpy()
    el = df[elcol].astype(float).to_numpy()

    theta_deg = 90.0 - el
    phi_deg = az

    out["theta_s_rad"] = np.deg2rad(theta_deg)
    out["phi_s_rad"] = wrap_0_2pi(np.deg2rad(phi_deg))

    if vcol is not None:
        valid = df[vcol].astype(float).to_numpy() > 0.5
    else:
        valid = el > 0.0

    out["valid"] = valid.astype(bool)
    out.loc[~out["valid"], "theta_s_rad"] = np.nan
    out.loc[~out["valid"], "phi_s_rad"] = np.nan

    out = out.dropna(subset=["time"]).copy()
    return out[["time", "theta_s_rad", "phi_s_rad", "valid"]]


# ============================================================
# Hourly preview
# ============================================================

def save_hourly_previews(
    df_angles: pd.DataFrame,
    V: np.ndarray,
    proj: Tuple[float, float, float, int, int],
    outdir: str,
    basename: str,
    window_w: float,
    window_h: float,
    D_eff: float,
    viewpoint_xyz: Tuple[float, float, float],
    phi_offset_deg: float,
    horizon_tol_deg: float,
    nx: int = 240,
    ny: int = 200,
    point_radius: int = 2,
    point_alpha: int = 120,
) -> None:
    """
    Save one preview PNG per daylight whole hour.
    Draw projected window sample points in red.
    The background outside the fisheye circle is transparent.

    """
    os.makedirs(outdir, exist_ok=True)

    cx, cy, R, H, W = proj
    date_str = get_configured_date()

    # Select one row per full hour
    dfv = df_angles.dropna(subset=["time"]).copy()
    dfv["time"] = normalize_time_to_HHMM(dfv["time"])
    dfv = dfv.dropna(subset=["time"]).copy()
    dfv = dfv[dfv["time"].str.endswith(":00")].copy()

    # --------------------------------------------------------
    # Build transparent fisheye base image
    # --------------------------------------------------------
    base_u8 = np.clip(V * 255.0, 0, 255).astype(np.uint8)

    yy, xx = np.indices((H, W))
    fisheye_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (R + 0.5) ** 2

    base_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    base_rgba[..., 0] = base_u8
    base_rgba[..., 1] = base_u8
    base_rgba[..., 2] = base_u8
    base_rgba[..., 3] = np.where(fisheye_circle, 255, 0).astype(np.uint8)

    # --------------------------------------------------------
    # Font
    # --------------------------------------------------------
    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, int(round(W / 60))))
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, int(round(W / 60))))
        except:
            font = ImageFont.load_default()

    # Point drawing offsets
    offsets = []
    r2 = point_radius * point_radius
    for oy in range(-point_radius, point_radius + 1):
        for ox in range(-point_radius, point_radius + 1):
            if ox * ox + oy * oy <= r2:
                offsets.append((ox, oy))

    for _, row in dfv.iterrows():
        if not bool(row["valid"]):
            continue

        hhmm = str(row["time"])
        theta_s = float(row["theta_s_rad"])
        phi_s = float(row["phi_s_rad"])

        if np.isnan(theta_s) or np.isnan(phi_s):
            continue

        sx, sy, sz = sun_vec_local(theta_s, phi_s)
        elev = np.pi / 2 - theta_s
        if not (elev > 0 and sz > 0):
            continue

        X0, Y0 = center_shift_on_zD(theta_s, phi_s, D_eff)
        if X0 is None:
            continue

        a, b = window_w / 2.0, window_h / 2.0
        xs = np.linspace(-a, a, nx)
        ys = np.linspace(-b, b, ny)
        Xg, Yg = np.meshgrid(xs, ys)

        X = X0 + Xg
        Y = Y0 + Yg

        theta_t, phi_t = xy_to_angles_with_viewpoint(
            X,
            Y,
            D_eff,
            viewpoint_xyz,
            phi_offset_deg
        )

        r = R * np.sin(theta_t)
        px = cx + r * np.sin(phi_t)
        py = cy - r * np.cos(phi_t)

        tol = np.deg2rad(float(horizon_tol_deg))
        inside = (
            (theta_t >= 0.0) &
            (theta_t <= (np.pi / 2 + tol)) &
            (px >= 0) & (px < W) &
            (py >= 0) & (py < H) &
            ((px - cx) ** 2 + (py - cy) ** 2 <= (R + 0.5) ** 2)
        )

        if np.count_nonzero(inside) < 50:
            continue

        # Start from transparent fisheye base
        img = Image.fromarray(base_rgba).convert("RGBA")

        # --------------------------------------------------------
        # Draw projected window points in red
        # --------------------------------------------------------
        overlay = np.zeros((H, W, 4), dtype=np.uint8)

        xi0 = np.clip(np.round(px[inside]).astype(np.int32), 0, W - 1)
        yi0 = np.clip(np.round(py[inside]).astype(np.int32), 0, H - 1)

        for ox, oy in offsets:
            xi = xi0 + ox
            yi = yi0 + oy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            if not np.any(ok):
                continue

            overlay[yi[ok], xi[ok], 0] = 255
            overlay[yi[ok], xi[ok], 1] = 0
            overlay[yi[ok], xi[ok], 2] = 0
            overlay[yi[ok], xi[ok], 3] = point_alpha

        img = Image.alpha_composite(img, Image.fromarray(overlay))
        draw = ImageDraw.Draw(img)

        # --------------------------------------------------------
        # Time stamp
        # --------------------------------------------------------
        stamp = f"{date_str} {hhmm}" if date_str else hhmm

        try:
            bbox = draw.textbbox((0, 0), stamp, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (len(stamp) * 10, 20)

        margin = max(10, int(round(W / 80)))

        text_x = int(cx + 0.35 * R)
        text_y = int(cy + 0.72 * R)

        text_x = max(margin, min(W - tw - margin, text_x))
        text_y = max(margin, min(H - th - margin, text_y))

        filename = f"window_{hhmm[:2]}.png"

        savepath = os.path.join(outdir, filename)
        img.save(savepath)


# ============================================================
# Main
# ============================================================

def run() -> Dict[str, Any]:
    """
    Main entry of Step5.
    """
    print("\n[Step 5/6] Computing direct window gain")

    # --------------------------------------------------------
    # Skip control case
    # --------------------------------------------------------
    is_control_case = bool(getattr(cfg, "is_control_case", False))
    if is_control_case:
        print("Skipped: Direct window gain is not required for a control case.")
        print("[Step 5/6] Completed")
        return {
            "skipped": True,
            "reason": "control_case"
        }

    # --------------------------------------------------------
    # Resolve paths and parameters
    # --------------------------------------------------------
    paths = resolve_paths()
    p = params_from_cfg()

    cam_dx, cam_dy, cam_dz = p.cam_offset_xyz
    print(f"Effective shading-plane distance: {p.D_effective_m:.4f} m")
    print(f"Camera offset: ({cam_dx:.4f}, {cam_dy:.4f}, {cam_dz:.4f}) m")

    # --------------------------------------------------------
    # Load Step4 sunpath table
    # --------------------------------------------------------
    df_angles = read_step4_sunpath_excel(paths.sunpath_xlsx_path)

    # --------------------------------------------------------
    # Load Step3 raw sky mask
    # --------------------------------------------------------
    V, proj = load_sky_mask(paths.sky_mask_npy_path)

    # --------------------------------------------------------
    # Compute gain per timestamp
    # --------------------------------------------------------
    theta_arr = df_angles["theta_s_rad"].to_numpy()
    phi_arr = df_angles["phi_s_rad"].to_numpy()

    gains = []
    for th, ph in zip(theta_arr, phi_arr):
        thf = float(th) if np.isfinite(th) else np.nan
        phf = float(ph) if np.isfinite(ph) else np.nan

        g = window_gain_raster(
            theta_s=thf,
            phi_s=phf,
            V=V,
            proj=proj,
            window_w=p.window_width_m,
            window_h=p.window_height_m,
            D_eff=p.D_effective_m,
            viewpoint_xyz=p.cam_offset_xyz,
            phi_offset_deg=p.phi_offset_deg,
            horizon_tol_deg=p.horizon_tol_deg,
            nx=p.raster_nx,
            ny=p.raster_ny,
        )
        gains.append(g)

    # --------------------------------------------------------
    # Build output table
    # --------------------------------------------------------
    out = pd.DataFrame()
    out[p.time_col] = df_angles["time"].astype(str).to_numpy()
    out[p.gain_col] = np.array(gains, dtype=float)

    if p.full_day_fill:
        out = pad_full_day_minutes(
            out,
            time_col=p.time_col,
            cols_to_fill_zero=[p.gain_col]
        )

    out.to_excel(paths.out_xlsx_path, index=False)

    # --------------------------------------------------------
    # Hourly previews
    # --------------------------------------------------------
    if p.save_hourly_previews:
        save_hourly_previews(
            df_angles=df_angles,
            V=V,
            proj=proj,
            outdir=paths.preview_dir,
            basename=paths.basename,
            window_w=p.window_width_m,
            window_h=p.window_height_m,
            D_eff=p.D_effective_m,
            viewpoint_xyz=p.cam_offset_xyz,
            phi_offset_deg=p.phi_offset_deg,
            horizon_tol_deg=p.horizon_tol_deg,
            nx=max(160, p.raster_nx),
            ny=max(140, p.raster_ny),
            point_radius=p.preview_point_radius,
            point_alpha=p.preview_point_alpha,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------
    print(f"Saved to: {paths.step5_dir}")
    print("[Step 5/6] Completed")

    return {
        "step5_dir": paths.step5_dir,
        "basename": paths.basename,
        "gain_xlsx_path": paths.out_xlsx_path,
        "preview_dir": paths.preview_dir,
    }


if __name__ == "__main__":
    run()

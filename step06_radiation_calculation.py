# ====================================================================
# Step 6 - Radiation Calculation
# ====================================================================
#
# Purpose
# -------
# This step computes the final vertical radiation components based on:
#   - weather data
#   - direct-shading input from Step 4 or Step 5
#   - sky mask for diffuse radiation
#   - ground mask for reflected radiation
#
# In this workflow:
#   - control case -> direct input comes from Step 4 point shading
#   - tree case    -> direct input comes from Step 5 window gain
#   - station_has_fixed_shading=False -> diffuse uses raw Sky_mask
#   - station_has_fixed_shading=True  -> diffuse uses Sky_mask_diffuse
#   - reflected radiation uses raw Ground_mask
#
# Outputs
# -------
# - *_radiation_output.xlsx
#
# ====================================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pvlib
from pvlib.location import Location
from pvlib.atmosphere import get_relative_airmass
from pvlib.tools import cosd

import config as cfg


# ============================================================
# Perez all-weather model (1993)
# ============================================================

# === Five-coefficient shape (a, b, c, d, e) ===

_P93_EPS_BINS = [1.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2, 100.0]

# Table: a1..a4, b1..b4, c1..c4, d1..d4, e1..e4
_P93_COEFFS = {
    0: {
        "a": ( 1.3525, -0.2576, -0.2690, -1.4366),
        "b": (-0.7670,  0.0007,  1.2734, -0.1233),
        "c": ( 2.8000,  0.6004,  1.2375,  1.0000),
        "d": ( 1.8734,  0.6297,  0.9738,  0.2809),
        "e": ( 0.0356, -0.1246, -0.5718,  0.9938),
    },
    1: {
        "a": (-1.2219, -0.7730,  1.4148,  1.1016),
        "b": (-0.2054,  0.0367, -3.9128,  0.9156),
        "c": ( 6.9750,  0.1774,  6.4477, -0.1239),
        "d": (-1.5798, -0.5081, -1.7812,  0.1080),
        "e": ( 0.2624,  0.0672, -0.2190, -0.4285),
    },
    2: {
        "a": (-1.1000, -0.2515,  0.8952,  0.0156),
        "b": ( 0.2782, -0.1812, -4.5000,  1.1766),
        "c": (24.7219,-13.0812,-37.7000, 34.8438),
        "d": (-5.0000,  1.5218,  3.9229, -2.6204),
        "e": (-0.0156,  0.1597,  0.4199, -0.5562),
    },
    3: {
        "a": (-0.5484, -0.6654, -0.2672,  0.7117),
        "b": ( 0.7234, -0.6219, -5.6812,  2.6297),
        "c": (33.3389,-18.3000,-62.2500, 52.0781),
        "d": (-3.5000,  0.0016,  1.1477,  0.1062),
        "e": ( 0.4659, -0.3296, -0.0876, -0.0329),
    },
    4: {
        "a": (-0.6000, -0.3566, -2.5000,  2.3250),
        "b": ( 0.2937,  0.0496, -5.6812,  1.8415),
        "c": (21.0000, -4.7656,-21.5906,  7.2492),
        "d": (-3.5000, -0.1554,  1.4062,  0.3988),
        "e": ( 0.0032,  0.0766, -0.0656, -0.1294),
    },
    5: {
        "a": (-1.0156, -0.3670,  1.0078,  1.4051),
        "b": ( 0.2875, -0.5328, -3.8500,  3.3750),
        "c": (14.0000, -0.9999, -7.1406,  7.5469),
        "d": (-3.4000, -0.1078, -1.0750,  1.5702),
        "e": (-0.0672,  0.4016,  0.3017, -0.4844),
    },
    6: {
        "a": (-1.0000,  0.0211,  0.5025, -0.5119),
        "b": (-0.3000,  0.1922,  0.7023, -1.6317),
        "c": (19.0000, -5.0000,  1.2438, -1.9094),
        "d": (-4.0000,  0.0250,  0.3844,  0.2656),
        "e": ( 1.0468, -0.3788, -2.4517,  1.4656),
    },
    7: {
        "a": (-1.0500,  0.0289,  0.4260,  0.3590),
        "b": (-0.3250,  0.1156,  0.7781,  0.0025),
        "c": (31.0625,-14.5000,-46.1148, 55.3750),
        "d": (-7.2312,  0.4050, 13.3500,  0.6234),
        "e": ( 1.5000, -0.6426,  1.8564,  0.5636),
    },
}

# c = exp((Δ*(c1 + c2*Z))**c3) - c4
# d = -exp(Δ*(d1 + d2*Z)) + d3 + Δ*d4
_P93_FIRST_BIN_EXCEPT = True

def _p93_pick_bin(eps: float) -> int:
    eps = float(np.clip(eps, 1.0, 99.999))
    idx = np.digitize(eps, _P93_EPS_BINS, right=True) - 1
    return int(np.clip(idx, 0, 7))

def _p93_linear_form(params, Z, Delta):
    # params = (x1, x2, x3, x4)
    x1, x2, x3, x4 = params
    return x1 + x2*Z + Delta*(x3 + x4*Z)

def compute_p93_ae(epsilon, Delta, Z_rad):

    idx = _p93_pick_bin(epsilon)
    coeffs = _P93_COEFFS[idx]

    Z = Z_rad

    a = _p93_linear_form(coeffs["a"], Z, Delta)
    b = _p93_linear_form(coeffs["b"], Z, Delta)

    if _P93_FIRST_BIN_EXCEPT and idx == 0:
        c1, c2, c3, c4 = coeffs["c"]
        d1, d2, d3, d4 = coeffs["d"]
        # c = exp( [Δ*(c1 + c2*Z)]^(c3) ) - c4
        c = np.exp((Delta * (c1 + c2 * Z)) ** c3) - c4
        # d = -exp( Δ*(d1 + d2*Z) ) + d3 + Δ*d4
        d = -np.exp(Delta * (d1 + d2 * Z)) + d3 + Delta * d4
    else:
        c = _p93_linear_form(coeffs["c"], Z, Delta)
        d = _p93_linear_form(coeffs["d"], Z, Delta)

    e = _p93_linear_form(coeffs["e"], Z, Delta)

    return float(a), float(b), float(c), float(d), float(e)

def compute_L_perez1993_Gd(theta_map, phi_map, theta_s, phi_s,
                           dhi, dni, Z_rad,
                           dni_extra_t, airmass_rel_t):

    if (dhi is None) or (dhi <= 0) or (np.cos(Z_rad) <= 0):
        return np.zeros_like(theta_map)

    # Δ = (DHI * m) / I0
    dni_extra_val = float(np.asarray(dni_extra_t).ravel()[0])
    am_rel = float(np.asarray(airmass_rel_t).ravel()[0])
    Delta = (float(dhi) * am_rel) / dni_extra_val

    # ε(clearness)
    dni_eff = max(0.0, float(dni))
    K = 1.041
    epsilon = ((dhi + dni_eff) / dhi + K * (Z_rad**3)) / (1.0 + K * (Z_rad**3))

    a, b, c, d, e = compute_p93_ae(epsilon, Delta, Z_rad)

    cos_theta = np.clip(np.cos(theta_map), 1e-6, None)
    gamma = compute_gamma(theta_map, phi_map, theta_s, phi_s)
    I = (1.0 + a * np.exp(b / cos_theta)) * (1.0 + c * np.exp(d * gamma) + e * (np.cos(gamma) ** 2))

    dtheta = np.abs(theta_map[1, 0] - theta_map[0, 0])
    dphi   = np.abs(phi_map[0, 1] - phi_map[0, 0])
    dOmega = np.sin(theta_map) * dtheta * dphi

    denom = float(np.sum(I * np.cos(theta_map) * dOmega)) + 1e-12
    scale = float(dhi) / denom
    L = scale * I

    return L

def compute_gamma(theta, phi, theta_s, phi_s):
    return np.arccos(np.clip(
        np.cos(theta) * np.cos(theta_s) +
        np.sin(theta) * np.sin(theta_s) * np.cos(phi - phi_s),
        -1, 1
    ))

def compute_vertical_direct_with_shading(dni, theta_s, phi_s,
                                         window_gain, wall_azimuth_deg=180):

    if dni is None or dni <= 0 or np.isnan(dni):
        return 0.0
    if np.isnan(theta_s) or np.isnan(phi_s):
        return 0.0
    phi_wall = np.radians(wall_azimuth_deg)
    cos_gamma = np.sin(theta_s) * np.cos(phi_s - phi_wall)
    if cos_gamma <= 0:
        return 0.0
    wg = 0.0 if (window_gain is None or np.isnan(window_gain)) else float(window_gain)
    return float(dni * cos_gamma * wg)

def compute_vertical_diffuse(L, theta_map, phi_map, wall_azimuth_deg=180):

    phi_wall = np.radians(wall_azimuth_deg)
    cos_gamma = np.sin(theta_map) * np.cos(phi_map - phi_wall)
    cos_gamma = np.maximum(0, cos_gamma)

    dtheta = np.abs(theta_map[1,0] - theta_map[0,0])
    dphi = np.abs(phi_map[0,1] - phi_map[0,0])
    dOmega = np.sin(theta_map) * dtheta * dphi

    return np.sum(L * cos_gamma * dOmega)

def compute_vertical_diffuse_with_shading(L, theta_map, phi_map, V_mask, wall_azimuth_deg=180):

    phi_wall = np.radians(wall_azimuth_deg)

    cos_gamma = np.sin(theta_map) * np.cos(phi_map - phi_wall)
    cos_gamma = np.maximum(0, cos_gamma)

    dtheta = np.abs(theta_map[1,0] - theta_map[0,0])
    dphi = np.abs(phi_map[0,1] - phi_map[0,0])
    dOmega = np.sin(theta_map) * dtheta * dphi

    contribution = L * cos_gamma * dOmega * V_mask
    vertical_diffuse_shaded = np.sum(contribution)

    return vertical_diffuse_shaded

def compute_horizontal_diffuse(L, theta_map, phi_map):

    cos_theta = np.cos(theta_map)
    dtheta = np.abs(theta_map[1,0] - theta_map[0,0])
    dphi = np.abs(phi_map[0,1] - phi_map[0,0])
    dOmega = np.sin(theta_map) * dtheta * dphi

    return np.sum(L * cos_theta * dOmega)

def save_L_overlay_orthographic(
    L,
    theta_deg,
    phi_deg,
    diffuse_mask_px,
    basename,
    time_label,
    folder,
    theta_s_deg,
    phi_s_deg,
    vmax_global,
):

    cmap = plt.get_cmap("turbo")

    mask = np.asarray(diffuse_mask_px).astype(np.float32)
    if mask.ndim != 2:
        raise ValueError("diffuse_mask_px must be a 2D array.")

    h, w = mask.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    R = 0.5 * float(min(h, w))

    yy, xx = np.indices((h, w))
    dx = xx - cx
    dy = cy - yy   # positive upward
    r = np.sqrt(dx ** 2 + dy ** 2)

    inside = r <= R + 1e-6
    r_norm = np.clip(r / R, 0.0, 1.0)

    # Orthographic sky fisheye: r = R * sin(theta)
    theta_img_deg = np.degrees(np.arcsin(r_norm))
    phi_img_deg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    if len(theta_deg) < 2 or len(phi_deg) < 2:
        raise ValueError("theta_deg / phi_deg must contain at least 2 values.")

    step_theta = float(theta_deg[1] - theta_deg[0])
    step_phi = float(phi_deg[1] - phi_deg[0])

    ti = np.clip(np.rint(theta_img_deg / step_theta).astype(int), 0, L.shape[0] - 1)
    pi = np.mod(np.rint(phi_img_deg / step_phi).astype(int), L.shape[1])

    # Use one shared scale for all hourly images
    vmax = max(float(vmax_global), 1e-12)

    L_img = np.zeros((h, w), dtype=np.float32)
    L_img[inside] = L[ti[inside], pi[inside]]

    L_norm = np.clip(L_img / vmax, 0.0, 1.0)

    heat_rgb = (cmap(L_norm)[..., :3] * 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    rgba[inside, 0:3] = heat_rgb[inside]
    rgba[inside, 3] = 255

    blocked = inside & (mask <= 0.5)
    if np.any(blocked):
        rgb_blocked = rgba[blocked, 0:3].astype(np.float32)
        rgb_blocked *= 0.22
        rgba[blocked, 0:3] = np.clip(rgb_blocked, 0, 255).astype(np.uint8)

    visible = inside & (mask > 0.5)
    if np.any(visible):
        rgb_visible = rgba[visible, 0:3].astype(np.float32)
        rgb_visible = 0.92 * rgb_visible + 0.08 * 255.0
        rgba[visible, 0:3] = np.clip(rgb_visible, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgba)
    draw = ImageDraw.Draw(img)

    if theta_s_deg is not None and phi_s_deg is not None and 0.0 <= theta_s_deg <= 90.0:
        r_s = R * np.sin(np.radians(theta_s_deg))
        x_s = cx + r_s * np.sin(np.radians(phi_s_deg))
        y_s = cy - r_s * np.cos(np.radians(phi_s_deg))

        sun_r = max(4, int(round(h * 0.006)))
        draw.ellipse(
            (x_s - sun_r, y_s - sun_r, x_s + sun_r, y_s + sun_r),
            fill=(255, 0, 0, 255)
        )

    save_path = os.path.join(folder, f"{basename}_DiffuseOverlay_{time_label}.png")
    img.save(save_path)

def save_diffuse_colorbar_horizontal(
    folder,
    basename,
    vmax_global,
    width_px,
    height_px=140,
):

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    vmax = max(float(vmax_global), 1e-12)

    dpi = 180
    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([0.06, 0.40, 0.88, 0.26])
    ax.set_facecolor((1, 1, 1, 0))

    sm = ScalarMappable(cmap="turbo", norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Diffuse sky luminance L (W/m²·sr)", fontsize=18)
    cbar.ax.tick_params(labelsize=16, length=6, width=1.2)

    save_path = os.path.join(folder, f"{basename}_DiffuseOverlay_Colorbar.png")
    plt.savefig(
        save_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05
    )
    plt.close(fig)

def merge_point_gain_by_minute(
    data: pd.DataFrame,
    ts_col: str,
    point_gain_path: str,
    pg_time_col: str = "time",
    pg_value_col: str = "unshaded",
    out_col: str = "Point Gain (direct)",
) -> pd.DataFrame:

    if point_gain_path is None or (not os.path.exists(point_gain_path)):
        raise FileNotFoundError("Step 4 point-gain file was not found.")

    # Step4 output is Excel: *_Sunpath_Shading.xlsx
    pg = pd.read_excel(point_gain_path)

    if pg_time_col not in pg.columns or pg_value_col not in pg.columns:
        raise ValueError(
            f"Point gain file missing required columns: {pg_time_col}, {pg_value_col}. "
            f"Got columns={list(pg.columns)}"
        )

    pg = pg[[pg_time_col, pg_value_col]].copy()
    pg["__minute__"] = pg[pg_time_col].astype(str).str.slice(0, 5)  # HH:MM
    pg = pg.drop_duplicates("__minute__", keep="last")

    tmp = data.copy()
    tmp["__minute__"] = pd.to_datetime(tmp[ts_col]).dt.strftime("%H:%M")
    tmp = tmp.merge(pg[["__minute__", pg_value_col]], on="__minute__", how="left")

    tmp[out_col] = tmp[pg_value_col].fillna(0.0).astype(float)
    tmp = tmp.drop(columns=["__minute__", pg_value_col], errors="ignore")
    return tmp

def merge_window_gain_by_minute(main_df, ts_col, window_gain_path,
                                wg_time_col="time", wg_value_col="Window Gain (direct)"):

    df = main_df.copy()
    t = pd.to_datetime(df[ts_col], errors="coerce")
    df["TimeHHMM"] = t.dt.strftime("%H:%M")

    if str(window_gain_path).lower().endswith(".csv"):
        wg = pd.read_csv(window_gain_path)
    else:
        wg = pd.read_excel(window_gain_path)

    if wg_time_col not in wg.columns:
        raise ValueError(f"WindowGain file is missing the time column: {wg_time_col}")
    if wg_value_col not in wg.columns:
        raise ValueError(f"WindowGain file is missing the gain column: {wg_value_col}")
    wg = wg.rename(columns={wg_time_col: "TimeHHMM"})
    wg["TimeHHMM"] = wg["TimeHHMM"].astype(str).str.slice(0,5)
    wg = wg[["TimeHHMM", wg_value_col]].drop_duplicates(subset=["TimeHHMM"], keep="first")

    df = df.merge(wg, on="TimeHHMM", how="left")
    df[wg_value_col] = df[wg_value_col].fillna(0.0).clip(0, 1)
    return df

def precompute_down_weights(shape, wall_azimuth_deg=180):
    """
      w = max(0, sinθ * cos(φ-φ_wall)) * sinθ * dθ * dφ
    """
    import numpy as np
    n_theta, n_phi = int(shape[0]), int(shape[1])

    # θ: [π/2, π],φ: [0, 2π)
    theta_vals = np.linspace(np.pi/2, np.pi, n_theta)
    phi_vals   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta_vals, phi_vals, indexing='ij')

    dtheta = (theta_vals[1]-theta_vals[0]) if n_theta > 1 else (np.pi/2)
    dphi   = (phi_vals[1]-phi_vals[0])     if n_phi   > 1 else (2*np.pi)

    phi_wall = np.radians(wall_azimuth_deg)
    cos_gamma = np.sin(TH) * np.cos(PH - phi_wall)
    cos_gamma = np.maximum(0.0, cos_gamma)

    dOmega = np.sin(TH) * dtheta * dphi
    W = cos_gamma * dOmega
    return W

# =========================
# Cache for down-hemisphere weights
# =========================
_DOWN_WEIGHTS_CACHE = {}

def get_down_weights(shape, wall_azimuth_deg=180):
    key = (int(shape[0]), int(shape[1]), int(wall_azimuth_deg))
    W = _DOWN_WEIGHTS_CACHE.get(key)
    if W is None:
        W = precompute_down_weights(shape, wall_azimuth_deg)
        _DOWN_WEIGHTS_CACHE[key] = W
    return W


# ============================================================
# Grid / mask helpers
# ============================================================

def _choose_nice_step_deg(step_min_deg, candidates):
    cands = sorted(float(x) for x in candidates)
    for s in cands:
        if s >= step_min_deg - 1e-12:
            return float(s)
    return float(cands[-1])

def infer_theta_phi_step_deg_from_mask(mask_shape, candidates, auto_min_step_deg=None):
    h, w = int(mask_shape[0]), int(mask_shape[1])
    R = 0.5 * float(min(h, w))  # effective fisheye radius in pixels
    if R <= 0:
        return float(candidates[0])
    step_min = 90.0 / R  # Δθ_px ≈ 90° / R

    if auto_min_step_deg is not None:
        step_min = max(step_min, float(auto_min_step_deg))

    return _choose_nice_step_deg(step_min, candidates)

def build_theta_phi_maps(step_deg, hemisphere="sky"):
    step_deg = float(step_deg)

    if hemisphere.lower() == "sky":
        theta_deg = np.arange(0.0, 90.0 + 1e-9, step_deg)
    elif hemisphere.lower() == "ground":
        theta_deg = np.arange(90.0, 180.0 + 1e-9, step_deg)
    else:
        raise ValueError("hemisphere must be 'sky' or 'ground'")

    phi_deg = np.arange(0.0, 360.0, step_deg)

    theta_map, phi_map = np.meshgrid(np.radians(theta_deg), np.radians(phi_deg), indexing="ij")
    return theta_deg, phi_deg, theta_map, phi_map

def sample_pixel_mask_to_theta_phi(mask_px, theta_deg, phi_deg, hemisphere="sky"):
    mask = np.asarray(mask_px)
    if mask.ndim != 2:
        raise ValueError("mask_px must be a 2D array (H,W).")
    h, w = mask.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    R = 0.5 * float(min(h, w))

    TH, PH = np.meshgrid(theta_deg, phi_deg, indexing="ij")

    THr = np.radians(TH)  # degrees -> radians
    r_norm = np.sin(THr)
    r_norm = np.clip(r_norm, 0.0, 1.0)

    r_px = r_norm * R

    phir = np.radians(PH)
    x = cx + r_px * np.sin(phir)
    y = cy - r_px * np.cos(phir)

    xi = np.clip(np.rint(x).astype(int), 0, w - 1)
    yi = np.clip(np.rint(y).astype(int), 0, h - 1)

    V = mask[yi, xi]

    if V.dtype != np.bool_:
        V = (V > 0).astype(np.float32)
    else:
        V = V.astype(np.float32)

    return V


# ============================================================
# Helpers
# ============================================================

@dataclass
class Step6Paths:
    image_path: str
    basename: str
    case_dir: str
    step3_dir: str
    step4_dir: str
    step5_dir: str
    step6_dir: str
    weather_path: str
    sky_mask_path: str
    sky_mask_diffuse_path: str
    ground_mask_path: str
    step4_sunpath_xlsx_path: str
    step5_window_gain_xlsx_path: Optional[str]
    out_xlsx_path: str

def resolve_paths() -> Step6Paths:
    image_path = getattr(cfg, "image_path", None)
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    weather_path = getattr(cfg, "weather_data_path", None)
    if not weather_path or not os.path.exists(weather_path):
        raise FileNotFoundError(f"Weather data not found: {weather_path}")

    image_path = os.path.abspath(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    case_dir = os.path.dirname(image_path)

    step3_dir = os.path.join(case_dir, "03 Binarized images")
    step4_dir = os.path.join(case_dir, "04 Sun path")
    step5_dir = os.path.join(case_dir, "05 Window Direct Gain")
    step6_dir = os.path.join(case_dir, "06 Radiation Results")
    os.makedirs(step6_dir, exist_ok=True)

    sky_mask_path = os.path.join(step3_dir, f"{basename}_Sky_mask.npy")
    sky_mask_diffuse_path = os.path.join(step3_dir, f"{basename}_Sky_mask_diffuse.npy")
    ground_mask_path = os.path.join(step3_dir, f"{basename}_Ground_mask.npy")
    step4_sunpath_xlsx_path = os.path.join(step4_dir, f"{basename}_Sunpath_Shading.xlsx")
    step5_window_gain_xlsx_path = os.path.join(step5_dir, f"{basename}_window_direct_gain.xlsx")
    out_xlsx_path = os.path.join(step6_dir, f"{basename}_radiation_output.xlsx")

    if not os.path.exists(sky_mask_path):
        raise FileNotFoundError(f"Missing Step3 raw sky mask: {sky_mask_path}")
    if not os.path.exists(ground_mask_path):
        raise FileNotFoundError(f"Missing Step3 ground mask: {ground_mask_path}")
    if not os.path.exists(step4_sunpath_xlsx_path):
        raise FileNotFoundError(f"Missing Step4 sunpath file: {step4_sunpath_xlsx_path}")

    if not os.path.exists(step5_window_gain_xlsx_path):
        step5_window_gain_xlsx_path = None

    return Step6Paths(
        image_path=image_path,
        basename=basename,
        case_dir=case_dir,
        step3_dir=step3_dir,
        step4_dir=step4_dir,
        step5_dir=step5_dir,
        step6_dir=step6_dir,
        weather_path=weather_path,
        sky_mask_path=sky_mask_path,
        sky_mask_diffuse_path=sky_mask_diffuse_path,
        ground_mask_path=ground_mask_path,
        step4_sunpath_xlsx_path=step4_sunpath_xlsx_path,
        step5_window_gain_xlsx_path=step5_window_gain_xlsx_path,
        out_xlsx_path=out_xlsx_path,
    )

def load_mask01(mask_path: str) -> np.ndarray:
    arr = np.load(mask_path)
    arr = np.asarray(arr)
    if arr.max() > 1.5:
        return (arr > 127).astype(np.float32)
    return (arr > 0.5).astype(np.float32)


# ============================================================
# Main Step6
# ============================================================

def run_step6_radiation():
    print("\n[Step 6/6] Calculating irradiance")

    # ------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------
    paths = resolve_paths()

    is_control_case = bool(getattr(cfg, "is_control_case", False))
    station_has_fixed_shading = bool(getattr(cfg, "station_has_fixed_shading", False))

    # Weather columns.
    date_col = str(getattr(cfg, "weather_date_col", "Date"))
    time_col = str(getattr(cfg, "weather_time_col", "Time"))
    ghi_col_cfg = getattr(cfg, "weather_ghi_col", None)
    ghi_col = str(ghi_col_cfg).strip() if ghi_col_cfg else None

    # Surface properties
    surface_tilt = float(getattr(cfg, "surface_tilt_deg", 90.0))
    surface_azimuth = float(getattr(cfg, "surface_azimuth_deg", 180.0))
    albedo = float(getattr(cfg, "albedo", 0.2))

    # Location
    loc = Location(cfg.latitude, cfg.longitude, tz=cfg.timezone, altitude=cfg.altitude)

    # ------------------------------------------------------------
    # Load masks
    # ------------------------------------------------------------
    print("Loading sky and ground masks...")
    V_sky_raw_px = load_mask01(paths.sky_mask_path)
    V_ground_px = load_mask01(paths.ground_mask_path)

    if station_has_fixed_shading:
        if not os.path.exists(paths.sky_mask_diffuse_path):
            raise FileNotFoundError(
                f"station_has_fixed_shading=True, but Sky_mask_diffuse not found:\n{paths.sky_mask_diffuse_path}"
            )
        V_sky_for_diffuse_px = load_mask01(paths.sky_mask_diffuse_path)
    else:
        V_sky_for_diffuse_px = V_sky_raw_px.copy()


    # ------------------------------------------------------------
    # Read weather data properly and build Timestamp
    # ------------------------------------------------------------
    print("Loading weather data...")
    data = pd.read_excel(paths.weather_path)

    missing_time_columns = [c for c in (date_col, time_col) if c not in data.columns]
    if missing_time_columns:
        raise ValueError(
            "Weather data is missing the configured date/time column(s): "
            f"{missing_time_columns}. Available columns: {list(data.columns)}"
        )

    # Auto-detect the GHI column if it is not explicitly configured.
    if ghi_col is None:
        _common_ghi_names = [
            "Weather station", "Weather Station", "GHI", "GHI(W/m2)", "GHI (W/m2)",
            "Global", "Global Horizontal", "Global Horizontal Irradiance",
            "Solar Rad.", "Solar", "Rad.", "Radiation", "HSR"
        ]
        for _name in _common_ghi_names:
            if _name in data.columns:
                ghi_col = _name
                break
    if ghi_col is None:
        raise ValueError(
            "Please set cfg.weather_ghi_col (GHI column name) in config.py "
            "or make sure your weather file contains a recognized GHI column."
        )
    if ghi_col not in data.columns:
        raise ValueError(
            f"Configured GHI column '{ghi_col}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    # Build timestamp
    ts = pd.to_datetime(data[date_col].astype(str) + " " + data[time_col].astype(str), errors="coerce")
    if ts.isna().any():
        bad_rows = list(data.index[ts.isna()][:10])
        raise ValueError(
            "Some Date/Time values could not be parsed. "
            f"First affected row indices: {bad_rows}"
        )
    data["Timestamp"] = ts.dt.tz_localize(cfg.timezone, nonexistent="shift_forward", ambiguous="NaT")

    data[ghi_col] = pd.to_numeric(data[ghi_col], errors="coerce")

    # ------------------------------------------------------------
    # Decide direct-gain source and merge after Timestamp is ready
    # ------------------------------------------------------------
    if is_control_case:
        direct_gain_col = "Point Gain (direct)"
        data = merge_point_gain_by_minute(
            data,
            ts_col="Timestamp",
            point_gain_path=paths.step4_sunpath_xlsx_path,
            pg_time_col="time",
            pg_value_col="unshaded",
            out_col=direct_gain_col,
        )

        t = pd.to_datetime(data["Timestamp"], errors="coerce")
        data["TimeHHMM"] = t.dt.strftime("%H:%M")

    else:
        if paths.step5_window_gain_xlsx_path is None:
            raise FileNotFoundError(
                "Tree case requires Step5 window gain file, but it was not found."
            )

        direct_gain_col = "Window Gain (direct)"
        data = merge_window_gain_by_minute(
            data,
            ts_col="Timestamp",
            window_gain_path=paths.step5_window_gain_xlsx_path,
            wg_time_col="time",
            wg_value_col=direct_gain_col
        )

    # ------------------------------------------------------------
    # Decide theta/phi step
    # ------------------------------------------------------------
    candidates = [0.05, 0.1, 0.125, 0.2, 0.25, 0.3, 0.5, 1.0]
    auto_min = 0.2

    step_deg = infer_theta_phi_step_deg_from_mask(
        np.asarray(V_sky_for_diffuse_px).shape,
        candidates,
        auto_min_step_deg=auto_min
    )

    # Build theta/phi maps
    theta_deg_sky, phi_deg, theta_map, phi_map = build_theta_phi_maps(step_deg, hemisphere="sky")
    theta_deg_ground, phi_deg_g, theta_map_g, phi_map_g = build_theta_phi_maps(step_deg, hemisphere="ground")

    # Sample pixel masks onto theta/phi grids
    V_sky_mask = sample_pixel_mask_to_theta_phi(V_sky_for_diffuse_px, theta_deg_sky, phi_deg, hemisphere="sky")
    Vr_mask = sample_pixel_mask_to_theta_phi(V_ground_px, theta_deg_ground, phi_deg_g, hemisphere="ground")

    # Precompute ground weights
    W_down = get_down_weights(Vr_mask.shape, wall_azimuth_deg=int(surface_azimuth))
    SUMW = float(np.sum(W_down))
    SUMW_SHADED = float(np.sum(W_down * np.nan_to_num(Vr_mask, nan=0.0)))

    # Hourly diffuse-overlay output
    overlay_folder = os.path.join(paths.step6_dir, "Hourly_Diffuse_Overlay")
    os.makedirs(overlay_folder, exist_ok=True)

    hourly_overlay_records = []
    global_overlay_vmax = 0.0

    # Init result lists
    poa_global_list = []
    poa_direct_list = []
    poa_sky_diffuse_list = []
    poa_ground_diffuse_list = []
    horizontal_direct_list = []
    horizontal_diffuse_list = []
    vertical_direct_with_shading_list = []
    vertical_diffuse_L_list = []
    vertical_diffuse_with_shading_list = []
    vertical_reflect_L_list = []
    vertical_reflect_direct_list = []
    vertical_reflect_L_shaded_list = []
    vertical_reflect_direct_shaded_list = []

    # ------------------------------------------------------------
    # Main row loop
    # ------------------------------------------------------------
    n_total = len(data)
    progress_marks = {
        max(1, int(n_total * 0.25)),
        max(1, int(n_total * 0.50)),
        max(1, int(n_total * 0.75)),
        n_total,
    }

    for i, (_, row) in enumerate(data.iterrows(), start=1):
        if i in progress_marks:
            pct = int(round(i / n_total * 100))
            print(f"Progress: {pct}%")

        timestamp = row["Timestamp"]
        ghi = row.get(ghi_col, None)

        if ghi is None or pd.isna(ghi) or float(ghi) <= 0:
            poa_global_list.append(0)
            poa_direct_list.append(0)
            poa_sky_diffuse_list.append(0)
            poa_ground_diffuse_list.append(0)
            horizontal_direct_list.append(0)
            horizontal_diffuse_list.append(0)
            vertical_direct_with_shading_list.append(0)
            vertical_diffuse_L_list.append(0)
            vertical_diffuse_with_shading_list.append(0)
            vertical_reflect_L_list.append(0)
            vertical_reflect_direct_list.append(0)
            vertical_reflect_L_shaded_list.append(0)
            vertical_reflect_direct_shaded_list.append(0)
            continue

        ghi = float(ghi)

        # Solar position (pvlib)
        solar_position = loc.get_solarposition(pd.DatetimeIndex([timestamp]))
        zenith = float(solar_position['apparent_zenith'].iloc[0])
        azimuth = float(solar_position['azimuth'].iloc[0])

        # Night / sun below horizon
        if np.cos(np.radians(zenith)) <= 0:
            poa_global_list.append(0)
            poa_direct_list.append(0)
            poa_sky_diffuse_list.append(0)
            poa_ground_diffuse_list.append(0)
            horizontal_direct_list.append(0)
            horizontal_diffuse_list.append(0)
            vertical_direct_with_shading_list.append(0)
            vertical_diffuse_L_list.append(0)
            vertical_diffuse_with_shading_list.append(0)
            vertical_reflect_L_list.append(0)
            vertical_reflect_direct_list.append(0)
            vertical_reflect_L_shaded_list.append(0)
            vertical_reflect_direct_shaded_list.append(0)
            continue

        # === DNI, DHI ===
        dni = pvlib.irradiance.disc(pd.Series([ghi], index=[timestamp]),
                                    solar_position['zenith'],
                                    pd.DatetimeIndex([timestamp]))['dni']
        cos_zenith = cosd(solar_position['apparent_zenith'].iloc[0])

        dni_val_local = float(dni.iloc[0])
        dni_val_local = max(dni_val_local, 0.0)

        dhi = max(ghi - dni_val_local * cos_zenith, 0.0)
        horizontal_direct = dni_val_local * cos_zenith
        horizontal_direct_list.append(horizontal_direct)
        horizontal_diffuse_list.append(dhi)

        # Perez POA reference (pvlib)
        dni_extra = pvlib.irradiance.get_extra_radiation(pd.DatetimeIndex([timestamp]))
        airmass = get_relative_airmass(solar_position['zenith'])

        total_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            dni=dni,
            ghi=pd.Series([ghi], index=[timestamp]),
            dhi=dhi,
            dni_extra=dni_extra,
            airmass=airmass,
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth'],
            albedo=albedo,
            model='perez'
        )

        poa_global_list.append(total_irradiance['poa_global'].iloc[0])
        poa_direct_list.append(total_irradiance['poa_direct'].iloc[0])
        poa_sky_diffuse_list.append(total_irradiance['poa_sky_diffuse'].iloc[0])
        poa_ground_diffuse_list.append(total_irradiance['poa_ground_diffuse'].iloc[0])

        # Perez L
        theta_s = np.radians(zenith)
        phi_s = np.radians(azimuth)

        L = compute_L_perez1993_Gd(
            theta_map, phi_map,
            theta_s, phi_s,
            dhi, dni.iloc[0], theta_s,
            dni_extra_t=dni_extra,
            airmass_rel_t=airmass
        )

        # Save hourly orthographic diffuse overlay image
        current_time = pd.to_datetime(timestamp)
        is_hourly = (current_time.minute == 0)

        if is_hourly:
            time_label = current_time.strftime("%H%M")
            hourly_overlay_records.append(
                (
                    time_label,
                    L.copy(),
                    float(np.degrees(theta_s)),
                    float(np.degrees(phi_s)),
                )
            )
            global_overlay_vmax = max(global_overlay_vmax, float(np.max(L)))

        # ----------------------------
        # Direct with shading
        # ----------------------------
        direct_gain = float(row.get(direct_gain_col, 0.0))
        vertical_direct_with_shading = compute_vertical_direct_with_shading(
            dni=dni.iloc[0],
            theta_s=theta_s,
            phi_s=phi_s,
            window_gain=direct_gain,
            wall_azimuth_deg=surface_azimuth
        )
        vertical_direct_with_shading_list.append(vertical_direct_with_shading)

        # Diffuse from L (no shading)
        vertical_diffuse_from_L = compute_vertical_diffuse(L, theta_map, phi_map, wall_azimuth_deg=surface_azimuth)
        vertical_diffuse_L_list.append(vertical_diffuse_from_L)

        # Diffuse with shading (mask)
        vertical_diffuse_shaded = compute_vertical_diffuse_with_shading(
            L, theta_map, phi_map, V_sky_mask, wall_azimuth_deg=surface_azimuth
        )
        vertical_diffuse_with_shading_list.append(vertical_diffuse_shaded)

        # Horizontal diffuse from L (used for reflect computations)
        horizontal_diffuse_from_L = compute_horizontal_diffuse(L, theta_map, phi_map)

        # Ground reflection from diffuse (no shading)
        E_down = horizontal_diffuse_from_L
        vertical_reflect_from_L = (albedo / np.pi) * E_down * SUMW
        vertical_reflect_L_list.append(vertical_reflect_from_L)

        # Ground reflection from direct (no shading)
        E_dir_ground = max(dni.iloc[0] * cos_zenith, 0.0)
        vertical_reflect_direct = (albedo / np.pi) * E_dir_ground * SUMW if E_dir_ground > 0 else 0.0
        vertical_reflect_direct_list.append(vertical_reflect_direct)

        # Ground reflection from diffuse (shaded)
        vertical_reflect_shaded = (albedo / np.pi) * E_down * SUMW_SHADED
        vertical_reflect_L_shaded_list.append(vertical_reflect_shaded)

        # Ground reflection from direct (shaded)
        reflect_direct_shaded = (albedo / np.pi) * E_dir_ground * SUMW_SHADED if E_dir_ground > 0 else 0.0
        vertical_reflect_direct_shaded_list.append(reflect_direct_shaded)

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    print("Saving hourly diffuse overlay images...")

    for time_label, L_hour, theta_s_deg, phi_s_deg in hourly_overlay_records:
        save_L_overlay_orthographic(
            L=L_hour,
            theta_deg=theta_deg_sky,
            phi_deg=phi_deg,
            diffuse_mask_px=V_sky_for_diffuse_px,
            basename=paths.basename,
            time_label=time_label,
            folder=overlay_folder,
            theta_s_deg=theta_s_deg,
            phi_s_deg=phi_s_deg,
            vmax_global=global_overlay_vmax,
        )

    save_diffuse_colorbar_horizontal(
        folder=overlay_folder,
        basename=paths.basename,
        vmax_global=global_overlay_vmax,
        width_px=V_sky_for_diffuse_px.shape[1],
        height_px=140,
    )

    data['Reference Vertical Irradiance'] = poa_global_list
    data['Direct Irradiance'] = poa_direct_list
    data['Sky Diffuse Irradiance'] = poa_sky_diffuse_list
    data['Ground Reflected Irradiance'] = poa_ground_diffuse_list
    data['Horizontal Direct (DNI*cosZ)'] = horizontal_direct_list
    data['Horizontal Diffuse (DHI)'] = horizontal_diffuse_list

    # Remove timezone for Excel
    if "Timestamp" in data.columns and hasattr(data["Timestamp"].dt, "tz_localize"):
        data["Timestamp"] = data["Timestamp"].dt.tz_localize(None)

    data['Vertical Direct with Shading'] = vertical_direct_with_shading_list
    data['Vertical Diffuse from L'] = vertical_diffuse_L_list
    data['Vertical Diffuse from L (shaded)'] = vertical_diffuse_with_shading_list
    data['Vertical Reflect from L'] = vertical_reflect_L_list
    data['Vertical Reflect from Direct'] = vertical_reflect_direct_list
    data['Vertical Reflect Total (L + Direct)'] = (
            data['Vertical Reflect from L'] + data['Vertical Reflect from Direct']
    )
    data['Vertical Reflect from L (shaded)'] = vertical_reflect_L_shaded_list
    data['Vertical Reflect from Direct (shaded)'] = vertical_reflect_direct_shaded_list
    data['Vertical Reflect Total (L + Direct) Shaded'] = (
            data['Vertical Reflect from L (shaded)'] + data['Vertical Reflect from Direct (shaded)']
    )

    data['Vertical Total Irradiance (shaded)'] = (
            data['Vertical Direct with Shading'] +
            data['Vertical Diffuse from L (shaded)'] +
            data['Vertical Reflect Total (L + Direct) Shaded']
    )

    computed_cols = [
        'Reference Vertical Irradiance',
        'Direct Irradiance',
        'Sky Diffuse Irradiance',
        'Ground Reflected Irradiance',
        'Horizontal Direct (DNI*cosZ)',
        'Horizontal Diffuse (DHI)',
        'Vertical Direct with Shading',
        'Vertical Diffuse from L',
        'Vertical Diffuse from L (shaded)',
        'Vertical Reflect from L',
        'Vertical Reflect from Direct',
        'Vertical Reflect Total (L + Direct)',
        'Vertical Reflect from L (shaded)',
        'Vertical Reflect from Direct (shaded)',
        'Vertical Reflect Total (L + Direct) Shaded',
        'Vertical Total Irradiance (shaded)',
    ]

    front_cols = [c for c in data.columns if c not in computed_cols]
    data_out = data[front_cols + computed_cols]

    data_out.to_excel(paths.out_xlsx_path, index=False)

    print(f"Saved to: {paths.step6_dir}")
    print("[Step 6/6] Completed")


def run():
    run_step6_radiation()


if __name__ == "__main__":
    run()

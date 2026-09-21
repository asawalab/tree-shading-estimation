# ====================================================================
# Region Selector - Interactive Binary Mask Selection Tool
# ====================================================================
#
# Purpose
# -------
# This tool provides an interactive interface for selecting the unshaded sky/ground area on orthographic fisheye images.
#
# Contents
# --------
# 1. Load the fisheye image (RGBA) and use the alpha channel to define the valid circular region.
# 2. Display a preview image for fast interaction while preserving the original full-resolution mask for output.
# 3. Support click-based region selection using color similarity.
# 4. Support click-based region erasing using color similarity.
# 5. Support polygon-based local correction:
#       - Polygon        : create a polygon
#       - Polygon Undo   : remove the last polygon point
#       - Polygon Clear  : clear all polygon points
#       - Polygon Select : add the polygon interior to the selected mask
#       - Polygon Erase  : erase the polygon interior from the selected mask
# 6. Support fast whole-area operations:
#       - Select All : select the whole valid circle
#       - Clear All  : clear all selected regions
# 7. Return:
#       - mask_select : selected binary region
#       - mask_valid  : valid circular region
#
# Inputs
# ------
# - img_path : path to an RGBA fisheye image
# - preview_max_size : maximum display size for fast preview interaction
#
# Outputs
# -------
# - mask_select (bool) : selected region mask
# - mask_valid  (bool) : valid circular fisheye region
#
# Notes
# -----
# - The displayed preview may be downsampled for speed, but the returned
#   masks always keep the original full resolution.
# - Red overlay = selected unshaded area.
# - This tool is generic and can be used for both sky and ground fisheye images.
#
# ====================================================================

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.path import Path


def select_region_mask(
    img_path: str,
    preview_max_size: int = 1400,
    instruction: str = "Select a region",
):
    """
      - Select        : click to add a color-similar region
      - Select All    : select the whole valid fisheye circle
      - Erase         : click to remove a color-similar region
      - Clear All     : clear all selected regions
      - Polygon       : draw polygon points
      - Polygon Undo  : remove the last polygon point
      - Polygon Clear : clear all current polygon points
      - Polygon Select: add the polygon interior to the selected mask
      - Polygon Erase : erase the polygon interior from the selected mask
      - Confirm       : finish and return masks

      Returns:
        mask_select (bool), mask_valid (bool)


    """
    # --------------------------------------------------------
    # 1) Load original full-resolution image
    # --------------------------------------------------------
    img_rgba = Image.open(img_path).convert("RGBA")
    img_rgba_np = np.array(img_rgba)
    background_full = img_rgba_np[..., :3].astype(np.uint8)

    H_full, W_full = background_full.shape[:2]

    # Valid fisheye circle from alpha channel
    alpha = img_rgba_np[..., 3]
    mask_valid_full = alpha > 0

    # Full-resolution image for color-distance calculation
    img_full_i32 = background_full.astype(np.int32)

    # Use only valid pixels for faster color-distance computation
    valid_flat_idx = np.flatnonzero(mask_valid_full.ravel())
    valid_pixels = img_full_i32.reshape(-1, 3)[valid_flat_idx]

    # --------------------------------------------------------
    # 2) Build preview image (display only)
    # --------------------------------------------------------
    scale = min(1.0, preview_max_size / max(H_full, W_full))
    W_disp = max(1, int(round(W_full * scale)))
    H_disp = max(1, int(round(H_full * scale)))

    if scale < 1.0:
        bg_disp_pil = Image.fromarray(background_full).resize(
            (W_disp, H_disp), Image.Resampling.BILINEAR
        )
        valid_disp_pil = Image.fromarray(mask_valid_full.astype(np.uint8) * 255).resize(
            (W_disp, H_disp), Image.Resampling.NEAREST
        )
        background_disp = np.array(bg_disp_pil, dtype=np.uint8)
        mask_valid_disp = np.array(valid_disp_pil, dtype=np.uint8) > 127
    else:
        background_disp = background_full.copy()
        mask_valid_disp = mask_valid_full.copy()

    # --------------------------------------------------------
    # 3) State
    # --------------------------------------------------------
    tolerance = 30
    mask_select_full = np.zeros((H_full, W_full), dtype=bool)

    polygon_points_disp = []
    polygon_complete = False
    current_mode = "select"   # select / erase / polygon
    selection_confirmed = False

    # Precompute display-grid points for polygon operations
    j_grid_disp, i_grid_disp = np.meshgrid(np.arange(W_disp), np.arange(H_disp))
    all_points_disp = np.stack([j_grid_disp.ravel(), i_grid_disp.ravel()], axis=-1)

    # --------------------------------------------------------
    # 4) UI setup
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.28, right=0.96, bottom=0.22)

    im_artist = ax.imshow(background_disp)
    ax.axis("off")

    poly_line, = ax.plot([], [], color="cyan", linewidth=2.0)
    poly_start = ax.scatter([], [], color="yellow", s=60, marker="o")

    button_defs = [
        ("Select", "select", [0.03, 0.88, 0.22, 0.045]),
        ("Select All", "select_all", [0.03, 0.82, 0.22, 0.045]),
        ("Erase", "erase", [0.03, 0.76, 0.22, 0.045]),
        ("Clear All", "clear_all", [0.03, 0.70, 0.22, 0.045]),

        ("Polygon", "polygon", [0.03, 0.63, 0.22, 0.045]),
        ("Polygon Undo", "undo", [0.03, 0.57, 0.22, 0.045]),
        ("Polygon Clear", "clear_polygon", [0.03, 0.51, 0.22, 0.045]),
        ("Polygon Select", "select_area", [0.03, 0.45, 0.22, 0.045]),
        ("Polygon Erase", "erase_area", [0.03, 0.39, 0.22, 0.045]),

        ("Confirm", "save", [0.03, 0.32, 0.22, 0.05]),
    ]

    button_objects = []

    ax_tol = plt.axes([0.30, 0.10, 0.60, 0.03])
    slider_tol = Slider(ax_tol, "Tolerance", 1, 120, valinit=tolerance, valstep=1)

    # --------------------------------------------------------
    # 5) Helper functions
    # --------------------------------------------------------
    def full_mask_to_display(mask_full: np.ndarray) -> np.ndarray:
        """
        Convert full-resolution selection mask to display resolution.
        """
        if scale >= 1.0:
            return mask_full.copy()

        pil = Image.fromarray(mask_full.astype(np.uint8) * 255).resize(
            (W_disp, H_disp), Image.Resampling.NEAREST
        )
        return np.array(pil, dtype=np.uint8) > 127

    def blend_preview() -> np.ndarray:
        """
        Build preview image with red overlay.
        """
        mask_disp = full_mask_to_display(mask_select_full)

        overlay = np.zeros_like(background_disp, dtype=np.uint8)
        overlay[mask_disp] = [255, 0, 0]

        blended = (0.65 * background_disp + 0.35 * overlay).astype(np.uint8)
        return blended

    def update_display():
        """
        Fast update without clearing axes.
        """
        im_artist.set_data(blend_preview())

        if len(polygon_points_disp) >= 1:
            xs, ys = zip(*polygon_points_disp)
            poly_line.set_data(xs, ys)
            poly_start.set_offsets([[xs[0], ys[0]]])
        else:
            poly_line.set_data([], [])
            poly_start.set_offsets(np.empty((0, 2)))

        # Only show current mode to avoid confusion between sky/ground/building selection.
        ax.set_title(f"{instruction}\nMode: {current_mode}", fontsize=12)

        fig.canvas.draw_idle()

    def set_mode(mode: str):
        nonlocal current_mode
        current_mode = mode
        update_display()

    def select_all(event):
        nonlocal mask_select_full
        mask_select_full[:] = False
        mask_select_full[mask_valid_full] = True
        update_display()

    def clear_all(event):
        nonlocal mask_select_full
        mask_select_full[:] = False
        update_display()

    def undo_last_point(event):
        nonlocal polygon_complete
        if polygon_points_disp:
            polygon_points_disp.pop()
            polygon_complete = False
            update_display()
        else:
            print("No polygon point to undo.")

    def clear_polygon_points(event):
        """
        Clear all current polygon points.
        This only clears the polygon drawing, not the selected mask.
        """
        nonlocal polygon_complete

        polygon_points_disp.clear()
        polygon_complete = False

        update_display()

    def apply_polygon_area(event, action: str):
        """
        Apply the current polygon area to the selected mask.

        action:
        - "select": add polygon interior to selected mask
        - "erase" : remove polygon interior from selected mask

        Note:
        This function does NOT clear polygon points after applying.
        Use Polygon Undo to remove points one by one.
        Use Polygon Clear to remove all polygon points at once.
        """
        nonlocal mask_select_full, polygon_complete

        if not (polygon_complete and len(polygon_points_disp) >= 3):
            print("Polygon is not complete. Use Polygon mode to close it first.")
            return

        # Polygon on display coordinates
        path_disp = Path(polygon_points_disp)
        inside_disp = path_disp.contains_points(all_points_disp).reshape(H_disp, W_disp)

        # Upscale to full resolution
        if scale < 1.0:
            inside_full = np.array(
                Image.fromarray(inside_disp.astype(np.uint8) * 255).resize(
                    (W_full, H_full), Image.Resampling.NEAREST
                ),
                dtype=np.uint8
            ) > 127
        else:
            inside_full = inside_disp

        inside_full &= mask_valid_full

        if action == "select":
            mask_select_full |= inside_full
        elif action == "erase":
            mask_select_full &= ~inside_full
        else:
            raise ValueError(f"Unknown polygon action: {action}")

        update_display()

    def save_and_exit(event):
        nonlocal selection_confirmed
        selection_confirmed = True
        plt.close(fig)

    def update_slider(val):
        nonlocal tolerance
        tolerance = int(slider_tol.val)
        update_display()

    slider_tol.on_changed(update_slider)

    # --------------------------------------------------------
    # 6) Button callbacks
    # --------------------------------------------------------
    for name, mode, pos in button_defs:
        ax_btn = plt.axes(pos)
        btn = Button(ax_btn, name)
        button_objects.append(btn)

        if mode == "undo":
            btn.on_clicked(undo_last_point)
        elif mode == "clear_polygon":
            btn.on_clicked(clear_polygon_points)
        elif mode == "select_area":
            btn.on_clicked(lambda e: apply_polygon_area(e, action="select"))
        elif mode == "erase_area":
            btn.on_clicked(lambda e: apply_polygon_area(e, action="erase"))
        elif mode == "save":
            btn.on_clicked(save_and_exit)
        elif mode == "select_all":
            btn.on_clicked(select_all)
        elif mode == "clear_all":
            btn.on_clicked(clear_all)
        else:
            btn.on_clicked(lambda e, m=mode: set_mode(m))

    # --------------------------------------------------------
    # 7) Mouse click interaction
    # --------------------------------------------------------
    def onclick(event):
        nonlocal polygon_complete, mask_select_full

        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        j_disp = int(event.xdata)
        i_disp = int(event.ydata)
        if not (0 <= j_disp < W_disp and 0 <= i_disp < H_disp):
            return

        # Map display click to full-resolution coordinates
        j0 = int(round(j_disp / scale)) if scale < 1.0 else j_disp
        i0 = int(round(i_disp / scale)) if scale < 1.0 else i_disp

        j0 = max(0, min(W_full - 1, j0))
        i0 = max(0, min(H_full - 1, i0))

        # Polygon mode
        if current_mode == "polygon":
            if polygon_complete:
                print("Polygon already closed. Use Polygon Clear or Polygon Undo to modify it.")
                return

            if polygon_points_disp:
                dx = j_disp - polygon_points_disp[0][0]
                dy = i_disp - polygon_points_disp[0][1]
                if dx * dx + dy * dy < 100:   # close if near start point
                    polygon_complete = True
                    polygon_points_disp.append(polygon_points_disp[0])
                    update_display()
                    return

            polygon_points_disp.append((j_disp, i_disp))
            update_display()
            return

        # Select / Erase mode
        if not mask_valid_full[i0, j0]:
            return

        color_click = img_full_i32[i0, j0, :]
        diff = valid_pixels - color_click
        dist2 = np.sum(diff * diff, axis=1)
        tol2 = int(tolerance) * int(tolerance)

        region_valid = dist2 < tol2

        mask_select_flat = mask_select_full.ravel()
        target_idx = valid_flat_idx[region_valid]

        if current_mode == "select":
            mask_select_flat[target_idx] = True
        elif current_mode == "erase":
            mask_select_flat[target_idx] = False

        update_display()

    fig.canvas.mpl_connect("button_press_event", onclick)

    # Initial display
    update_display()
    plt.show()

    if not selection_confirmed:
        raise RuntimeError("Region selection was not confirmed.")

    return mask_select_full.astype(bool), mask_valid_full.astype(bool)

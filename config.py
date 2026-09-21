# ==========================================================================================
# config.py
# ==========================================================================================
# This file stores the essential user inputs required by the current Step1 ~ Step6 workflow.
# Please edit this file before running the workflow.
# ==========================================================================================


# ============================================================
# 1) Basic case information
# ============================================================
# Path to the original equirectangular panorama image.
# Replace this placeholder with a path on your computer before running.
image_path = r"C:\tree-shading-estimation\example\case2_panorama.jpg"
# Site information of the measurement location.
latitude = 35.5138
longitude = 139.4856
altitude = 64
timezone = "Asia/Tokyo"
# Capture date and time of the panorama image.
date_str = "2024-08-12"
time_str = "15:00:00"


# ============================================================
# 2) Weather data
# ============================================================
# Path to the weather data file (.xlsx).
# Replace this placeholder with a path on your computer before running.
weather_data_path = r"C:\tree-shading-estimation\example\case2_weather.xlsx"

# Column names in the weather workbook.
# GHI values must be in W/m2.
weather_date_col = "Date"
weather_time_col = "Time"
weather_ghi_col = "Weather station"


# ============================================================
# 3) Building / window geometry
# ============================================================
# Window geometry (meters).
window_base_height = 0.575
window_height = 0.685
window_width = 0.653
# Window center height (meters). This is automatically derived from the window base height and window height.
window_center_height = window_base_height + window_height / 2
# Window / target surface orientation. For example, south-facing vertical window: tilt=90, azimuth=180
surface_tilt_deg = 90.0
surface_azimuth_deg = 180.0


# ============================================================
# 4) Camera position
# ============================================================
# Camera height from the ground/platform (meters).
camera_height_m = 0.795
# Camera distance outside the window plane, along the outward normal (meters).
camera_outward_m = 0.07
# Optional left-right offset of the camera relative to the window center (meters). Positive = east side, negative = west side.
camera_dx_m = 0


# ============================================================
# 5) Tree position relative to the window (NOT the camera)
# ============================================================
# Horizontal straight-line distance from the WINDOW CENTER to the trunk / dominant tree shading layer (m).
# This is a direct center-to-tree distance in plan view, NOT the perpendicular distance to the window plane.
tree_range_m = 0.6
# Azimuth from the WINDOW CENTER toward the tree (deg). For example, 0=N, 90=E, 180=S, 270=W
tree_bearing_az_deg = 180


# ============================================================
# 6) Surface properties
# ============================================================
# Ground albedo.
albedo = 0.17


# ============================================================
# 7) Case type and diffuse-mask logic
# ============================================================
# These switches control how the diffuse sky mask is generated.
# Whether the weather station data already includes the effect of fixed local shading (for example: nearby buildings, mountains, etc.).
station_has_fixed_shading = True
# Whether the current case is the control case (no tree).
is_control_case = False
# This *.npy path is required ONLY when:
#   1) is_control_case = False
#   2) station_has_fixed_shading = True
# In that case, it should point to the raw sky mask (*.npy) of the corresponding control case.
# If the current case is the control case itself, or if the station does not have fixed shading, this value is not used.
control_case_sky_mask_path = r"C:\tree-shading-estimation\example\case2_control_sky_mask.npy"

# Tree Shading Estimation

Python code for estimating solar irradiance on a window under tree shading using an equirectangular panoramic image, global horizontal irradiance (GHI), and basic site and geometry information.

Associated article:

> Dong, F., Asawa, T., and Kawai, H. (2026). Image-based method for estimating vertical solar irradiance on building windows under tree shading. *Building and Environment*, 115262. https://doi.org/10.1016/j.buildenv.2026.115262

## Installation

Python 3.9 or later is required.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The workflow uses interactive Matplotlib windows and should be run in a desktop environment.

## Inputs and configuration

The workflow requires three primary input categories:

1. an equirectangular panoramic image;
2. weather data containing date, time, and GHI; and
3. case-specific site and geometry information defined in `config.py`.

Update the file paths in `config.py` before running. The existing paths illustrate the required path format.

The weather workbook should contain data for the same single day specified by `date_str` in `config.py`.

## Example

The `example` folder contains the following files for Case 2 described in the associated article:

- `case2_panorama.jpg`: the equirectangular panorama;
- `case2_weather.xlsx`: the weather data; the `Weather station` column contains GHI in W/m²;
- `case2_control_sky_mask.npy`: the raw sky mask of the corresponding control case, used by the default diffuse-mask settings.

The default site, date, geometry, surface, and case settings in `config.py` also correspond to Case 2.

## Scope and validation

The target surface tilt and azimuth can be configured in `config.py`. The experimental validation reported in the associated article was conducted for vertical windows. Results for other tilt angles have not been validated against measurements.

## Run

```bash
python run.py
```

The pipeline performs:

1. panorama calibration;
2. orthographic sky and ground projection;
3. interactive obstruction-mask generation;
4. sun-path calculation;
5. whole-window direct solar access calculation; and
6. direct, sky-diffuse, ground-reflected, and total irradiance calculation.

Steps 1 and 3 open interactive windows for selecting the sun and visible sky and ground regions.

Outputs are saved next to the source panorama in numbered folders.

The final result is saved in `*_radiation_output.xlsx` under the column `Vertical Total Irradiance (shaded)`. The column name follows the vertical-window case evaluated in the associated article.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

This repository is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). Commercial use is not permitted. See `LICENSE` for details.

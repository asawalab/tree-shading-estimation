# run.py
import time

import step01_calibrate_panorama
import step02_generate_orthographic_fisheye
import step03_binarize_fisheye_images
import step04_sunpath
import step05_window_direct_gain
import step06_radiation_calculation


def main():
    total_start = time.time()

    steps = [
        ("Step 1", step01_calibrate_panorama.run),
        ("Step 2", step02_generate_orthographic_fisheye.run),
        ("Step 3", step03_binarize_fisheye_images.run),
        ("Step 4", step04_sunpath.run),
        ("Step 5", step05_window_direct_gain.run),
        ("Step 6", step06_radiation_calculation.run),
    ]

    for step_name, func in steps:
        try:
            func()
        except Exception as e:
            print(f"\nPipeline stopped during {step_name}.")
            print(f"Error: {e}")
            raise

    total_dt = time.time() - total_start
    print(f"\nPipeline completed successfully in {total_dt:.2f} s.")


if __name__ == "__main__":
    main()

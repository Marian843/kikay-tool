import cv2
import numpy as np
from pathlib import Path
from shutil import copyfile

# Check if image has enough resolution
def is_high_resolution(img, min_width=128, min_height=128):
    h, w = img.shape[:2]
    return w >= min_width and h >= min_height

# Check if image is not blurry
def is_not_blurry(img, threshold=60):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var > threshold

# Reject grayscale or nearly grayscale images
def is_black_and_white(img, std_threshold=10, diff_threshold=5):
    b, g, r = cv2.split(img)
    max_diff = np.max([np.abs(b - g), np.abs(b - r), np.abs(g - r)])
    avg_std = np.mean([np.std(b), np.std(g), np.std(r)])
    return max_diff < diff_threshold and avg_std < std_threshold

# Filter all images in input folder and save passing ones to output folder
def filter_images(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    for img_file in input_path.iterdir():
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        output_file = output_path / img_file.name
        if output_file.exists():
            continue  # Skip already processed images

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        total += 1

        try:
            if (
                is_high_resolution(img) and
                is_not_blurry(img) and
                not is_black_and_white(img)
            ):
                copyfile(str(img_file), str(output_file))
                kept += 1
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

    print(f"\nFiltered: {kept} new of {total} checked")
    print(f"Total saved now in '{output_path}': {len(list(output_path.iterdir()))}")

# Run the filter
if __name__ == "__main__":
    filter_images("data/UTKFace_unfiltered", "data/UTKFace_filtered")

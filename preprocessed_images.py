import cv2
import numpy as np
from pathlib import Path
from shutil import copyfile

def preprocess_image(img, apply_denoise=False):
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Equalize histogram on L* channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)


    # Merge channels back
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Optional denoising
    if apply_denoise:
        img_eq = cv2.fastNlMeansDenoisingColored(img_eq, None, 10, 10, 7, 21)

    return img_eq

def preprocess_folder(input_folder, output_folder, apply_denoise=False):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    processed = 0

    for img_file in input_path.iterdir():
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        output_file = output_path / img_file.name
        if output_file.exists():
            continue  # Skip if already processed

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        total += 1
        try:
            img_proc = preprocess_image(img, apply_denoise=apply_denoise)
            cv2.imwrite(str(output_file), img_proc)
            processed += 1
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

    print(f"\nPreprocessed: {processed} of {total} images saved to '{output_path}'")

if __name__ == "__main__":
    preprocess_folder(
        input_folder="data/UTKFace_filtered",
        output_folder="data/UTKFace_preprocessed",
        apply_denoise=False  # Set to True if needed later
    )

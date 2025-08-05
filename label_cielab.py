import cv2
import numpy as np
from pathlib import Path
import csv
from collections import Counter

# ---- Labeling thresholds (after fixing OpenCV ranges) ----
def classify_skin_tone(l_value):
    if l_value >= 70:
        return "light"
    elif 40 <= l_value < 70:
        return "medium"
    else:
        return "dark"

def classify_undertone(a_value, b_value):
    # Warm: higher red (a*) and yellow (b*)
    if a_value >= 15 and b_value >= 15:
        return "warm"
    # Cool: lower a* and b* (less red/yellow), more blue/green
    elif a_value <= 10 and b_value <= 13:
        return "cool"
    # Neutral: middle range
    elif 10 < a_value < 15 and 13 < b_value < 20:
        return "neutral"
    else:
        return "unclassified"


# ---- Get average CIELAB values from center face region ----
def get_lab_avg(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]

    # Use central 40% area of the image
    h_margin = int(h * 0.3)
    w_margin = int(w * 0.3)
    region = lab[h_margin:h - h_margin, w_margin:w - w_margin]

    # Normalize L*, shift a* and b* from OpenCV ranges
    l_mean = np.mean(region[:, :, 0]) * (100 / 255.0)  # 0–100
    a_mean = np.mean(region[:, :, 1]) - 128            # -128 to +127
    b_mean = np.mean(region[:, :, 2]) - 128

    return l_mean, a_mean, b_mean

# ---- Main labeling function ----
def label_images(image_folder, output_csv):
    image_path = Path(image_folder)
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg")) + list(image_path.glob("*.png"))

    tone_counter = Counter()
    undertone_counter = Counter()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "l*", "a*", "b*", "skin_tone", "undertone"])

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            try:
                l, a, b = get_lab_avg(img)
                tone = classify_skin_tone(l)
                undertone = classify_undertone(a, b)

                tone_counter[tone] += 1
                undertone_counter[undertone] += 1

                writer.writerow([img_file.name, round(l, 2), round(a, 2), round(b, 2), tone, undertone])
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")

    print(f"Labeling complete! Results saved to: {output_csv}")
    print("Class Distribution:")
    print("- Skin Tones:", dict(tone_counter))
    print("- Undertones:", dict(undertone_counter))

# ---- Run script ----
if __name__ == "__main__":
    label_images("data/UTKFace_preprocessed", "labels.csv")

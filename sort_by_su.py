import csv
from pathlib import Path
from shutil import copyfile
from collections import defaultdict

# --- Paths ---
csv_file = "labels.csv"
cache_file = "last_labels.csv"
source_dir = Path("data/UTKFace_preprocessed")
output_dir = Path("data/UTKFace_sorted")

# --- Valid class labels ---
skin_tone_classes = {"light", "medium", "dark", "unclassified"}
undertone_classes = {"cool", "neutral", "warm", "unclassified"}

# --- Load previous labels from cache ---
def load_cache():
    cache = {}
    if Path(cache_file).exists():
        with open(cache_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache[row["filename"]] = (row["skin_tone"], row["undertone"])
    return cache

# --- Save current labels to cache ---
def save_cache(current_labels):
    with open(cache_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "skin_tone", "undertone"])
        for filename, (tone, undertone) in current_labels.items():
            writer.writerow([filename, tone, undertone])

# --- Main delta-only processing function ---
def sort_images_delta_only():
    current_labels = {}
    updated = 0
    skipped = 0
    missing = 0
    deleted_old = 0
    new_copies = 0
    copy_errors = 0
    delete_errors = 0

    cache = load_cache()

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            tone = row["skin_tone"]
            undertone = row["undertone"]
            current_labels[filename] = (tone, undertone)

            # Skip if label unchanged
            if filename in cache and cache[filename] == (tone, undertone):
                skipped += 1
                continue

            src = source_dir / filename
            if not src.exists():
                print(f"[Missing] {filename}")
                missing += 1
                continue

            # --- Delete from old folders if changed ---
            if filename in cache:
                old_tone, old_undertone = cache[filename]
                old_paths = [
                    output_dir / "skin_tone" / old_tone / filename,
                    output_dir / "undertone" / old_undertone / filename
                ]
                for path in old_paths:
                    try:
                        if path.exists():
                            path.unlink()
                            deleted_old += 1
                    except Exception as e:
                        print(f"[Delete error] {path.name}: {e}")
                        delete_errors += 1

            # --- Copy to updated skin tone folder ---
            if tone in skin_tone_classes:
                try:
                    tone_folder = output_dir / "skin_tone" / tone
                    tone_folder.mkdir(parents=True, exist_ok=True)
                    copyfile(src, tone_folder / filename)
                    new_copies += 1
                except Exception as e:
                    print(f"[Copy error - skin tone] {filename}: {e}")
                    copy_errors += 1

            # --- Copy to updated undertone folder ---
            if undertone in undertone_classes:
                try:
                    undertone_folder = output_dir / "undertone" / undertone
                    undertone_folder.mkdir(parents=True, exist_ok=True)
                    copyfile(src, undertone_folder / filename)
                    new_copies += 1
                except Exception as e:
                    print(f"[Copy error - undertone] {filename}: {e}")
                    copy_errors += 1

            updated += 1

    # --- Save updated labels ---
    save_cache(current_labels)

    # --- Summary ---
    print("Sort by Skin Tone & Undertone: Delta-Only Mode")
    print("--------------------------------------------------")
    print(f"Skipped unchanged:           {skipped}")
    print(f"Updated (label changed):     {updated}")
    print(f"New files copied:            {new_copies}")
    print(f"Old files deleted:           {deleted_old}")
    print(f"Copy errors:                 {copy_errors}")
    print(f"Delete errors:               {delete_errors}")
    print(f"Missing/unreadable files:    {missing}")
    print("--------------------------------------------------")
    print(f"Total files processed:       {updated + skipped}")

# --- Run script ---
if __name__ == "__main__":
    sort_images_delta_only()

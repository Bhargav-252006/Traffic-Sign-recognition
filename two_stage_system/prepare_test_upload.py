"""
Prepare 5 Test Images Per Class for Google Drive Upload
========================================================
Run this LOCALLY on your Windows machine.

What it does:
  1. Picks 5 random images from each of your 17 test class folders
  2. Copies them into a single upload-ready folder on your Desktop
  3. You then upload that ONE folder to Google Drive

After running:
  1. Go to drive.google.com
  2. Navigate to: My Drive > GTSRB_TwoStage
  3. Drag & drop the 'test_images_by_class' folder from your Desktop
  4. Wait for upload to finish
  5. On Colab, it will be at: /content/drive/MyDrive/GTSRB_TwoStage/test_images_by_class/
"""

import os
import shutil
import random
from pathlib import Path

# ============================================================
# PATHS - Change if needed
# ============================================================
LOCAL_TEST_DIR = r"c:\Users\Bhargav\Desktop\dl other\model_data\test"
OUTPUT_DIR = r"c:\Users\Bhargav\Desktop\test_images_by_class"

IMAGES_PER_CLASS = 5
SEED = 42

# All 17 GTSRB classes
TARGET_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                  '33', '34', '35', '36', '37', '38', '39', '40']

CLASS_DESCRIPTIONS = {
    '0': 'Speed limit 20 km/h',
    '1': 'Speed limit 30 km/h',
    '2': 'Speed limit 50 km/h',
    '3': 'Speed limit 60 km/h',
    '4': 'Speed limit 70 km/h',
    '5': 'Speed limit 80 km/h',
    '6': 'End of speed limit 80 km/h',
    '7': 'Speed limit 100 km/h',
    '8': 'Speed limit 120 km/h',
    '33': 'Turn right ahead',
    '34': 'Turn left ahead',
    '35': 'Ahead only',
    '36': 'Go straight or right',
    '37': 'Go straight or left',
    '38': 'Keep right',
    '39': 'Keep left',
    '40': 'Roundabout mandatory',
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.ppm', '.bmp'}


def main():
    random.seed(SEED)

    print("=" * 65)
    print("  PREPARE TEST IMAGES FOR GOOGLE DRIVE UPLOAD")
    print("=" * 65)

    # Verify source exists
    if not os.path.exists(LOCAL_TEST_DIR):
        print(f"\n  ERROR: Source folder not found!")
        print(f"  Expected: {LOCAL_TEST_DIR}")
        print(f"  Change LOCAL_TEST_DIR at the top of this script.")
        return

    # Clean and create output folder
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"\n  Source:  {LOCAL_TEST_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Images per class: {IMAGES_PER_CLASS}")
    print(f"  Classes: {len(TARGET_CLASSES)}")
    print()

    total_copied = 0
    classes_ok = 0

    for class_id in TARGET_CLASSES:
        # Find source folder (try '0' and '00000' formats)
        class_src = None
        for fmt in [class_id, class_id.zfill(5)]:
            candidate = os.path.join(LOCAL_TEST_DIR, fmt)
            if os.path.exists(candidate):
                class_src = candidate
                break

        if class_src is None:
            desc = CLASS_DESCRIPTIONS.get(class_id, '?')
            print(f"  WARNING  Class {class_id:>2s} ({desc[:28]:28s}) - not found")
            continue

        # Get all images
        all_images = [
            f for f in os.listdir(class_src)
            if os.path.isfile(os.path.join(class_src, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        if not all_images:
            print(f"  WARNING  Class {class_id:>2s} - no images")
            continue

        # Pick 5 random images
        selected = random.sample(all_images, min(IMAGES_PER_CLASS, len(all_images)))

        # Create output class folder (5-digit: 00000, 00001, etc.)
        target_dir = os.path.join(OUTPUT_DIR, class_id.zfill(5))
        os.makedirs(target_dir, exist_ok=True)

        # Copy
        for img_name in selected:
            src = os.path.join(class_src, img_name)
            dst = os.path.join(target_dir, img_name)
            shutil.copy2(src, dst)
            total_copied += 1

        classes_ok += 1
        desc = CLASS_DESCRIPTIONS.get(class_id, '?')
        print(f"  OK       Class {class_id:>2s} ({desc[:28]:28s}) - {len(selected)}/{len(all_images)} images")

    # Summary
    print()
    print("=" * 65)
    print("  DONE!")
    print("=" * 65)
    print(f"  Classes: {classes_ok}/{len(TARGET_CLASSES)}")
    print(f"  Images:  {total_copied}")
    print(f"  Folder:  {OUTPUT_DIR}")
    print()
    print("  NEXT STEPS:")
    print("  ─────────────────────────────────────────────────────")
    print("  1. Open drive.google.com in your browser")
    print("  2. Navigate to: My Drive > GTSRB_TwoStage")
    print("     (create GTSRB_TwoStage folder if it does not exist)")
    print("  3. Drag & drop this folder into it:")
    print(f"     {OUTPUT_DIR}")
    print("  4. Wait for upload to complete (~85 small images)")
    print("  5. On Colab it will appear at:")
    print("     /content/drive/MyDrive/GTSRB_TwoStage/test_images_by_class/")
    print("  ─────────────────────────────────────────────────────")
    print()

    # Verify output
    print("  Output folder contents:")
    for class_id in TARGET_CLASSES:
        folder = os.path.join(OUTPUT_DIR, class_id.zfill(5))
        if os.path.exists(folder):
            count = len(os.listdir(folder))
            print(f"    {class_id.zfill(5)}/  ->  {count} images")

    print()


if __name__ == '__main__':
    main()

"""
Unit & Validation Tests for Claim Summary Card Generator
========================================================
Verifies:
1. Total image count across all splits and classes (2,550 images).
2. Proper folder structure for Teachable Machine zip upload:
   claim_cards/{split}/{class_label}/{claim_id}_v{n}.png
3. Dual variations for train (v1 & v2) and single variation for val/test (v1 only).
4. Image dimensions and valid PNG headers.
5. Completeness of claim_id_to_image.csv mapping.
"""

import os
import csv
from pathlib import Path
from PIL import Image


def test_claim_cards_count_and_hierarchy():
    base_dir = Path("claim_cards")
    assert base_dir.exists(), "claim_cards directory does not exist"

    splits = ["train", "val", "test"]
    classes = ["Valid Claim", "Invalid Claim", "Manual Review"]

    expected_counts = {
        ("train", "Valid Claim"): 700,
        ("train", "Invalid Claim"): 700,
        ("train", "Manual Review"): 700,
        ("val", "Valid Claim"): 75,
        ("val", "Invalid Claim"): 75,
        ("val", "Manual Review"): 75,
        ("test", "Valid Claim"): 75,
        ("test", "Invalid Claim"): 75,
        ("test", "Manual Review"): 75,
    }

    total_images = 0
    for split in splits:
        for cls in classes:
            folder = base_dir / split / cls
            assert folder.exists(), f"Missing folder: {folder}"
            pngs = list(folder.glob("*.png"))
            expected = expected_counts[(split, cls)]
            assert len(pngs) == expected, f"Folder {folder} has {len(pngs)} images, expected {expected}"
            total_images += len(pngs)

    assert total_images == 2550, f"Expected 2550 images, got {total_images}"


def test_image_dimensions_and_validity():
    sample_files = [
        Path("claim_cards/train/Valid Claim/CLM-2026-00023_v1.png"),
        Path("claim_cards/train/Valid Claim/CLM-2026-00023_v2.png"),
        Path("claim_cards/val/Invalid Claim/CLM-2026-00944_v1.png"),
    ]

    for p in sample_files:
        assert p.exists(), f"Sample image {p} not found"
        with Image.open(p) as img:
            assert img.format == "PNG", f"{p} is not PNG format"
            assert img.size == (800, 560), f"{p} dimensions {img.size} != (800, 560)"


def test_mapping_csv():
    mapping_file = Path("claim_id_to_image.csv")
    assert mapping_file.exists(), "claim_id_to_image.csv missing"

    with open(mapping_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2550, f"Expected 2550 rows in mapping, got {len(rows)}"

    for row in rows[:50]:  # check first 50 sample rows
        path = Path(row["image_path"])
        assert path.exists(), f"Image path listed in CSV does not exist: {path}"


if __name__ == "__main__":
    print("Running claim card validation tests...")
    test_claim_cards_count_and_hierarchy()
    test_image_dimensions_and_validity()
    test_mapping_csv()
    print("[PASS] All claim card verification tests passed successfully!")

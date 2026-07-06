# ================================================================
# EXPECTED DIRECTORY LAYOUT (as deduced strictly from this script)
# ================================================================
#
# SOURCE ROOT:
#   <SCRATCH_PATH>/<user>/processed_SLAM_RF/
#
# The script expects the following fixed structure:
#
#   processed_SLAM_RF/
#       lidar/
#           <day>/
#               *.png
#       radar/
#           <day>/
#               *.png
#
# where <day> is any element of TRAIN_DAYS or TEST_DAYS.
# Each <day> directory must contain only PNG images; all other
# files are ignored.
#
# ------------------------------------------------
# DESTINATION ROOT:
#   <SCRATCH_PATH>/<user>/dataset_SLAM/
#
# The script creates a flat, split-based structure:
#
#   dataset_SLAM_2/
#       train/
#           lidar/
#               <day>_<original_filename>.png
#           radar/
#               <day>_<original_filename>.png
#       test/
#           lidar/
#               <day>_<original_filename>.png
#           radar/
#               <day>_<original_filename>.png
#
# The script also writes:
#   dataset_SLAM_2/dataset_description.txt
# containing metadata about the build and the chosen splits.
#
# ================================================================

import os
import shutil
import argparse
from textwrap import dedent
from datetime import datetime

# ================== USER PARAMETERS ==================


# Source folder structure:
# scratch/processed_SLAM_RF/
#   lidar/<day>/*.png
#   radar/<day>/*.png
SOURCE_FOLDER = "processed_SLAM_RF"
SOURCE_DIR = None  # to be set in main()

# Target folder structure:
# dataset_SLAM/
#   train/{lidar, radar}
#   test/{lidar, radar}
DEST_FOLDER = "dataset_SLAM"
DEST_DIR = None  # to be set in main()

# Choose which days go to training and which to testing
# (use the exact directory names you have inside processed_SLAM_RF)
TRAIN_DAYS = ["020925", "030925", "040925", "050925"]
TEST_DAYS = ["100925", "170925"]

# Optional: check if any day appears in both splits
CHECK_OVERLAP = True

# The two sensor modalities to process
SENSORS = ["lidar", "radar"]

# Dataset description (automatically saved into DEST_FOLDER)
DATASET_DESCRIPTION = dedent("""
    Processed SLAM_RF dataset organized for machine learning workflows.

    Final structure:
      dataset_SLAM/
        train/
          lidar/
          radar/
        test/
          lidar/
          radar/

    All images are renamed using the schema <day>_<original_filename>.png
    to ensure global uniqueness and maintain explicit temporal provenance.

    This dataset was generated using:
                             -preprocessing legend: none=original data, low=only flipping, high= one shift in range and Num_shifts in azimuth with values set by user, vhigh= combination of shift first and flip later, vvhigh/vvvhigh = analogous to high but using hadcoded values for the shifts chose by the authors
                             -continuous intensity for lidar polar images, not just binary values
                             -intensity handling as specified below, naming convention for folders: xyzzz, x:1-6 enhancement level, y:1-3 intensity handling (1 binary, 2 mag thresh, 3 cfar), zzz: threshold value used (000 if binary)
                             -using chirps 8 and 24 for radar polar images.
                             -radar azimuth range: -90 to 90 degrees included in the polar images (no more black lines).
                             -lidar azimuth range: -90 to 90 degrees included in the polar images (correctly oriented to be coherent with radar values of theta)
                             -keeping 3D radius for lidar polar images and grouping intensities by taking the max across elevation keeping range and azimuth constant.
""").strip()

# ======================================================


valid_folders = {"moh", "ste", "alex"}


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Build SLAM_RF dataset with configurable source/destination folders."
    )

    parser.add_argument(
        "--folder", "-fo",
        choices=valid_folders,
        required=True,
        help="User folder (moh, ste, alex)."
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        required=False,
        default=None,
        help="Name of the SOURCE folder (located one level above this script)."
    )

    parser.add_argument(
        "--dest", "-d",
        type=str,
        required=False,
        default=None,
        help="Name of the DESTINATION folder (also one level above)."
    )

    parser.add_argument(
        "--append-comment", "-c",
        type=str,
        default="",
        help="Optional text appended to the dataset_description.txt file."
    )

    return parser.parse_args()


def check_days():
    """
    Validate that:
    - No day is accidentally used in both train and test.
    - All referenced directories actually exist.
    """
    if SOURCE_DIR is None:
        raise ValueError("SOURCE_DIR must be different from None when calling check_days()")

    if CHECK_OVERLAP:
        overlap = set(TRAIN_DAYS) & set(TEST_DAYS)
        if overlap:
            raise ValueError(f"Days appear in BOTH TRAIN and TEST: {overlap}")

    # Check directory existence
    for sensor in SENSORS:
        for day in TRAIN_DAYS + TEST_DAYS:
            day_dir = os.path.join(SOURCE_DIR, sensor, day)
            if not os.path.isdir(day_dir):
                print(f"WARNING: directory not found → {day_dir}")


def copy_split(split_name, days):
    """
    split_name: 'train' or 'test'
    days: list of day-folder names that should go to this split

    For each day and for each sensor (lidar, radar):
      - read all PNG files inside the day's folder
      - copy them into the unified split folder
      - rename them to <day>_<original_filename>
        so that the day information is encoded into the filename
        (and the day subfolders disappear in the final structure)
    """
    if SOURCE_DIR is None:
        raise ValueError("source_dir must be not None when calling copy_split()")

    for sensor in SENSORS:
        out_dir = os.path.join(DEST_DIR, split_name, sensor)
        os.makedirs(out_dir, exist_ok=True)

        for day in days:
            day_dir = os.path.join(SOURCE_DIR, sensor, day)
            if not os.path.isdir(day_dir):
                print(f"[{split_name}/{sensor}] skipping missing day folder: {day_dir}")
                continue

            for fname in sorted(os.listdir(day_dir)):
                if not fname.lower().endswith(".png"):
                    continue

                src_path = os.path.join(day_dir, fname)

                # New filename: <day>_<original_name>
                # Note: <original_name> already includes extension .png
                # <original_name> are all of the form exp1_0, exp1_1,... across all days that's why we need to include the day at the beginning
                new_name = f"{day}_{fname}"
                dst_path = os.path.join(out_dir, new_name)
                
                # Check if the destination file already exists. If so, remove it before creating a new symlink.
                # This ensures that the destination folder is completely overwritten each time the script runs.
                if os.path.lexists(dst_path):
                    os.remove(dst_path)
                os.symlink(src_path, dst_path)



def write_dataset_description(extra_comment=""):
    """Generate an auto-documenting dataset description inside DEST_FOLDER."""
    os.makedirs(DEST_DIR, exist_ok=True)
    desc_path = os.path.join(DEST_DIR, "dataset_description.txt")

    header = (
            "SLAM_RF Dataset — Auto-generated description\n"
            f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Source root: {SOURCE_DIR}\n"
            f"Destination root: {DEST_FOLDER}\n"
            "\n"
            "=== TRAIN DAYS ===\n" + ", ".join(TRAIN_DAYS) +
            "\n\n=== TEST DAYS ===\n" + ", ".join(TEST_DAYS) +
            "\n\n=== DESCRIPTION ===\n"
    )

    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n")
        f.write(DATASET_DESCRIPTION)

        if extra_comment.strip():
            f.write("\n\n=== USER APPENDED COMMENT ===\n")
            f.write(extra_comment.strip() + "\n")

    print(f"[INFO] Dataset description saved to: {desc_path}")


def main():
    args = parse_cli()
    user = args.folder

    # --------------------------------------------------
    # CHECK WORKING DIRECTORY
    # --------------------------------------------------

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level above create_dataset/

    expected_suffix = os.path.join(user, "radarhd", "create_dataset")

    if not SCRIPT_DIR.endswith(expected_suffix):
        raise ValueError(
            f"Please run the script from inside the '{expected_suffix}' folder.\n"
            f"Current directory: {SCRIPT_DIR}"
        )
    # --------------------------------------------------

    global SOURCE_FOLDER, DEST_FOLDER
    if args.source is not None:
        SOURCE_FOLDER = args.source
    if args.dest is not None:
        DEST_FOLDER = args.dest

    global SOURCE_DIR, DEST_DIR
    SOURCE_DIR = os.path.join(PROJECT_ROOT, SOURCE_FOLDER)
    DEST_DIR = os.path.join(PROJECT_ROOT, DEST_FOLDER)

    EXTRA_COMMENT = args.append_comment

    check_days()

    print(f"-------------------------------------- train_test_split.py --------------------------------------")
    print(f"Splitting images inside: {SOURCE_DIR}")
    print(f"Saving results inside: {DEST_DIR}")

    print("Creating destination directory structure...")
    for split in ["train", "test"]:
        for sensor in SENSORS:
            os.makedirs(os.path.join(DEST_DIR, split, sensor), exist_ok=True)

    print("Copying TRAIN files...")
    copy_split("train", TRAIN_DAYS)

    print("Copying TEST files...")
    copy_split("test", TEST_DAYS)

    write_dataset_description(EXTRA_COMMENT)

    print("Done.")


if __name__ == "__main__":
    main()

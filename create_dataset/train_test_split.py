# ================================================================
# EXPECTED DIRECTORY LAYOUT (as deduced strictly from this script)
# ================================================================
#
# SOURCE ROOT:
#   <SCRATCH_PATH>/<user>/processed_SLAM_112005/
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
# where <day> is any directory name under processed_SLAM_RF/{lidar,radar}.
# Each <day> directory must contain only PNG images; all other
# files are ignored.
#
# ------------------------------------------------
# DESTINATION ROOT:
#   <SCRATCH_PATH>/<user>/dataset_SLAM_121005/
#
# The script creates a flat, split-based structure:
#
#   dataset_SLAM_2/
#       train/
#           lidar/
#           radar/
#       test/
#           lidar/
#           radar/
#
# Files are renamed as:
#   <day>_<original_filename>.png
#
# The split is done by EXPERIMENT (expK), mixed across days:
#   - randomly select TRAIN_RATIO of experiments for train
#   - remaining experiments go to test
#
# The script also writes:
#   dataset_SLAM_2/dataset_description.txt
# containing metadata about the build and the chosen splits.
#
# ================================================================

import os
import re
import shutil
import argparse
import random
from textwrap import dedent
from datetime import datetime

# ================== USER PARAMETERS ==================

SOURCE_ROOT = None  # to be set in main()
SOURCE_FOLDER = "processed_SLAM_112005"
 


DEST_ROOT = None #to be set in main
DEST_FOLDER = "dataset_SLAM_112005"
CHECK_OVERLAP = False  # no longer split by day lists
SENSORS = ["lidar", "radar"]

TRAIN_RATIO = 0.8
SEED = 0

# Dataset description (automatically saved into DEST_ROOT)
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

_EXP_RE = re.compile(r"^exp(?P<exp>\d+)_\d+\.png$", re.IGNORECASE)

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
        "--train_ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Fraction of experiments to use for training (default: 0.8)."
    )
    
    parser.add_argument(
        "--append-comment", "-c",
        type=str,
        default="",
        help="Optional text appended to the dataset_description.txt file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for experiment split."
    )
    return parser.parse_args()

def list_days(sensor: str):
    """Return list of day directories under SOURCE_ROOT/<sensor>/."""
    root = os.path.join(SOURCE_ROOT, sensor)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Missing sensor folder: {root}")
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def extract_exp_from_fname(fname: str):
    """
    Expect filenames like exp1_0.png, exp12_114.png, ...
    Return 'exp1', 'exp12', etc. or None if not matching.
    """
    m = _EXP_RE.match(fname)
    if m is None:
        return None
    return f"exp{int(m.group('exp'))}"

def collect_experiments():
    """
    Scan SOURCE_ROOT over all days and return:
      - all_days: sorted list of day strings
      - all_exps: sorted list of experiment ids (e.g. ['exp1','exp2',...]) found in data
    Uses radar folder as reference (lidar should match).
    """
    all_days = list_days("radar")

    exp_set = set()
    for day in all_days:
        day_dir = os.path.join(SOURCE_ROOT, "radar", day)
        for fname in os.listdir(day_dir):
            if not fname.lower().endswith(".png"):
                continue
            exp = extract_exp_from_fname(fname)
            if exp is not None:
                exp_set.add(exp)

    all_exps = sorted(exp_set, key=lambda s: int(s.replace("exp", "")))
    if not all_exps:
        raise RuntimeError("No experiments found. Expected filenames like 'expK_idx.png'.")
    return all_days, all_exps

def split_experiments(all_exps, train_ratio, seed):
    rnd = random.Random(seed)
    exps = list(all_exps)
    rnd.shuffle(exps)

    n_train = int(round(train_ratio * len(exps)))
    n_train = max(1, min(n_train, len(exps) - 1))  # ensure non-empty train/test

    train_exps = set(exps[:n_train])
    test_exps  = set(exps[n_train:])
    return train_exps, test_exps

def copy_split_by_exps(split_name, days, exps_for_split):
    """
    Referencing with symlink all PNGs whose experiment id (expK) is in exps_for_split,
    across all days, for both sensors, into DEST_ROOT/<split_name>/<sensor>/.
    """
    for sensor in SENSORS:
        out_dir = os.path.join(DEST_ROOT, split_name, sensor)
        os.makedirs(out_dir, exist_ok=True)

        for day in days:
            day_dir = os.path.join(SOURCE_ROOT, sensor, day)
            if not os.path.isdir(day_dir):
                print(f"[{split_name}/{sensor}] skipping missing day folder: {day_dir}")
                continue

            for fname in sorted(os.listdir(day_dir)):
                if not fname.lower().endswith(".png"):
                    continue

                exp = extract_exp_from_fname(fname)
                if exp is None or exp not in exps_for_split:
                    continue

                src_path = os.path.join(day_dir, fname)
                new_name = f"{day}_{fname}"
                dst_path = os.path.join(out_dir, new_name)
                if os.path.lexists(dst_path):
                    os.remove(dst_path)
                os.symlink(src_path, dst_path)

def write_dataset_description(train_exps, test_exps, days, train_ratio, seed, extra_comment=""):
    os.makedirs(DEST_ROOT, exist_ok=True)
    desc_path = os.path.join(DEST_ROOT, "dataset_description.txt")

    header = (
        "SLAM-RF Dataset — Auto-generated description\n"
        f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Source root: {SOURCE_ROOT}\n"
        f"Destination root: {DEST_ROOT}\n"
        f"Split unit: experiment id (expK), mixed across days\n"
        f"Train ratio: {train_ratio}\n"
        f"Random seed: {seed}\n"
        "\n"
        "=== DAYS USED ===\n" + ", ".join(days) +
        "\n\n=== TRAIN EXPERIMENTS ===\n" + ", ".join(sorted(train_exps, key=lambda s: int(s.replace("exp","")))) +
        "\n\n=== TEST EXPERIMENTS ===\n" + ", ".join(sorted(test_exps, key=lambda s: int(s.replace("exp","")))) +
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
    print(f"-------------------------------------- train_test_split.py --------------------------------------")
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

    global SOURCE_ROOT, DEST_ROOT
    SOURCE_ROOT = os.path.join(PROJECT_ROOT, SOURCE_FOLDER)
    DEST_ROOT = os.path.join(PROJECT_ROOT, DEST_FOLDER)

    EXTRA_COMMENT = args.append_comment

    days, all_exps = collect_experiments()
    train_exps, test_exps = split_experiments(all_exps, args.train_ratio, args.seed)

    print(f"Found {len(days)} days: {days}")
    print(f"Found {len(all_exps)} experiments: {all_exps}")
    print(f"Train experiments ({len(train_exps)}): {sorted(train_exps, key=lambda s: int(s.replace('exp','')))}")
    print(f"Test experiments  ({len(test_exps)}): {sorted(test_exps,  key=lambda s: int(s.replace('exp','')))}")

    print(f"Splitting images inside: {SOURCE_ROOT}")
    print(f"Saving results inside: {DEST_ROOT}")

    print("Creating destination directory structure...")
    for split in ["train", "test"]:
        for sensor in SENSORS:
            os.makedirs(os.path.join(DEST_ROOT, split, sensor), exist_ok=True)

    print("Copying TRAIN files...")
    copy_split_by_exps("train", days, train_exps)

    print("Copying TEST files...")
    copy_split_by_exps("test", days, test_exps)

    write_dataset_description(train_exps, test_exps, days, args.train_ratio, args.seed)

    print("Done.")

if __name__ == "__main__":
    main()

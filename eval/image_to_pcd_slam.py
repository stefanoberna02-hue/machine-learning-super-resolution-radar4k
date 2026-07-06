# Converts cartesian images (from pol_to_cart_slam.py) into point clouds (PCD)
# Naming format:
#   <epoch>_<day>_<exp>_<frame>_<pred/label>.png
#
# Assumptions (ENFORCED):
#   image height  = RBINS  (range)
#   image width   = ABINS  (azimuth / lateral)

import glob
import os
import numpy as np
from PIL import Image

# ----------------------------------------------------------
# PARAMETERS (PHYSICAL, NON-NEGOTIABLE)
# ----------------------------------------------------------

params = {
    "model_name": "13",
    "expt": 1,
    "dt": "20251211-134019",
    "epoch_num": 40,
}

RMAX = 10.8
RBINS = 256        # range bins  -> image height
ABINS = 512        # azimuth bins -> image width
MIN_THRESHOLD = 1

# Physical grids (defined ONCE)
range_grid   = np.linspace(0.0, RMAX, RBINS)          # forward (meters)
lateral_grid = np.linspace(-RMAX, RMAX, ABINS)        # left-right (meters)

# ----------------------------------------------------------
# PCD WRITER (ASCII, STANDARD)
# ----------------------------------------------------------

def write_pcd_ascii(path, points):
    """
    points: (N, 3) float32
    """
    N = points.shape[0]

    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {N}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {N}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

# ----------------------------------------------------------
# CARTESIAN IMAGE → PCD
# ----------------------------------------------------------

def convert2pcd(img_files, DIR):
    print("Processing:", DIR)

    PCD_DIR = os.path.join(DIR, "pcd")
    os.makedirs(PCD_DIR, exist_ok=True)

    for f in img_files:
        filename = os.path.basename(f).replace(".png", "")

        # --------------------------------------------------
        # Load image (grayscale)
        # --------------------------------------------------
        img = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
        H, W = img.shape

        # --------------------------------------------------
        # HARD PHYSICAL CONSISTENCY CHECK
        # --------------------------------------------------
        assert H == RBINS, f"{f}: height {H} != RBINS {RBINS}"
        assert W == ABINS, f"{f}: width  {W} != ABINS {ABINS}"

        # --------------------------------------------------
        # Threshold (TOZERO semantics)
        # --------------------------------------------------
        mask = img >= MIN_THRESHOLD
        coords = np.argwhere(mask)

        if coords.size == 0:
            # valid empty cloud
            pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        else:
            # coords: (N, 2) -> [row, col]
            r_idx = coords[:, 0]   # range index   ∈ [0, RBINS-1]
            a_idx = coords[:, 1]   # azimuth index ∈ [0, ABINS-1]

            r = range_grid[r_idx]
            y = lateral_grid[a_idx]

            pts = np.column_stack((
                r,
                y,
                np.zeros_like(r)
            )).astype(np.float32)

        write_pcd_ascii(
            os.path.join(PCD_DIR, filename + ".pcd"),
            pts
        )

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

name_str = f"{params['model_name']}_{params['expt']}_{params['dt']}"
root_path = f"./processed_imgs_{name_str}_test_imgs/"

print("Converted images root:", root_path)

trajs = sorted(
    [d for d in glob.glob(root_path + "*") if os.path.isdir(d)]
)

print("Found trajectories:", trajs)

epoch = f"{params['epoch_num']:03d}"

# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------

for traj in trajs:
    EPOCH_DIR = os.path.join(traj, epoch)

    PRED_DIR  = os.path.join(EPOCH_DIR, "pred")
    LABEL_DIR = os.path.join(EPOCH_DIR, "label")

    def extract_frame_idx(path):
        return int(os.path.basename(path).split("_")[3])

    pred_files = sorted(
        glob.glob(PRED_DIR + "/*.png"),
        key=extract_frame_idx
    )
    label_files = sorted(
        glob.glob(LABEL_DIR + "/*.png"),
        key=extract_frame_idx
    )

    convert2pcd(pred_files, PRED_DIR)
    convert2pcd(label_files, LABEL_DIR)

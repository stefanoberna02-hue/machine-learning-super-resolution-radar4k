#Script used to check the object type in python od various file formats (.npz, .npy, .ply, .pcd) and the header of contents inside the files
#usage example:
# python3 inspect_file.py "/mnt/course-cs-433-group03/scratch/SLAM-RF/lidar/020925/020925_exp0/PointCloudFrame7.ply" -n 20

import numpy as np
import sys
import os
import argparse

EXPECTED_FIELDS = ["x", "y", "z", "timestamp", "ring", "intensity"]

def check_positive_y(pcd_path, show_examples=10, verbose=True):
    """
    Inspect a PCD ASCII file and check whether any point has y > 0.
    Validates file structure, extracts point-cloud data, and prints detailed statistics.
    """
    with open(pcd_path, "r") as f:
        lines = f.readlines()

    fields = None
    data_start = None
    declared_points = None

    # ---------------------------------------------------------
    # Parse the PCD header
    # ---------------------------------------------------------
    for i, line in enumerate(lines):
        ls = line.strip().lower()

        if ls.startswith("fields"):
            fields = line.split()[1:]

        elif ls.startswith("points"):
            declared_points = int(line.split()[1])

        elif ls.startswith("data"):
            if "ascii" not in ls:
                raise ValueError("Unsupported PCD format: only ASCII is handled.")
            data_start = i + 1
            break

    if fields is None:
        raise ValueError("Missing FIELDS entry in PCD header.")

    if fields != EXPECTED_FIELDS:
        raise ValueError(f"Unexpected field layout: {fields} (expected {EXPECTED_FIELDS})")

    if data_start is None:
        raise RuntimeError("DATA ascii section not found.")

    # ---------------------------------------------------------
    # Load the numerical data
    # ---------------------------------------------------------
    data = np.loadtxt(lines[data_start:])
    if data.ndim != 2 or data.shape[1] != len(fields):
        raise ValueError(
            f"Invalid number of columns: {data.shape[1]}, expected {len(fields)}."
        )

    if declared_points is not None and data.shape[0] != declared_points:
        print(f"Warning: header declares {declared_points} points, "
              f"but file contains {data.shape[0]}.")

    # Extract columns
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    intensity = data[:, 5]

    theta = np.rad2deg(np.arctan2(y, x))

    positive_mask = y >= 0
    num_positive = np.sum(positive_mask)

    # ---------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------
    print("\n=== PCD FILE ANALYSIS ===")
    print(f"File path: {pcd_path}")
    print(f"Total points: {data.shape[0]}")
    print(f"Points with y >= 0: {num_positive} "
          f"({100 * num_positive / data.shape[0]:.2f}%)")
    print(f"first negative y index: {np.argmax(~positive_mask)}")

    print("\n--- Global Bounding Box ---")
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]\t avg: {np.mean(x):.3f}\t mae: {np.mean(np.abs(x - np.mean(x))):.3f}")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]\t avg: {np.mean(y):.3f}\t mae: {np.mean(np.abs(y - np.mean(y))):.3f}")
    print(f"z range: [{z.min():.3f}, {z.max():.3f}]\t avg: {np.mean(z):.3f}\t mae: {np.mean(np.abs(z - np.mean(z))):.3f}")
    print(f"theta range: [{theta.min():.3f}, {theta.max():.3f}]")
    print(f"avg theta : {np.mean(theta):.3f}")
    print(f"std theta : {np.std(theta):.3f}")
   
    print(f"intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")


    if verbose:
        if num_positive > 0:
            y_neg = y[~positive_mask]

            print("\n--- Statistics only among y < 0 ---")
            print(f"min:  {y_neg.min():.3f}")
            print(f"max:  {y_neg.max():.3f}")
            print(f"mean: {y_neg.mean():.3f}")
            print(f"std:  {y_neg.std():.3f}")
            print(f"avg theta: {np.mean(theta[~positive_mask]):.3f}")

            print(f"\n--- First {show_examples} examples with y < 0 ---")
            neg_indices = np.where(~positive_mask)[0]
            for idx in neg_indices[:show_examples]:
                print(f"#{idx:>6}:  x={x[idx]:.3f},  y={y[idx]:.3f},  "
                    f"z={z[idx]:.3f}, theta={theta[idx]:.3f},  intensity={intensity[idx]:.3f}")







def inspecting(path ,N=10 ):
    try:
        from plyfile import PlyData
    except ImportError:
        PlyData = None

    try:
        import open3d as o3d
    except ImportError:
        o3d = None

    ext = os.path.splitext(path)[1].lower()

    print(f"\nInspecting file: {path}")
    print(f"Detected extension: {ext}")
    print(f"Showing {N} sample values.\n")


    # ---- 1) NPZ files ----
    if ext == ".npz":
        data = np.load(path)
        print("NPZ archive keys:", data.files)
        for key in data.files:
            arr = data[key]
            print(f"\nArray '{key}': shape={arr.shape}, dtype={arr.dtype}")
            print("Sample:", arr.flatten()[:N])


    # ---- 2) NPY files ----
    elif ext == ".npy":
        arr = np.load(path)
        print(f"NPY array: shape={arr.shape}, dtype={arr.dtype}")
        print("Sample:", arr.flatten()[:N])


    # ---- 3) PLY files ----
    elif ext == ".ply":
        if o3d is not None:
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points)
            print("PLY loaded via Open3D.")
            print("Points shape:", pts.shape)
            print("Sample points:\n", pts[:N])
        else:
            print("Open3D not installed. Cannot inspect .ply files.")


    # ---- 4) PCD files ----
    elif ext == ".pcd":
        if o3d is not None:
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points)
            print("PCD points shape:", pts.shape)
            print("Sample points:\n", pts[:N])
        else:
            print("Open3D not installed. Cannot inspect .pcd files.")

    else:
        print("Unsupported file type.")




# Usage:
# python3 inspect_file.py -p "/mnt/course-cs-433-group03/scratch/SLAM-RF/lidar/050925/050925_exp1/PointCloudFrame3.pcd"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic file inspector")
    parser.add_argument("-p","--path", type=str, required=True, help="Path to file (.npz, .npy, .ply, .pcd)")
    parser.add_argument("-n", "--num", type=int, default=20, 
                        help="Number of sample points/values to display (default=10)")
    args = parser.parse_args()
    path = args.path
    N = args.num

    inspecting(path, N=N)
    #check_positive_y(path)

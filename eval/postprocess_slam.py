#!/usr/bin/env python3
"""
Post-processing SLAM pipeline (monolithic).

Pipeline stages:
1) Polar → Cartesian image conversion
2) Cartesian images → Point Clouds (.pcd)
3) Point Cloud comparison (Chamfer + Modified Hausdorff)

Assumed project layout:
- This script is located in: <project_root>/eval/
- Project root is one level above this file
- Logs are stored in: <project_root>/logs/<run_name>/
- Test images are in: <project_root>/logs/<run_name>/test_imgs/

Outputs:
- Cartesian images and PCDs:
  <project_root>/processed_imgs_<run_name>_test_imgs/
- CDF plot:
  <project_root>/eval/pointcloud_error_cdf_<run_name>_<epoch>.png
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
except Exception as e:
    raise ImportError("SciPy is required (cKDTree not available).") from e


# ============================================================
# PATH RESOLUTION (ROBUST, VIA __file__)
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))          # <project_root>/eval
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # <project_root>

# ============================================================
# GLOBAL DEFAULT PARAMETERS
# ============================================================

DEFAULT_PARAMS = {
    "model_name": "13",
    "expt": 1,
    "dt": "debug",
    "epoch_num": None,

    "logs_dir": "logs",
    "out_dir": None,

    "rmax": 10.8,
    "rbins": 256,
    "abins": 512,
    "min_threshold": 1,

    "clean": False,
}

# ============================================================
# CLI
# ============================================================
def parse_cli():
    parser = argparse.ArgumentParser(
        description="Postprocess SLAM results (polar→cart, cart→pcd, pc comparison)."
    )

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--expt", type=int, default=None)
    parser.add_argument("--dt", type=str, default=None)

    parser.add_argument("--epoch-num", type=int, default=None)

    parser.add_argument("--logs-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--clean", action="store_true")

    return parser.parse_args()

def build_params_from_args(args):
    """
    Merge CLI arguments into global defaults.
    CLI values override defaults only if not None.
    """
    params = DEFAULT_PARAMS.copy()

    if args.model_name is not None:
        params["model_name"] = args.model_name

    if args.expt is not None:
        params["expt"] = args.expt 
    
    if args.dt is not None:
        params["dt"] = args.dt
    
    if args.epoch_num is not None:
        params["epoch_num"] = args.epoch_num
    

    return params



# ============================================================
# UTILITIES
# ============================================================

def find_last_checkpoint(log_dir: str) -> int:
    """
    Return the maximum epoch index among available .pt_gen checkpoints.
    """
    epochs = []
    if not os.path.isdir(log_dir):
        raise RuntimeError(f"Log directory not found: {log_dir}")

    for f in os.listdir(log_dir):
        if f.endswith(".pt_gen"):
            try:
                epochs.append(int(os.path.splitext(f)[0]))
            except ValueError:
                pass

    if not epochs:
        raise RuntimeError(f"No checkpoints (.pt_gen) found in {log_dir}")

    return max(epochs)


def write_pcd_ascii(path: str, points: np.ndarray) -> None:
    """
    Write a point cloud in ASCII PCD format.
    """
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {points.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {points.shape[0]}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def read_pcd(path: str) -> np.ndarray:
    """
    Read a PCD file (ASCII or binary) and return Nx3 array.
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Malformed PCD (EOF before DATA): {path}")
            header.append(line)
            if line.startswith(b"DATA"):
                dtype = line.split()[1].decode().lower()
                break

        header_txt = b"".join(header).decode(errors="ignore")
        npts = int(
            [l for l in header_txt.splitlines() if l.startswith("POINTS")][0].split()[1]
        )

        if dtype == "ascii":
            pts = np.loadtxt(f, dtype=np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            return pts

        raw = f.read(npts * 12)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)


def _pc_key(filename: str) -> str:
    """
    Extract a common key from a PCD filename by removing the trailing 'pred/label' token.
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return "_".join(parts[:-1])


def pc_distance(A: np.ndarray, B: np.ndarray):
    """
    Compute Chamfer distance and Modified Hausdorff distance between two point clouds.
    """
    treeB = cKDTree(B)
    dA, _ = treeB.query(A, k=1, workers=-1)

    treeA = cKDTree(A)
    dB, _ = treeA.query(B, k=1, workers=-1)

    chamfer = 0.5 * (dA.mean() + dB.mean())
    modhaus = max(np.median(dA), np.median(dB))

    return chamfer, modhaus


# ============================================================
# 1) POLAR → CARTESIAN
# ============================================================

def pol_to_cart_main(img_dir: str, save_path: str,
                     rmax: float, rbins: int, abins: int):
    """
    Convert polar radar images to Cartesian representation.
    """
    os.makedirs(save_path, exist_ok=True)

    agrid = np.linspace(-90, 90, abins)
    rgrid = np.linspace(0, rmax, rbins)

    cosgrid = np.cos(np.deg2rad(agrid))
    singrid = np.sin(np.deg2rad(agrid))

    sine_theta, range_d = np.meshgrid(singrid, rgrid)
    cos_theta = np.sqrt(1 - sine_theta**2)

    x_axis = range_d * cos_theta
    y_axis = range_d * sine_theta

    x_axis_grid = np.linspace(0, rmax, rbins)
    y_axis_grid = np.linspace(-rmax, rmax, abins)

    def convert_pol2cart(a: np.ndarray) -> np.ndarray:
        b = np.zeros((rbins, abins), dtype=a.dtype)
        loc = np.argwhere(a > 0)
        if loc.size == 0:
            return b

        xloc, yloc = loc[:, 0], loc[:, 1]
        x = x_axis[xloc, yloc]
        y = y_axis[xloc, yloc]

        new_xloc = np.searchsorted(x_axis_grid, x)
        new_yloc = np.searchsorted(y_axis_grid, y)

        new_xloc = np.clip(new_xloc, 0, rbins - 1)
        new_yloc = np.clip(new_yloc, 0, abins - 1)

        b[new_xloc, new_yloc] = a[xloc, yloc]
        return b

    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No PNG files found in {img_dir}")

    for file in files:
        name = os.path.basename(file).replace(".png", "")
        # Expected format: <epoch>_<day>_<exp>_<idx>_<kind>.png
        epoch, day, exp, idx, kind = name.split("_")
        traj_folder = f"{day}_{exp}"

        img = np.asarray(Image.open(file))
        cart = convert_pol2cart(img).astype(np.uint8)
        cart = Image.fromarray(cart)

        outdir = os.path.join(save_path, traj_folder, epoch, kind)
        os.makedirs(outdir, exist_ok=True)

        cart.save(os.path.join(outdir, f"{epoch}_{day}_{exp}_{idx}_{kind}.png"))

    print("[OK] Polar → Cartesian conversion completed.")


# ============================================================
# 2) CARTESIAN → PCD
# ============================================================

def image_to_pcd_main(root_path: str, epoch: str,
                      rmax: float, rbins: int, abins: int,
                      min_threshold: int):
    """
    Convert Cartesian images to point clouds.
    """
    range_grid = np.linspace(0.0, rmax, rbins)
    lateral_grid = np.linspace(-rmax, rmax, abins)

    trajs = sorted(d for d in glob.glob(os.path.join(root_path, "*")) if os.path.isdir(d))
    if not trajs:
        raise RuntimeError(f"No trajectory folders found in {root_path}")

    for traj in trajs:
        for kind in ["pred", "label"]:
            img_dir = os.path.join(traj, epoch, kind)
            if not os.path.isdir(img_dir):
                continue

            pcd_dir = os.path.join(img_dir, "pcd")
            os.makedirs(pcd_dir, exist_ok=True)

            files = sorted(
                glob.glob(os.path.join(img_dir, "*.png")),
                key=lambda p: int(os.path.basename(p).split("_")[3])
            )

            for f in files:
                img = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
                mask = img >= min_threshold
                coords = np.argwhere(mask)

                if coords.size == 0:
                    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                else:
                    r = range_grid[coords[:, 0]]
                    y = lateral_grid[coords[:, 1]]
                    pts = np.column_stack((r, y, np.zeros_like(r))).astype(np.float32)

                write_pcd_ascii(
                    os.path.join(pcd_dir, os.path.basename(f).replace(".png", ".pcd")),
                    pts
                )

    print("[OK] Cartesian → PCD conversion completed.")


# ============================================================
# 3) POINT CLOUD COMPARISON
# ============================================================

def pc_compare_main(root: str, epoch: str, plot_path: str):
    """
    Compute Chamfer and Modified Hausdorff distances and plot CDFs.
    """
    chamfer, modhaus = [], []

    trajs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not trajs:
        raise RuntimeError(f"No trajectory folders found in {root}")

    for traj in trajs:
        pdir = os.path.join(root, traj, epoch, "pred", "pcd")
        ldir = os.path.join(root, traj, epoch, "label", "pcd")

        if not (os.path.isdir(pdir) and os.path.isdir(ldir)):
            continue

        pred_files = [f for f in os.listdir(pdir) if f.endswith(".pcd")]
        label_files = [f for f in os.listdir(ldir) if f.endswith(".pcd")]

        pred_map = {_pc_key(f): f for f in pred_files}
        label_map = {_pc_key(f): f for f in label_files}

        for k in sorted(set(pred_map) & set(label_map)):
            pc_p = read_pcd(os.path.join(pdir, pred_map[k]))[:, :2]
            pc_l = read_pcd(os.path.join(ldir, label_map[k]))[:, :2]

            c, h = pc_distance(pc_l, pc_p)
            chamfer.append(c)
            modhaus.append(h)

    chamfer = np.asarray(chamfer)
    modhaus = np.asarray(modhaus)

    if chamfer.size == 0:
        raise RuntimeError("No matching pred/label point clouds found.")

    ch_mean, ch_med = chamfer.mean(), np.median(chamfer)
    mh_mean, mh_med = modhaus.mean(), np.median(modhaus)

    print(f"Chamfer: mean={ch_mean:.6f}, median={ch_med:.6f}")
    print(f"ModHaus: mean={mh_mean:.6f}, median={mh_med:.6f}")

    plt.figure(figsize=(8, 6))
    plt.plot(np.sort(chamfer), np.linspace(0, 1, len(chamfer)), label="Chamfer")
    plt.plot(np.sort(modhaus), np.linspace(0, 1, len(modhaus)), "--", label="Modified Hausdorff")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Point Cloud Error (m)")
    plt.ylabel("CDF")
    plt.title(f"Point Cloud Error CDF — epoch={epoch}")

    stats = (
        f"Chamfer : mean = {ch_mean:.4f} m, median = {ch_med:.4f} m\n"
        f"ModHaus : mean = {mh_mean:.4f} m, median = {mh_med:.4f} m"
    )
    plt.figtext(0.5, -0.15, stats, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Point cloud comparison completed. Plot saved to {plot_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_cli()
    params = build_params_from_args(args)

    run_name = f"{params['model_name']}_{params['expt']}_{params['dt']}"

    logs_root = os.path.join(PROJECT_ROOT, params["logs_dir"])
    log_dir = os.path.join(logs_root, run_name)

    if params["epoch_num"] is None:
        epoch = find_last_checkpoint(log_dir)
    else:
        candidate = os.path.join(log_dir, f"{params['epoch_num']:03d}.pt_gen")
        if os.path.isfile(candidate):
            epoch = params["epoch_num"]
        else:
            print(
                f"[WARN] Requested epoch {params['epoch_num']} not found. "
                "Using last available checkpoint."
            )
            epoch = find_last_checkpoint(log_dir)

    epoch_str = f"{epoch:03d}"

    img_dir = os.path.join(log_dir, "test_imgs")
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Missing test_imgs directory: {img_dir}")

    out_root = params["out_dir"] or os.path.join(
        SCRIPT_DIR, f"processed_imgs_{run_name}_test_imgs"
    )

    if params["clean"] and os.path.isdir(out_root):
        import shutil
        shutil.rmtree(out_root)

    os.makedirs(out_root, exist_ok=True)

    pol_to_cart_main(
        img_dir=img_dir,
        save_path=out_root,
        rmax=params["rmax"],
        rbins=params["rbins"],
        abins=params["abins"],
    )

    image_to_pcd_main(
        root_path=out_root,
        epoch=epoch_str,
        rmax=params["rmax"],
        rbins=params["rbins"],
        abins=params["abins"],
        min_threshold=params["min_threshold"],
    )

    plot_path = os.path.join(
    SCRIPT_DIR, f"pointcloud_error_cdf_{params['dt']}.png"
    )


    pc_compare_main(out_root, epoch_str, plot_path)


if __name__ == "__main__":
    main()


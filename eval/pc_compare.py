import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTS
# ============================================================

RMAX = 10.8

# ============================================================
# PCD READER (ASCII + BINARY)
# ============================================================

def read_pcd(path):
    """
    Minimal PCD reader supporting:
    - DATA ascii
    - DATA binary
    Assumes fields: x y z (float32)
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PCD header")

            header_lines.append(line)
            if line.strip().startswith(b"DATA"):
                data_type = line.strip().split()[1].decode()
                break

        header = b"".join(header_lines).decode("ascii", errors="ignore")

        # Extract number of points
        num_points = None
        for l in header.splitlines():
            if l.startswith("POINTS"):
                num_points = int(l.split()[1])
                break

        if num_points is None:
            raise RuntimeError("POINTS field not found in PCD header")

        # ---------------- ASCII ----------------
        if data_type == "ascii":
            pts = []
            for _ in range(num_points):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line:
                    pts.append([float(v) for v in line.split()])
            return np.asarray(pts, dtype=np.float32)

        # ---------------- BINARY ----------------
        elif data_type == "binary":
            raw = f.read(num_points * 12)  # 3 float32
            pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)
            return pts

        else:
            raise RuntimeError(f"Unsupported PCD DATA type: {data_type}")

# ============================================================
# POINT CLOUD METRICS
# ============================================================

def bin_pc(pc, bin_size):
    """Bins points to a grid, exactly like MATLAB bin_pc()."""
    if bin_size == 0:
        return pc.copy()

    x_grid = np.arange(0, RMAX + bin_size, bin_size)
    y_grid = np.arange(-RMAX, RMAX + bin_size, bin_size)

    new_pc = np.zeros_like(pc)

    for i in range(pc.shape[0]):
        x, y = pc[i]

        x_idx = np.searchsorted(x_grid, x)
        y_idx = np.searchsorted(y_grid, y)

        x_idx = min(x_idx, len(x_grid) - 1)
        y_idx = min(y_idx, len(y_grid) - 1)

        new_pc[i, 0] = x_grid[x_idx]
        new_pc[i, 1] = y_grid[y_idx]

    return new_pc


def pc_distance(pc_A, pc_B, metric, bin_size=0):
    """
    Computes Chamfer / Hausdorff / Modified Hausdorff
    pc_A, pc_B: (N,2) arrays (x,y)
    """
    pc_A = bin_pc(pc_A, bin_size)
    pc_B = bin_pc(pc_B, bin_size)

    if pc_A.size == 0 or pc_B.size == 0:
        return np.nan

    # ---------- Chamfer ----------
    if metric == "chamfer":
        dA = np.mean([
            np.min(np.linalg.norm(pc_B - p, axis=1))
            for p in pc_A
        ])
        dB = np.mean([
            np.min(np.linalg.norm(pc_A - p, axis=1))
            for p in pc_B
        ])
        return 0.5 * (dA + dB)

    # ---------- Hausdorff ----------
    elif metric == "hausdorff":
        dA = np.max([
            np.min(np.linalg.norm(pc_B - p, axis=1))
            for p in pc_A
        ])
        dB = np.max([
            np.min(np.linalg.norm(pc_A - p, axis=1))
            for p in pc_B
        ])
        return max(dA, dB)

    # ---------- Modified Hausdorff ----------
    elif metric == "mod_hausdorff":
        dA = np.median([
            np.min(np.linalg.norm(pc_B - p, axis=1))
            for p in pc_A
        ])
        dB = np.median([
            np.min(np.linalg.norm(pc_A - p, axis=1))
            for p in pc_B
        ])
        return max(dA, dB)

    else:
        raise ValueError(f"Unknown metric: {metric}")

# ============================================================
# OPTIONAL: STANDALONE EVALUATION SCRIPT
# ============================================================

def main():
    root_folder = "./processed_imgs_13_1_20251211-134019_test_imgs"
    epoch = "120"
    bin_size = 0

    print("Evaluating point clouds in:\n", root_folder)

    traj_folders = sorted([
        os.path.join(root_folder, d)
        for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ])

    all_chamfer = []
    all_modhaus = []

    for traj in traj_folders:

        pred_dir = os.path.join(traj, epoch, "pred", "pcd")
        label_dir = os.path.join(traj, epoch, "label", "pcd")

        if not (os.path.isdir(pred_dir) and os.path.isdir(label_dir)):
            continue

        pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".pcd"))
        label_files = sorted(f for f in os.listdir(label_dir) if f.endswith(".pcd"))

        for pf, lf in zip(pred_files, label_files):
            pc_pred = read_pcd(os.path.join(pred_dir, pf))[:, :2]
            pc_lab  = read_pcd(os.path.join(label_dir, lf))[:, :2]

            ch = pc_distance(pc_lab, pc_pred, "chamfer", bin_size)
            mh = pc_distance(pc_lab, pc_pred, "mod_hausdorff", bin_size)

            if not np.isnan(ch):
                all_chamfer.append(ch)
                all_modhaus.append(mh)

    print("Mean Chamfer:", np.mean(all_chamfer))
    print("Mean ModHaus:", np.mean(all_modhaus))


if __name__ == "__main__":
    main()
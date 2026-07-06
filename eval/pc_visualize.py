import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
exp_datetime = "20251215-124042"
epoch = "120"

root_folder = f"./processed_imgs_13_1_{exp_datetime}_test_imgs/"
save_root = f"./visualizations_{exp_datetime}/"
os.makedirs(save_root, exist_ok=True)

RMAX = 10.8

# ----------------------------------------------------------------------
# PCD READER (ASCII + BINARY)
# ----------------------------------------------------------------------
def read_pcd(path):
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

        num_points = None
        for l in header.splitlines():
            if l.startswith("POINTS"):
                num_points = int(l.split()[1])
                break
        if num_points is None:
            raise RuntimeError("POINTS field not found")

        if data_type == "ascii":
            pts = []
            for _ in range(num_points):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line:
                    pts.append([float(v) for v in line.split()])
            return np.asarray(pts, dtype=np.float32)

        elif data_type == "binary":
            raw = f.read(num_points * 12)
            return np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)

        else:
            raise RuntimeError(f"Unsupported DATA type: {data_type}")

# ----------------------------------------------------------------------
def extract_frame_index(filename):
    return int(filename.split("_")[3])

def sorted_pcd_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".pcd")]
    return [os.path.join(folder, f)
            for f in sorted(files, key=extract_frame_index)]

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if not os.path.isdir(root_folder):
    raise FileNotFoundError(root_folder)

traj_list = sorted(
    d for d in os.listdir(root_folder)
    if os.path.isdir(os.path.join(root_folder, d))
)

plt.ioff()

for traj_name in traj_list:
    traj_path = os.path.join(root_folder, traj_name)
    epoch_path = os.path.join(traj_path, epoch)

    pred_folder = os.path.join(epoch_path, "pred", "pcd")
    label_folder = os.path.join(epoch_path, "label", "pcd")

    if not (os.path.isdir(pred_folder) and os.path.isdir(label_folder)):
        continue

    pred_files = sorted_pcd_files(pred_folder)
    label_files = sorted_pcd_files(label_folder)

    traj_save_dir = os.path.join(save_root, traj_name)
    os.makedirs(traj_save_dir, exist_ok=True)

    for pred_path, label_path in zip(pred_files, label_files):
        label = read_pcd(label_path)
        pred  = read_pcd(pred_path)

        if label.size == 0:
            label = np.zeros((1, 3))
        if pred.size == 0:
            pred = np.zeros((1, 3))

        frame_idx = extract_frame_index(os.path.basename(label_path))

        fig = plt.figure(figsize=(10, 10))

        # ---------------- LABEL ----------------
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(label[:, 1], label[:, 0], s=1)
        ax1.set_title(f"{traj_name} — LABEL")
        ax1.grid(True)
        ax1.set_xlim([-RMAX, RMAX])
        ax1.set_ylim([0, RMAX])
        ax1.set_aspect("equal", adjustable="box")

        # ---------------- PRED ----------------
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(pred[:, 1], pred[:, 0], s=1)
        ax2.set_title(f"{traj_name} — RadarHD Prediction")
        ax2.grid(True)
        ax2.set_xlim([-RMAX, RMAX])
        ax2.set_ylim([0, RMAX])
        ax2.set_aspect("equal", adjustable="box")

        save_path = os.path.join(
            traj_save_dir, f"frame_{frame_idx:04d}.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

print("Done.")

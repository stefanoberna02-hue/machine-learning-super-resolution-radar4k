import os
import json
import random
import itertools
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from train_test_utils.model import UNet3
from train_test_utils.dice_score import dice_loss
from train_test_utils.dataloader_slam import SLAMDataset
from eval.pc_compare import pc_distance

# ============================================================
# GLOBAL CONFIG
# ============================================================

DATASET_ROOT = "./dataset_SLAM"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KFOLDS = 5
NUM_EPOCHS = 10
BATCH_SIZE = 6
NUM_WORKERS = 8
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# GRID SEARCH
# ============================================================

HISTORY_GRID = [30, 40, 50]
LOSS_GRID = [(0.9, 0.1), (0.7, 0.3)]   # (BCE, Dice)
LR_GRID = [1e-4, 1e-3]

# ============================================================
# POLAR → CARTESIAN (VALIDATION ONLY)
# ============================================================

RMAX = 10.8
RBINS = 256
ABINS = 512

agrid = np.linspace(-90, 90, ABINS)
rgrid = np.linspace(0, RMAX, RBINS)

cosgrid = np.cos(np.deg2rad(agrid))
singrid = np.sin(np.deg2rad(agrid))

sine_theta, range_d = np.meshgrid(singrid, rgrid)
cos_theta = np.sqrt(1 - sine_theta**2)

x_axis = range_d * cos_theta
y_axis = range_d * sine_theta

x_axis_grid = np.linspace(0, RMAX, RBINS)
y_axis_grid = np.linspace(-RMAX, RMAX, ABINS)

# Precompute polar->cart index maps for faster conversion (validation only)
IX_MAP = np.searchsorted(x_axis_grid, x_axis)
IY_MAP = np.searchsorted(y_axis_grid, y_axis)
IX_MAP = np.clip(IX_MAP, 0, RBINS - 1).astype(np.int32)
IY_MAP = np.clip(IY_MAP, 0, ABINS - 1).astype(np.int32)


def pol_to_cart(polar_img):
    """
    polar_img: (RBINS, ABINS) numpy array
    returns:   (RBINS, ABINS) cartesian occupancy image

    Uses precomputed IX_MAP/IY_MAP for speed.
    """
    cart = np.zeros((RBINS, ABINS), dtype=np.float32)

    coords = np.argwhere(polar_img > 0)
    if coords.size == 0:
        return cart

    r_idx = coords[:, 0]
    a_idx = coords[:, 1]

    ix = IX_MAP[r_idx, a_idx]
    iy = IY_MAP[r_idx, a_idx]

    cart[ix, iy] = polar_img[r_idx, a_idx]
    return cart


def img_to_pc(img, threshold=0.1):
    """Convert a cartesian occupancy image to a point cloud in meters.

    The input 'img' lives on the (x_axis_grid, y_axis_grid) lattice:
      - axis 0 (rows)  -> x in [0, RMAX]
      - axis 1 (cols)  -> y in [-RMAX, RMAX]
    """
    coords = np.argwhere(img >= threshold)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    ix = coords[:, 0]
    iy = coords[:, 1]

    x = x_axis_grid[ix]
    y = y_axis_grid[iy]

    return np.stack([x, y], axis=1).astype(np.float32)


# ============================================================
# DATASET / SPLIT UTILITIES
# ============================================================

def extract_experiment_from_path(path):
    """
    Example:
      100925_exp2_114.png → 100925_exp2
    """
    base = os.path.basename(path)
    return "_".join(base.split("_")[:2])


def kfold_split(items, k):
    items = list(items)
    random.shuffle(items)
    return np.array_split(items, k)

# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(model, dataset, device=None):
    """Run validation computing Chamfer and Modified Hausdorff distances
    after polar→cart postprocessing.

    This version evaluates the model in batches (faster) but keeps the same
    per-sample postprocessing and distance computation.
    """
    model.eval()
    chamfers, modhaus = [], []
    BIN_SIZE = 0  # match pc_compare.py: no binning, distances in meters

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = DEVICE

    # Use a DataLoader for speed (batched inference)
    if isinstance(dataset, DataLoader):
        loader = dataset
        n_samples = len(dataset.dataset) if hasattr(dataset, "dataset") else None
    else:
        n_samples = len(dataset)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    if n_samples is None:
        try:
            n_samples = len(loader.dataset)
        except Exception:
            n_samples = 0

    print(f"    → Validation on {n_samples} samples")

    used = 0
    skipped_empty = 0
    skipped_nan = 0

    with torch.no_grad():
        seen = 0
        for radar, lidar in loader:
            # radar: (B,C,H,W) tensor
            radar = radar.to(device, non_blocking=True)
            pred = model(radar)

            pred_polar_batch = (
                pred.detach().float().cpu().numpy()
            )
            # squeeze channel if present
            if pred_polar_batch.ndim == 4:
                pred_polar_batch = pred_polar_batch[:, 0]

            lidar_np = lidar.detach().float().cpu().numpy()
            if lidar_np.ndim == 4:
                gt_polar_batch = lidar_np[:, 0]
            elif lidar_np.ndim == 3:
                gt_polar_batch = lidar_np
            else:
                raise ValueError(f"Unexpected lidar batch shape: {lidar_np.shape}")

            bsz = pred_polar_batch.shape[0]

            for b in range(bsz):
                pred_polar = pred_polar_batch[b]
                gt_polar = gt_polar_batch[b]

                pred_cart = pol_to_cart(pred_polar)
                gt_cart = pol_to_cart(gt_polar)

                pc_pred = img_to_pc(pred_cart)
                pc_gt = img_to_pc(gt_cart)

                if pc_pred.size == 0 or pc_gt.size == 0:
                    skipped_empty += 1
                    continue

                if np.isnan(pc_pred).any() or np.isnan(pc_gt).any():
                    skipped_nan += 1
                    continue

                try:
                    ch = pc_distance(pc_gt, pc_pred, "chamfer", BIN_SIZE)
                    mh = pc_distance(pc_gt, pc_pred, "mod_hausdorff", BIN_SIZE)
                except Exception:
                    continue

                chamfers.append(ch)
                modhaus.append(mh)
                used += 1

            seen += bsz
            if seen % 200 == 0:
                print(
                    f"      progress: {seen}/{n_samples} | "
                    f"used={used}, skipped_empty={skipped_empty}, skipped_nan={skipped_nan}"
                )

    if used == 0:
        print("    [WARN] No valid samples for distance evaluation.")
        return float("nan"), float("nan")

    print(f"    Done. used={used}, skipped_empty={skipped_empty}, skipped_nan={skipped_nan}")
    return float(np.mean(chamfers)), float(np.mean(modhaus))



def main():

    results = []

    for history, (bce_w, dice_w), lr in itertools.product(
        HISTORY_GRID, LOSS_GRID, LR_GRID
    ):

        print("\n" + "=" * 90)
        print(
            f"GRID CONFIG → history={history}, "
            f"BCE={bce_w}, Dice={dice_w}, lr={lr}"
        )
        print("=" * 90)

        # ---------------- Dataset ----------------
        print("→ Loading dataset")
        full_dataset = SLAMDataset(
            DATASET_ROOT,
            split="train",
            M=history
        )
        print(f"  Total samples: {len(full_dataset)}")

        # map samples → experiment
        idx_to_exp = np.array([
            extract_experiment_from_path(p)
            for p in full_dataset.radar_files[history:]
        ])

        experiments = sorted(set(idx_to_exp))
        folds = kfold_split(experiments, KFOLDS)

        print(f"  Experiments: {len(experiments)}")
        print(f"  Using {KFOLDS}-fold CV")

        fold_scores = []

        # ---------------- FOLDS ----------------
        for k in range(KFOLDS):

            print(f"\n--- Fold {k+1}/{KFOLDS} ---")

            val_exps = set(folds[k])
            train_exps = set(experiments) - val_exps

            train_idx = np.where(np.isin(idx_to_exp, list(train_exps)))[0]
            val_idx   = np.where(np.isin(idx_to_exp, list(val_exps)))[0]

            print(f"    Train samples: {len(train_idx)}")
            print(f"    Val samples:   {len(val_idx)}")

            train_set = Subset(full_dataset, train_idx)
            val_set   = Subset(full_dataset, val_idx)

            train_loader = DataLoader(
                train_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=NUM_WORKERS
            )

            model = UNet3(history + 1, 1).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            bce = torch.nn.BCELoss()

            # ---------------- TRAIN ----------------
            model.train()
            for epoch in range(NUM_EPOCHS):

                epoch_loss = 0.0
                nb = 0

                for radar, lidar in train_loader:
                    radar = radar.to(DEVICE)
                    lidar = lidar.to(DEVICE)

                    optimizer.zero_grad()
                    out = model(radar)
                    loss = (
                        bce_w * bce(out, lidar) +
                        dice_w * dice_loss(out, lidar)
                    )
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    nb += 1

                epoch_loss /= max(1, nb)
                print(
                    f"    Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
                    f"Train loss: {epoch_loss:.4f}"
                )

            # ---------------- VALIDATE ----------------
            print("    → Starting validation")
            ch, mh = evaluate_model(model, val_set)
            fold_scores.append((ch, mh))

            print(
                f"    Fold result → "
                f"Chamfer={ch:.4f}, ModHaus={mh:.4f}"
            )

        mean_ch = float(np.nanmean([x[0] for x in fold_scores]))
        mean_mh = float(np.nanmean([x[1] for x in fold_scores]))

        print("\n>>> MEAN over folds")
        print(f"    Chamfer = {mean_ch:.4f}")
        print(f"    ModHaus = {mean_mh:.4f}")

        results.append({
            "history": history,
            "bce_weight": bce_w,
            "dice_weight": dice_w,
            "lr": lr,
            "mean_chamfer": mean_ch,
            "mean_modhaus": mean_mh
        })

    os.makedirs("./cv_results", exist_ok=True)
    with open("./cv_results/cv_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✔ Cross-validation completed")
    print("✔ Results saved to ./cv_results/cv_summary.json")

# ============================================================

if __name__ == "__main__":
    main()

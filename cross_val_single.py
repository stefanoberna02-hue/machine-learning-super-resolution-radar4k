"""
Cross-validation runner for SLAM-RF training (model selection only).

Key properties:
- Accepts the same CLI arguments as train_slam.py (plus CV-specific flags).
- Training loop matches train_slam.py logic (optimizer, ReduceLROnPlateau, range-weighted BCE + Dice, ...).
- Computes distances using the same pc_distance()/bin_pc() logic as eval/pc_compare.py.
- DOES NOT save images, checkpoints, tensorboard logs, or any other artifacts.
- Writes only a JSON summary (and prints results to stdout).
"""

import os
import re
import json
import argparse
import random
import itertools
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy.spatial import cKDTree


from train_test_utils.dataloader_slam import SLAMDataset

from train_slam import (
    parse_cli as train_parse_cli,
    model_map as model_map,
    build_params_from_args as train_build_params_from_args,
    run_training,
)
from create_dataset.sync_slam_rf import valid_enhancements, augmentation_map


# -------------------------------
# Parameters that characterize this grid search, others will be passed by cli()
# -------------------------------

#all these parameters are used only when not provided in cli() as it is convention in all our code
MODEL_ARCHITECTURE='unet3'
OPTIM = 'adam'
EPOCHS=40
RANGE_EPS = 0.05
RANGE_CLIP_MAX = 3.0
SHUFFLE_DATA=False
EXPT='112005'
SOURCE_FOLDER='dataset_SLAM_'+EXPT


# -------------------------------
# Constants (match your pipeline)
# -------------------------------
K_FOLDS=5
SEED=0

# output folder is set in main with this structure
#OUT_JSON = f"./cv_results/cv_{params['model_name']}.json"


DISTANCE_THRESHOLD=0.1 #threshold to binarize outputs before batch distances
BIN_SIZE=0.0
RMAX = 10.8

#these are the shape of our final images in polar form returned by the model (radar generated and lidar label) 
#they are also used to create a cartesian grid with same dimesnions knowing the kind of conversion we need to apply
RBINS = 256
ABINS = 512  # cartesian grid width used in eval; do not change unless you change eval

# Polar grids (for pol->cart mapping)
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

# Precompute polar->cart index maps once (speed)
_IX_MAP = np.searchsorted(x_axis_grid, x_axis, side="left")
_IY_MAP = np.searchsorted(y_axis_grid, y_axis, side="left")
_IX_MAP = np.clip(_IX_MAP, 0, RBINS - 1).astype(np.int32)
_IY_MAP = np.clip(_IY_MAP, 0, ABINS - 1).astype(np.int32)

def pol_to_cart(polar_img: np.ndarray) -> np.ndarray:
    """
    polar_img: (RBINS, ABINS) float array
    returns:   (RBINS, ABINS) cartesian occupancy image
    """
    cart = np.zeros((RBINS, ABINS), dtype=np.float32)
    coords = np.argwhere(polar_img > 0)
    if coords.size == 0:
        return cart
    r_idx = coords[:, 0]
    a_idx = coords[:, 1]
    ix = _IX_MAP[r_idx, a_idx]
    iy = _IY_MAP[r_idx, a_idx]
    cart[ix, iy] = polar_img[r_idx, a_idx]
    return cart

def img_to_pc_xy(cart_img: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Convert cartesian occupancy image (RBINS x ABINS) to point cloud in METERS (x,y),
    matching the convention used by the .pcd pipeline (pc_compare.py compares x,y).
    """
    coords = np.argwhere(cart_img >= threshold)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ix = coords[:, 0]
    iy = coords[:, 1]
    x = x_axis_grid[ix]
    y = y_axis_grid[iy]
    return np.stack([x, y], axis=1).astype(np.float32)

# -----------------------------------------
# pc_compare.py distance logic (copy 1:1)
# -----------------------------------------
def bin_pc(pc: np.ndarray, bin_size: float) -> np.ndarray:
    if bin_size == 0:
        return pc
    if pc.size == 0:
        return pc
    binned = np.floor(pc / bin_size) * bin_size
    # unique rows
    binned = np.unique(binned, axis=0)
    return binned

def pc_distance(pc_A: np.ndarray, pc_B: np.ndarray, metric: str, bin_size: float) -> float:
    """
    pc_A, pc_B: (N,2) arrays (x,y) in meters.
    metric: 'chamfer' or 'mod_hausdorff'
    bin_size: 0 to disable binning (as in your pc_compare.py default)
    """
    tree_A = cKDTree(pc_A)
    tree_B = cKDTree(pc_B)

    # nearest neighbor distances
    dA, _ = tree_A.query(pc_B, k=1)  # B -> A
    dB, _ = tree_B.query(pc_A, k=1)  # A -> B

    if metric == "chamfer":
        return float(np.mean(dA) + np.mean(dB))
    elif metric == "mod_hausdorff":
        return float(max(np.mean(dA), np.mean(dB)))
    else:
        raise ValueError(f"Unknown metric: {metric}")



def update_confusion_from_logits(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int]:

    pred_bin = (pred >= threshold)
    gt_bin   = (gt >= 0.5)

    tp = (pred_bin & gt_bin).sum().item()
    fp = (pred_bin & (~gt_bin)).sum().item()
    fn = ((~pred_bin) & gt_bin).sum().item()
    tn = ((~pred_bin) & (~gt_bin)).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def f1_from_counts(tp: int, fp: int, fn: int, eps: float = 1e-12) -> float:
    return float((2.0 * tp) / (2.0 * tp + fp + fn + eps))


# -------------------------------
# CV split utilities
# -------------------------------
_EXP_RE = re.compile(r"^(?P<day>\d+)_exp(?P<exp>\d+)_\d+\.png$", re.IGNORECASE)

def extract_experiment_key(path: str) -> str:
    """
    020925_exp1_114.png -> 020925_exp1
    """
    base = os.path.basename(path)
    m = _EXP_RE.match(base)
    if m is None:
        # fallback: first two fields split by "_"
        parts = base.split("_")
        return "_".join(parts[:2])
    return f"{m.group('day')}_exp{int(m.group('exp'))}"

def kfold_split(experiments: List[str], k: int, seed: int) -> List[np.ndarray]:
    exps = list(experiments)
    rnd = random.Random(seed)
    rnd.shuffle(exps)
    return list(np.array_split(exps, k))

def infer_label_path_for_index(dataset: SLAMDataset, i: int) -> str:
    """
     get the label filename that corresponds to dataset index i.
    - If history==0: label file is lidar_files[i]
    - Else: label file is label_sequences[i] (aligned with __getitem__)
    """
    if getattr(dataset, "history", 0) == 0:
        return dataset.lidar_files[i]
    return dataset.label_sequences[i]

# -------------------------------
# CLI (same as train_slam + CV flags)
# -------------------------------
valid_folders = {"moh", "ste", "alex"}

def parse_args():
    # train_slam parser 
    return train_parse_cli()



# -------------------------------
# Main
# -------------------------------
def main():
    print(f"\n------------------ cross_val_single.py ----------------")
    args = parse_args()
    params = train_build_params_from_args(args)
    if args.source is None: params["source_folder"]=SOURCE_FOLDER
    if args.expt is None: params["expt"]=EXPT
    if args.model is None: params["model_architecture"]=MODEL_ARCHITECTURE
    if args.optimizer is None: params["optim"]=OPTIM
    if args.epochs is None: params["num_epochs"]=EPOCHS
    if args.range_eps is None: params["range_eps"]=RANGE_EPS
    if args.range_clip is None: params["range_clip_max"]=RANGE_CLIP_MAX
    if args.shuffle_data is None: params["shuffle_data"]=SHUFFLE_DATA
    OUT_JSON = f"./cv_results/cv_{params['model_name']}.json"
    # Device (same convention: gpu flag not passed here; use cuda if available)
    device = torch.device('cuda' if params['gpu'] == 1 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True



    # Dataset path (train_slam uses SOURCE_FOLDER located under SCRIPT_DIR)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, params["source_folder"])

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)



    # Load full dataset
    full_dataset = SLAMDataset(dataset_root, split="train", M=params["history"],num_augs=augmentation_map[params["augmentation"]])
    n = len(full_dataset)
    if n == 0:
        raise RuntimeError("Training dataset is empty.")

    # Map sample indices to experiment key (by label filename, aligned with __getitem__)
    idx_to_exp = np.array([extract_experiment_key(infer_label_path_for_index(full_dataset, i)) for i in range(n)])
    experiments = sorted(set(idx_to_exp))
    if len(experiments) < K_FOLDS:
        raise ValueError(f"Not enough experiments ({len(experiments)}) for kfolds={K_FOLDS}.")

    folds = kfold_split(experiments, K_FOLDS, SEED)

    results = {
        "config": {
            "source": params["source_folder"],
            "model_architecture": params["model_architecture"],
            "num_epochs": params["num_epochs"],
            "model_batch_size": params["batch_size"],
            "lr": params["lr"],
            "bcew": params["bcew"],
            "dicew": params["dicew"],
            "use_range_weight":params["use_range_weight"],
            "optimizer": params["optim"],
            "augmentation":params["augmentation"],
            "history": params["history"],
            "shuffle_data": params["shuffle_data"],
            "adam_beta1": params["adam_beta1"],
            "adam_beta2": params["adam_beta2"],
            "lrs_factor": params["lrs_factor"],
            "lrs_patience": params["lrs_patience"],
            "range_p": params["range_p"],
            "clip_max": params["range_clip_max"],
            "eps": params["range_eps"],
            "kfolds": K_FOLDS,
            "seed": SEED,
            "pc_distance": {"bin_size": float(BIN_SIZE), "threshold": float(DISTANCE_THRESHOLD)},
        },
        "folds": [],
        "mean": {},
        "median":{},
        "standard_deviation":{},
        "f1_score":{},
        "false_positives":{},
    }

    # -------------------------------
    # Fold loop
    # -------------------------------
    for k in range(K_FOLDS):
        val_exps = set(folds[k])
        train_exps = set(experiments) - val_exps

        train_idx = np.where(np.isin(idx_to_exp, list(train_exps)))[0]
        val_idx = np.where(np.isin(idx_to_exp, list(val_exps)))[0]

        print(f"\n--- Fold {k+1}/{K_FOLDS} ---")
        print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")

        train_set = Subset(full_dataset, train_idx.tolist())
        val_set = Subset(full_dataset, val_idx.tolist())

            
        val_loader = DataLoader(
            val_set,
            batch_size=params["batch_size"],
            shuffle=False, #no need to shuffle when in validation
            drop_last=False,
            num_workers=8,
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
            prefetch_factor=4,
        )

        assert len(train_set) > 0 and len(val_set) > 0

        model = run_training(
            params,
            dataset_override=train_set,
            seed=SEED + k,
            save_artifacts=False,
        )


        # -------------------------------
        # Validate distances (no artifacts saved)
        # -------------------------------
        model.eval()
        chamfers: List[float] = []
        modhaus: List[float] = []

        used = 0
        skipped_empty = 0

        tp_total = fp_total = fn_total = tn_total = 0
        
        use_amp = (device.type == "cuda")

        with torch.no_grad():
            for radar, lidar in val_loader:
                radar = radar.to(device, non_blocking=True)
                lidar = lidar.to(device, non_blocking=True)

                pred = model(radar)

                # update confusion matrix (pixel-wise, directly on polar)
                pred_det = pred.detach()
                lidar_det = lidar.detach()
                tp, fp, fn, tn = update_confusion_from_logits(pred_det, lidar_det)

                tp_total += tp
                fp_total += fp
                fn_total += fn
                tn_total += tn

                pred_np = pred.detach().cpu().numpy()  # (B,1,H,W)
                gt_np = lidar.detach().cpu().numpy()  # (B,1,H,W)

                for b in range(pred_np.shape[0]):
                    pred_polar = pred_np[b, 0]
                    gt_polar = gt_np[b, 0]

                    pred_cart = pol_to_cart(pred_polar)
                    gt_cart = pol_to_cart(gt_polar)

                    pc_pred = img_to_pc_xy(pred_cart, threshold=float(DISTANCE_THRESHOLD))
                    pc_gt = img_to_pc_xy(gt_cart, threshold=float(DISTANCE_THRESHOLD))

                    if pc_pred.size == 0 or pc_gt.size == 0:
                        skipped_empty += 1
                        continue

                    ch = pc_distance(pc_gt, pc_pred, "chamfer", float(BIN_SIZE))
                    mh = pc_distance(pc_gt, pc_pred, "mod_hausdorff", float(BIN_SIZE))

                    chamfers.append(ch)
                    modhaus.append(mh)
                    used += 1

        mean_ch = float(np.mean(chamfers)) 
        mean_mh = float(np.mean(modhaus)) 
        median_ch = float(np.median(chamfers)) 
        median_mh = float(np.median(modhaus)) 
        fold_f1 = f1_from_counts(tp_total, fp_total, fn_total)
        fp_rate = float(fp_total / (fp_total + tn_total + 1e-12))


        print(f"Fold {k+1} distances | used={used}, skipped_empty={skipped_empty} | Chamfer={mean_ch:.6f} | ModHaus={mean_mh:.6f}| Chamfer={median_ch:.6f} | ModHaus={median_mh:.6f}")

        results["folds"].append({
            "fold": k + 1,
            "train_experiments": sorted(list(train_exps)),
            "val_experiments": sorted(list(val_exps)),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "used_for_distance": int(used),
            "skipped_empty": int(skipped_empty),
            "mean_chamfer": mean_ch,
            "mean_modhaus": mean_mh,
            "median_chamfer": median_ch,
            "median_modhaus": median_mh,
        })
        results["folds"][-1].update({
            "tp": int(tp_total),
            "fp": int(fp_total),
            "fn": int(fn_total),
            "tn": int(tn_total),
            "f1_pixel": float(fold_f1),
            "fp_rate_pixel": float(fp_rate),
        })


    # Aggregate
    chs = [f["mean_chamfer"] for f in results["folds"] if np.isfinite(f["mean_chamfer"])]
    mhs = [f["mean_modhaus"] for f in results["folds"] if np.isfinite(f["mean_modhaus"])]

    results["mean"] = {
        "mean_chamfer": float(np.mean(chs)) if chs else float("nan"),
        "mean_modhaus": float(np.mean(mhs)) if mhs else float("nan"),
    }
    results["median"] = {
        "median_chamfer": float(np.median(chs)) if chs else float("nan"),
        "median_modhaus": float(np.median(mhs)) if mhs else float("nan"),
    }

    print("\n>>> MEAN over folds")
    print(f"Chamfer = {results['mean']['mean_chamfer']:.6f}")
    print(f"ModHaus = {results['mean']['mean_modhaus']:.6f}")
    print(f"Chamfer = {results['median']['median_chamfer']:.6f}")
    print(f"ModHaus = {results['median']['median_modhaus']:.6f}")

    f1s = [f["f1_pixel"] for f in results["folds"] if np.isfinite(f.get("f1_pixel", float("nan")))]
    fps = [f["fp"] for f in results["folds"] if "fp" in f]

    results["f1_score"] = {
        "mean_f1_pixel": float(np.mean(f1s)) if f1s else float("nan"),
        "median_f1_pixel": float(np.median(f1s)) if f1s else float("nan"),
    }
    results["false_positives"] = {
        "mean_fp": float(np.mean(fps)) if fps else float("nan"),
        "median_fp": float(np.median(fps)) if fps else float("nan"),
    }


    # Write JSON (only artifact)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved CV summary to: {OUT_JSON}")
    
if __name__ == "__main__":
    main()
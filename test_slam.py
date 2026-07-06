# File for testing RadarHD on the NEW SLAM-RF processed dataset

import time
import os
import json
import torch
import numpy as np
from torchsummary import summary
from PIL import Image

from train_test_utils.model import *
from train_test_utils.dataloader_slam import SLAMDataset   # <-- nuovo dataloader

import argparse

from train_slam import (
    model_map as model_map,
)
from create_dataset.sync_slam_rf import valid_enhancements, augmentation_map


"""
## Parameters for model loading
"""

GLOBAL_TEST_PARAMS = {
    'model_name': '13',          # folder prefix used during training
    'expt': 1,                   # experiment number from training
    'dt': '20251215-124042',     # now we changed to using as dt only the nomber of epochs the model was trained for, to make it possible to do gridsearch
    'epoch_num': None   ,            # checkpoint to load (XXX.pt_gen), keep in mind che since the epochs start from 0, a model trained with 130 epochs will reach maximum epoch 129
    'gpu': 1,
}


SOURCE_FOLDER = "dataset_SLAM"
SOURCE_DIR = "./dataset_SLAM"

DEST_FOLDER = "logs"
DEST_DIR = "./logs"

RBINS_RADAR, ABINS_RADAR = 256, 180  # radar image dimensions

def load_test_loader(history,num_aug):
    """
    Loads the NEW dataset_SLAM/test/ using the SLAMDataset loader.
    """
    print("Loading dataset_SLAM test split...")

    test_set = SLAMDataset(SOURCE_DIR, split="test", M=history, num_augs=num_aug)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
                )

    # filenames without extension
    ordered_filenames = [os.path.basename(f).replace(".png", "") 
                         for f in test_set.lidar_files]

    print("# test samples:", len(test_set))
    return test_loader, ordered_filenames

def find_last_checkpoint(log_dir):
    epochs = []
    for f in os.listdir(log_dir):
        if f.endswith(".pt_gen"):
            epochs.append(int(f.split(".")[0]))
    if not epochs:
        raise RuntimeError(f"No checkpoints found in {log_dir}")
    return max(epochs)


def main():
    print(f"\n--------------------  test_slam.py ------------------------------")

    #this is train specific parsing. for model parameters all the saved ones will be used
    args = parse_cli()

    if args.model_name is not None:
        GLOBAL_TEST_PARAMS["model_name"] = args.model_name

    if args.expt is not None:
        GLOBAL_TEST_PARAMS["expt"] = args.expt 
    
    if args.dt is not None:
        GLOBAL_TEST_PARAMS["dt"] = args.dt
    
    if args.epoch_num is not None:
        GLOBAL_TEST_PARAMS["epoch_num"] = args.epoch_num
    
    
    
    
    # --------------------------------------------------
    # SET WORKING DIRECTORY
    # --------------------------------------------------

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    

    global SOURCE_FOLDER,DEST_FOLDER, SOURCE_DIR, DEST_DIR
    if args.source is not None:
        SOURCE_FOLDER = args.source
        SOURCE_DIR = os.path.join(SCRIPT_DIR, SOURCE_FOLDER) #dataset folder where to take the data

    if args.dest is not None:
        DEST_FOLDER = args.dest
        DEST_DIR   = os.path.join(SCRIPT_DIR, DEST_FOLDER) #log folder where to save data


    # -------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------
    device = torch.device('cuda' if GLOBAL_TEST_PARAMS['gpu'] == 1 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pin_memory = (device.type == "cuda")

    torch.manual_seed(0)


    # -------------------------------------------------------------------
    # LOAD TRAIN PARAMS (to know history, architecture, etc.)
    # -------------------------------------------------------------------
    name_str = GLOBAL_TEST_PARAMS['model_name'] + '_' + str(GLOBAL_TEST_PARAMS['expt']) + '_' + GLOBAL_TEST_PARAMS['dt']
        
    LOG_DIR = os.path.join(DEST_DIR,name_str) #folder inside DEST_FOLD that encodes the parameters, this is the folder that will contain weights
    os.makedirs(LOG_DIR, exist_ok=True)


    with open(os.path.join(LOG_DIR, 'params.json'), 'r') as f:
        train_params = json.load(f)

    history = train_params["history"]
    in_channels = history + 1

    # -------------------------------------------------------------------
    # LOAD TEST DATA
    # -------------------------------------------------------------------
    print(f"Loading {SOURCE_FOLDER}...")

    if "augmentation" in train_params:
        aug=augmentation_map[train_params["augmentation"]]
    else:
        aug=1
    
    test_loader, ordered_filenames = load_test_loader(history,aug)


    # -------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------
    if "model_architecture" in train_params:
        model_arch = train_params["model_architecture"]
    else:
        model_arch = "unet3"

    if model_arch not in model_map:
        raise ValueError(f"Model {model_arch} is not supported. Must be unet1–6.")

    ModelClass = model_map[model_arch]

    gen = ModelClass(in_channels, 1).to(device)
   

    #if not otherwise specified, we reload last epoch
    requested_epoch = GLOBAL_TEST_PARAMS["epoch_num"]
    if requested_epoch is None:
        # default: last available
        reloaded_epoch = find_last_checkpoint(LOG_DIR)
        print(f"Reloading last available checkpoint: epoch {reloaded_epoch}")
    else:
        epoch_str = f"{requested_epoch:03d}"
        model_file = os.path.join(LOG_DIR, epoch_str + ".pt_gen")

        if os.path.isfile(model_file):
            reloaded_epoch = requested_epoch
        else:
            print(
                f"[WARN] Requested epoch {requested_epoch} not found. "
                f"Reloading last available checkpoint instead."
            )
            reloaded_epoch = find_last_checkpoint(LOG_DIR)
    
    epoch_str = f"{reloaded_epoch:03d}"
    model_file = os.path.join(LOG_DIR, epoch_str + ".pt_gen")

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")


    checkpoint = torch.load(model_file, map_location=device)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()

    # -------------------------------------------------------------------
    # OUTPUT FOLDER
    # -------------------------------------------------------------------
    save_path = os.path.join(LOG_DIR, "test_imgs")
    os.makedirs(save_path, exist_ok=True)

    # -------------------------------------------------------------------
    # TEST LOOP
    # -------------------------------------------------------------------
    t0 = time.time()

    for i, (radar, lidar) in enumerate(test_loader):

        radar, lidar = radar.to(device), lidar.to(device)

        with torch.no_grad():
            pred = gen(radar)

        # Convert predictions to [0..255] image
        pred_np = np.squeeze(pred.cpu().numpy())
        pred_np = (pred_np * 255).astype(np.uint8)
        pred_img = Image.fromarray(pred_np)

        # Convert label
        label_np = np.squeeze(lidar.cpu().numpy())
        label_np = (label_np * 255).astype(np.uint8)
        label_img = Image.fromarray(label_np)

        # Save
        fname = ordered_filenames[i]
        pred_img.save(os.path.join(save_path, f"{epoch_str}_{fname}_pred.png"))
        label_img.save(os.path.join(save_path, f"{epoch_str}_{fname}_label.png"))

        print(f"[{i+1}/{len(test_loader)}] {fname}")

    t1 = time.time()
    print("Inference time:", t1 - t0)



def parse_cli():

    parser = argparse.ArgumentParser(
        description="Build SLAM_RF dataset with configurable source/destination folders."
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
        "--model-name",
        type=str,
        required=False,
        default=None,
        help="Model identifier to prepend to log directory.",
    )
    parser.add_argument(
        "--expt",
        type=int,
        required=False,
        default=None,
        help="Experiment identifier to append to log directory.",
    )
    parser.add_argument( #here represents the epoch you want to reload
        "--epoch-num",
        type=int,
        required=False,
        default=None,
        help="Epoch number to get parameters from.",
    )
    
    parser.add_argument(
    "--dt",
    type=str,
    required=False,
    default=None,
    )
    
    return parser.parse_args()


# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------

if __name__ == "__main__":
    main()
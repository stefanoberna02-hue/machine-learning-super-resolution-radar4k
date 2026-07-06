
# ==============================================================================
# EXPECTED PROJECT & DATASET DIRECTORY STRUCTURE
# ==============================================================================
#
# This training script assumes the following directory layout.
# Paths are resolved RELATIVE to the location of THIS FILE (__file__),
# not relative to the execution shell. If everything is kept as the original structure provided by our group, no further changes needed
#
# ------------------------------------------------------------------------------
# 1. PROJECT ROOT (same level as this script)
# ------------------------------------------------------------------------------
#
# /project-2-radar4k/
# ├── train_slam.py              # THIS SCRIPT
# ├── imgs/                      # (optional, auto-created)
# │   └── <run_name>/            # saved qualitative predictions (if enabled)
# │       ├── epoch_000/
# │       ├── epoch_010/
# │       └── ...
# ├── logs/                      # (auto-created)
# │   └── <run_name>/            # one folder per training run
# │       ├── params.json        # full hyperparameter dump
# │       ├── train_log.txt      # textual training log
# │       ├── events.out.tfevents*  # TensorBoard logs
# │       ├── 000.pt_gen         # model checkpoint (epoch 0)
# │       ├── 010.pt_gen
# │       └── ...
# ├── dataset_SLAM_<EXPT>/       # (passed via --source)
# │   ├── train/
# │   │   ├── radar/
# │   │   │   ├── <day>_<id>.png
# │   │   │   └── ...
# │   │   └── lidar/
# │   │       ├── <day>_<id>.png
# │   │       └── ...
# │   └── test/
# │       ├── radar/
# │       └── lidar/
# │
# └── train_test_utils/
#     ├── model.py               # UNet1 / UNet3 / UNet5 / UNet6
#     ├── dataloader_slam.py     # SLAMDataset
#     └── dice_score.py
#
# ------------------------------------------------------------------------------
# 2. DATASET ASSUMPTIONS
# ------------------------------------------------------------------------------
#
# - Radar and LiDAR images are already preprocessed and stored as PNGs.
# - Radar images are shaped [RBINS=256, ABINS=180].
# - LiDAR images are aligned to the same polar grid.
# - File naming convention:
#       <day>_<global_index>.png
#   ensures global uniqueness across days.
#
# ------------------------------------------------------------------------------
# 3. TRAINING OUTPUTS
# ------------------------------------------------------------------------------
#
# - Logs, checkpoints, and TensorBoard events are stored in:
#       ./logs/<run_name>/
#
# - Optional qualitative image outputs (if enabled) are stored in:
#       ./imgs/<run_name>/epoch_<N>/
#
# - <run_name> encodes:
#       model identifier + dataset experiment id + number of epochs
#
# ------------------------------------------------------------------------------
# 4. MINIMAL REQUIRED CLI ARGUMENTS
# ------------------------------------------------------------------------------
#
#   --model-name <ID>              (e.g. 13)
#   --expt <EXPT_ID>               (e.g. 1)
#   --source <dataset_SLAM_EXPT>   (e.g. SLAM_dataset_112005)
#
# All cli() parameters are OPTIONAL and have safe defaults.
#
# ==============================================================================

import time
import os
import datetime
import json
import gc
import random

import torch
import torch.optim as optim
import numpy as np
from torchsummary import summary

from train_test_utils.model import *     
from train_test_utils.dice_score import dice_loss
from train_test_utils.dataloader_slam import SLAMDataset   # <-- IMPORT NEW DATASET

import argparse
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import inspect


ABINS_RADAR=180
RBINS_RADAR=256

# -------------------------------------------------------------
# RANGE-WEIGHTED BCE (more weight at higher range bins)
# -------------------------------------------------------------

valid_optimizers = {"adam", "rmsprop","adamw"}
valid_architectures={"unet1", "unet2","unet3","unet4","unet5","unet6"}
#used to select the model architecture from command line
model_map = {
    "unet1": UNet1,
    "unet2": UNet2, 
    "unet3": UNet3,
    "unet4": UNet4,#in final version of the code UNet4 is not implemented but keps as a developement opporturnity
    "unet5": UNet5,
    "unet6": UNet6,
   }

from create_dataset.sync_slam_rf import valid_enhancements, augmentation_map

"""
## Constants and hyperparameters
"""
GLOBAL_PARAMS = {
    'model_name': '13',    #cli()    #Refers to the model definition being used e.g. 'unet3' with or without weight dropout  
    'dt':'20251215-124042',
    'expt': 112005,        #cli() #Refers to the number of dataset that the model was trained with e.g. expt 11000 = dataset_SLAM_11000, convention is to use first digit 1-6 for enhancement level, second digit for intensity handling (1 binary, 2 mag thresh, 3 cfar), followed by the threshold value used (000 if binary)
    'model_architecture': 'unet3',#cli()
    'num_epochs': 130,      #cli()
    'batch_size': 6, 
    'lr': 1e-4,             #cli()
    'bcew': 0.9,            #cli()
    'dicew': 0.1,                    #1-bcew
    'use_range_weight':False,        #penalization for the loss function based on range, parametrized by range_eps 
    'range_eps': 0.05 ,      #cli()       # avoid near-zero weights at close range
    'range_p' : 1.5  ,       #cli()       # moderate exponent
    'range_clip_max': 3.0 ,  #cli()      # avoid far range dominating too much
    'optim': 'adam',         #cli()
    'augmentation': 'none',#cli() #refers to the augmentation made on the original dataset, required for dataloading
    'expt_caption': '',
    'history': 40,              #in-channels will be history+1
    'reload': False,            #cli() when passed reloads last epoch available in the folder with same name as the model and correctly restars epoch numbering from reload_epoch +1
    'reload_namestr': '',
    'reload_epoch': -1,
    'gpu': 1,
    'shuffle_data': True,       #cli()
    'adam_beta1': None,         #cli()
    'adam_beta2': None,         #cli()
    'lrs_factor': 0.1,          #cli()
    'lrs_patience': 5,          #cli()
    'source_folder':"dataset_SLAM",#cli()
    'dest_folder':"logs", #cli()
}

def find_last_checkpoint(log_dir: str) -> int:
    if not os.path.isdir(log_dir):
        return -1
    epochs = []
    for f in os.listdir(log_dir):
        if f.endswith(".pt_gen"):
            try:
                epochs.append(int(f.split(".")[0]))
            except ValueError:
                pass
    return max(epochs) if epochs else -1


def make_range_weight_map(h, device, range_eps,range_p,range_clip_max):
    """
    Returns a tensor of shape (1,1,H,1) to be broadcast on (B,1,H,W).
    Range increases with row index (0 -> near, H-1 -> far).
    """
    r = torch.linspace(0.0, 1.0, steps=h, device=device).view(1, 1, h, 1)
    w = (r + range_eps) ** range_p
    w = w / w.mean()                  # normalize mean weight to 1
    w = torch.clamp(w, 1.0 / range_clip_max, range_clip_max)
    return w


def run_training(params, dataset_override=None, save_artifacts=True, seed=None):

    print(f"\n--------------------  train_slam.py ------------------------------")

    torch.backends.cudnn.benchmark = True#fixing shape of input throught all training, to get pytorch speedup

    # --------------------------------------------------
    # CHECK WORKING DIRECTORY
    # --------------------------------------------------
   

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    SOURCE_FOLDER = params['source_folder']

    DEST_FOLDER = params['dest_folder']

    SOURCE_DIR = os.path.join(SCRIPT_DIR, SOURCE_FOLDER) #dataset folder where to take the data
    DEST_DIR   = os.path.join(SCRIPT_DIR, DEST_FOLDER) #log folder where to save data

    
    
    # -------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------
    device = torch.device('cuda' if params['gpu'] == 1 and torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    pin_memory = (device.type == "cuda")
    # -------------------------------------------------------------
    # SLAM DATA LOADING 
    # -------------------------------------------------------------
    if dataset_override is None:
        print(f"Loading {SOURCE_FOLDER}...")
    else:
        print(f"Overriding default dataset with provided one...")


    # Use new SLAMDataset
    if dataset_override is None:
        training_set = SLAMDataset(
                SOURCE_DIR, 
                split="train", 
                M=params['history'],
                num_augs=augmentation_map[params['augmentation']]
                ) 
    else:
        training_set = dataset_override
   

    drop_last = len(training_set) >= params["batch_size"]

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=params['batch_size'],
        shuffle=params['shuffle_data'],       
        drop_last=drop_last,
        num_workers=8,       
        pin_memory=pin_memory,    
        persistent_workers=True
    )
    # -------------------------------------------------------------
        
    print(torch.__version__)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)


    # -------------------------------------------------------------
    # LOG DIRECTORY
    # -------------------------------------------------------------
    name_str = f"{params['model_name']}_{params['expt']}_{params['dt']}"

    LOG_DIR = os.path.join(DEST_DIR,name_str) #folder inside DEST_FOLD that encodes the parameters, this is the folder that will ocntain weights
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if save_artifacts:
        writer = SummaryWriter(LOG_DIR)
    
        with open(os.path.join(LOG_DIR, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)

    train_log = os.path.join(LOG_DIR, 'train_log.txt')

    # -------------------------------------------------------------
    # IMAGE OUTPUT DIRECTORY (for plots / qualitative outputs)
    # -------------------------------------------------------------
    IMGS_ROOT = os.path.join(SCRIPT_DIR, "imgs")
    IMG_DIR = os.path.join(IMGS_ROOT, name_str)
    os.makedirs(IMG_DIR, exist_ok=True)


    # -------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------
    in_channels = params['history'] + 1  # radar uses M past + current


    if params["model_architecture"] not in model_map:
        raise ValueError(f"Model {params['model_architecture']} is not supported. Must be unet1–6.")

    ModelClass = model_map[params["model_architecture"]]

    gen = ModelClass(in_channels, 1).to(device)

    # IMPORTANT: summary expects shape (C, H, W)
    if save_artifacts:
        summary(gen, (in_channels, RBINS_RADAR, ABINS_RADAR))  # radar images are 256×180

    # -------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------
    if params["optim"] not in valid_optimizers:
        raise ValueError(
            f"Unsupported optimizer '{params['optim']}'. "
            f"Valid options are: {sorted(valid_optimizers)}"
        )


    if params['optim'] == 'adam':
        if params['adam_beta1'] is not None and params['adam_beta2'] is not None:
            gen_optimizer = optim.Adam(gen.parameters(),
                                   betas=(params["adam_beta1"], params["adam_beta2"]),
                                   lr=params['lr'],
                                   weight_decay=0.0005)
        else:
            gen_optimizer = optim.Adam(gen.parameters(),
                                   lr=params['lr'], 
                                  weight_decay=0.0005)
    elif params['optim'] == 'rmsprop':
        gen_optimizer = optim.RMSprop(gen.parameters(),
                                      lr=params['lr'],
                                      weight_decay=1e-8,
                                      momentum=0.9)
    elif params["optim"] == 'adamw':
    # AdamW = decoupled weight decay (corretto)
        if params['adam_beta1'] is not None and params['adam_beta2'] is not None:
            gen_optimizer = optim.AdamW(
                gen.parameters(),
                lr=params['lr'],
                betas=(params["adam_beta1"],
                params["adam_beta2"]),
                weight_decay=5e-4
            )
        else:
            gen_optimizer = optim.AdamW(
                gen.parameters(),
                lr=params['lr'],
                weight_decay=5e-4
            )
    else:
        gen_optimizer = optim.Adam(gen.parameters(),
                                   lr=params['lr'],
                                   weight_decay=0.0005)
        print("WARNING: OPTIMIZER NOT RECOGNIZED, USING ADAM.")

    # Define the  scheduler

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer,
        mode='min',  # We want the monitored loss to decrease
        factor=params["lrs_factor"],  # Multiply LR by 0.1 when reduced
        patience=params["lrs_patience"],  # Wait for 5 epochs with no improvement
        threshold=0.0001,  # Only consider significant changes
        cooldown=2,  # Wait 2 epochs after an LR reduction before resuming monitoring
        min_lr=1e-7  # Set a lower bound for the learning rate
    )

    # -------------------------------------------------------------
    # LOSS FUNCTIONS
    # -------------------------------------------------------------
    bce_loss_fn = torch.nn.BCELoss(reduction="none")


    # -------------------------------------------------------------
    # LOAD CHECKPOINT (OPTIONAL)
    # -------------------------------------------------------------
    start_epoch=0
    if params['reload'] and (params["reload_epoch"] is None or params['reload_epoch'] < 0):
        log_dir = os.path.join(DEST_DIR, params["reload_namestr"])
        params['reload_epoch'] = find_last_checkpoint(log_dir)
        if params['reload_epoch'] < 0:
            raise RuntimeError(
                f"Reload requested, but no valid checkpoint (.pt_gen) was found in "
                f"'{os.path.join(DEST_DIR, params['reload_namestr'])}'."
            )


    if params['reload']:
        epoch_num = '%03d' % params['reload_epoch']
        model_file = os.path.join(DEST_DIR, params["reload_namestr"], f"{epoch_num}.pt_gen")
        checkpoint = torch.load(model_file)
        gen.load_state_dict(checkpoint['state_dict'])
                
        if "optimizer_state_dict" in checkpoint:
            gen_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = params['reload_epoch'] + 1

    # -------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------
    t0 = time.time()
    epoch_losses = []  # will store average training loss per epoch (for the final plot)

    range_w = None  # cache weight map once we know output H

    for epoch in range(start_epoch, params['num_epochs']):
        print("="*10, "Epoch", epoch, "="*10)
        gen.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        epoch_bce_sum = 0.0
        epoch_dice_sum = 0.0



        for batch_idx, (radar, lidar) in enumerate(train_loader):
            radar = radar.to(device, non_blocking=True)
            lidar = lidar.to(device, non_blocking=True)


            gen_optimizer.zero_grad(set_to_none=True)


            # --------------------------------------------------
            # BCE loss (pixel-wise, optional range weighting)
            # --------------------------------------------------
            generated = gen(radar)
            
            generated_f = generated.float()
            lidar_f     = lidar.float()

            bce_pix = bce_loss_fn(generated_f, lidar_f)

            if params['use_range_weight']:
                h = bce_pix.shape[-2]
                if (range_w is None) or (range_w.shape[-2] != h) or (range_w.device != device):
                    range_w = make_range_weight_map(
                        h, device,
                        params['range_eps'],
                        params['range_p'],
                        params['range_clip_max']
                    )
                loss1 = (range_w * bce_pix).mean()
            else:
                loss1 = bce_pix.mean()

            loss2 = dice_loss(generated_f, lidar_f)
            loss = params['bcew'] * loss1 + params['dicew'] * loss2


            loss.backward()
            gen_optimizer.step()


            epoch_loss_sum += loss.detach()
            num_batches += 1

            epoch_bce_sum += loss1.detach()
            epoch_dice_sum += loss2.detach()


            if save_artifacts and batch_idx % 20 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("train/total_batch", loss.detach(), global_step)


            if batch_idx % 500 == 0:
                loss_val = float(loss.detach().cpu())
                msg = f"Epoch {epoch} [{batch_idx}/{len(train_loader)}]  Loss={loss_val:.6f}"
                print(msg)
                if save_artifacts:
                    with open(train_log, "a+") as f:
                        f.write(msg + "\n")
            
        avg_epoch_loss = epoch_loss_sum.item() / max(1, num_batches)
        epoch_losses.append(avg_epoch_loss)

        scheduler.step(avg_epoch_loss)

        if save_artifacts:
            writer.add_scalar("train_loss", avg_epoch_loss, epoch)
            writer.add_scalar("train_bce", (epoch_bce_sum / num_batches).item(), epoch)
            writer.add_scalar("train_dice", (epoch_dice_sum / num_batches).item(), epoch)

        current_lr = gen_optimizer.param_groups[0]['lr']

        msg_epoch = f"[EPOCH SUMMARY] Epoch {epoch} | avg_train_loss={avg_epoch_loss:.6f} | Current LR: {current_lr}"
        print(msg_epoch)
        if save_artifacts:
            with open(train_log, "a+", encoding="utf-8") as f:
                f.write(msg_epoch + "\n")


        # Save checkpoint
        if save_artifacts:
            if epoch % 10 == 0:
                checkpoint = {
                    'state_dict': gen.state_dict(),
                    'optimizer_state_dict': gen_optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(LOG_DIR, f'{epoch:03d}.pt_gen'))



    # -------------------------------------------------------------
    # FINAL TRAINING-LOSS PLOT (single figure)
    # -------------------------------------------------------------
    
    if save_artifacts:
        epochs_axis = np.arange(len(epoch_losses))

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs_axis, epoch_losses, linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Average training loss")
        plt.title(
            f"Training loss | arch={params['model_architecture']} | model_id={params['model_name']} | expt={params['expt']}\n"
            f"loss = {params['bcew']}*BCE + {params['dicew']}*Dice"
        )

        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(IMG_DIR, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"[INFO] Saved training loss curve to: {plot_path}")
        with open(train_log, "a+", encoding="utf-8") as f:
            f.write(f"[INFO] Saved training loss curve to: {plot_path}\n")

    if save_artifacts:
        writer.close()

    print("Training time:", time.time() - t0)

    return gen

def main():
    args = parse_cli()
    params = build_params_from_args(args) 
    run_training(params)


def build_params_from_args(args: argparse.Namespace) -> dict:
    params = GLOBAL_PARAMS.copy()
    
    if args.source is not None:
        params["source_folder"] = args.source
    
    if args.dest is not None:
        params["dest_folder"] = args.dest
   
    if args.model_name is not None:
        params["model_name"] = args.model_name
    
    if args.dt is not None:
        params["dt"] = args.dt

    if args.expt is not None:
        params["expt"] = args.expt 
    
    if args.model is not None:
        params["model_architecture"] = args.model
    

    if args.epochs is not None:
        params["num_epochs"] = args.epochs  
    
    if args.learning_rate is not None:
        params["lr"] = args.learning_rate
    
    if args.bcew is not None:
        params["bcew"] = args.bcew  
        params["dicew"] = 1-args.bcew  
  
    
    if args.enhancement is not None:
        params["augmentation"] = args.enhancement
    if params["augmentation"] not in augmentation_map:
        raise ValueError(
            f"Invalid augmentation '{params['augmentation']}'. "
            f"Valid: {list(augmentation_map.keys())}"
        )
    if args.shuffle_data is not None:
        params["shuffle_data"] = args.shuffle_data
    
    if args.optimizer is not None:
        params["optim"] = args.optimizer
      
    if args.adam_beta1 is not None:
        params["adam_beta1"] = args.adam_beta1
    
    if args.adam_beta2 is not None:
        params["adam_beta2"] = args.adam_beta2
    
    if args.lrs_factor is not None:
        params["lrs_factor"] = args.lrs_factor  
    
    if args.lrs_patience is not None:
        params["lrs_patience"] = args.lrs_patience 
    
    if args.range_p is not None:
        if args.range_p>0:
            params['range_p']=args.range_p
            params['use_range_weight'] = True
        else:
            params['use_range_weight'] = False
       
    if args.range_eps is not None:
        params["range_eps"] = args.range_eps

    if args.range_clip is not None:
        params["range_clip_max"] = args.range_clip 
    
    if (args.reload is not None) and  (args.reload):
        params["reload"] = args.reload
        if not params['reload_namestr']:
            params['reload_namestr'] = f"{params['model_name']}_{params['expt']}_{params['dt']}"


    return params
    
    
    


def parse_cli():

    parser = argparse.ArgumentParser(
        description="Build SLAM_RF dataset with configurable source/destination folders."
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        required=False,
        default=None,
        help="Name of the SOURCE folder (located on the same level of this script)."
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
        help="Name given to the model and corresponding folder.",
    )
    parser.add_argument(
    "--dt",
    type=str,
    required=False,
    default=None,
    )
    parser.add_argument(
        "--expt",
        type=int,
        required=False,
        default=None,
        help="Experiment identifier to append to log directory.",
    )
    parser.add_argument(
    "--model",
    type=str,
    required=False,
    default=None,
    choices=valid_architectures, 
    help="Choose which UNet architecture to use."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=None,
        help="Number of epochs to train.",
    )
    parser.add_argument(
    "--learning-rate",
    type=float,
    required=False,
    default=None,
    help="Change learning rate parameter."
    )
    parser.add_argument(
    "--bcew",
    type=float,
    required=False,
    default=None,
    help="Change weighting for bce loss function, dice loss function weight will be changed accordingly s.t. bcew + dicew =1."
    )
    
    parser.add_argument(
    "--enhancement",
    type=str,
    required=False,
    default=None,
    choices=valid_enhancements,
    help="The kind of enhancement performed on the original dataset."
    )
    parser.add_argument(
    "--shuffle-data",
    nargs="?",
    const=True,
    default=None,
    type=str2bool,
    help="Shuffle training data. "
         "If flag is present without value -> True. "
         "Use --shuffle false to disable."
    )

    parser.add_argument(
        "--optimizer","-opt",
        type=str,
        choices=valid_optimizers,
        required=False,
        default=None,
        help="Optimizer choice.",
    )
    parser.add_argument(
    "--adam-beta1",
    type=float,
    required=False,
    default=None,
    help="Change Adam optimizer beta1 parameter."
)
    parser.add_argument(
    "--adam-beta2",
    type=float,
    required=False,
    default=None,
    help="Change Adam optimizer beta2 parameter."
)
   
    parser.add_argument(
    "--lrs-factor",
    type=float,
    required=False,
    default=None,
    help="Change learning rate scheduler factor parameter."
)
    parser.add_argument(
    "--lrs-patience",
    type=float,
    required=False,
    default=None,
    help="Change learning rate scheduler patience parameter."
)
    parser.add_argument(
    "--range-p",
    type=float,
    required=False,
    default=None,
    help="Change p value for range balancing function."
)
    parser.add_argument(
    "--range-eps",
    type=float,
    required=False,
    default=None,
    help="Change epsilon value for range balancing function."
)
    parser.add_argument(
    "--range-clip",
    type=float,
    required=False,
    default=None,
    help="Change epsilon value for range balancing function."
)
    parser.add_argument(
    "--reload",
    nargs="?",
    const=True,
    default=None,
    type=str2bool,
    help="reload data from specified epoch (if none is passed finds the most recent one). "
         "If flag is present without value -> True. "
         "Use --load false to disable."
    )
    
    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------

if __name__ == "__main__":
    main()

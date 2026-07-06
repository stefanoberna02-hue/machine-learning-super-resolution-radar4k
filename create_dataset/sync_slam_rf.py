# 
# ================================================================================
# SLAM-RF DATASET: STRUCTURAL CONSTRAINTS AND SYNCHRONIZATION LOGIC
# (Technical documentation for maintainers)
# ================================================================================

# This script resolves a non-trivial alignment problem between LiDAR frames and 
# Radar frames in the SLAM-RF dataset. The two modalities use incompatible naming 
# schemes, incompatible indexing conventions, and do not share the same notion of 
# “frame”. The goal is to reconstruct a one-to-one mapping:
#             (radar_frame_m)  -->  (lidar_frame_k)
# for all usable samples.

# Below is a precise description of the directory layout, the relevant file formats, 
# and the structural issues that must be handled explicitly in code.

# --------------------------------------------------------------------------------
# 1. DIRECTORY LAYOUT (subset relevant to synchronization)
# --------------------------------------------------------------------------------

# SLAM-RF/
# ├── lidar/
# │   ├── <day>/                         # e.g., 020925, 030925, 040925 …
# │   │   ├── <day>_exp1/
# │   │   │   ├── PointCloudFrame0.pcd/.ply
# │   │   │   ├── PointCloudFrame1.pcd/.ply
# │   │   │   └── ...
# │   │   ├── <day>_exp2/
# │   │   └── ...
# │   │
# │   │   ├── <day>_exp1_radar_synchronized.csv   # one per experiment
# │   │   ├── <day>_exp2_radar_synchronized.csv
# │   │   └── ...
# │
# └── radar/
#     ├── <day>/
#     │   ├── <day>_exp1/
#     │   │   ├── exp1_master_loop_1_frm_2_RA.npz
#     │   │   ├── exp1_master_loop_1_frm_3_RA.npz
#     │   │   └── ...
#     │   ├── <day>_exp2/
#     │   └── ...
#     └── ...

# Only the three elements below matter for alignment:
#     • LiDAR frame files: PointCloudFrame<i>.<ply/pcd>
#     • Radar RA files: <exp>_master_loop_L_frm_F_RA.npz
#     • Synchronization CSV: <day>_expX_radar_synchronized.csv

# --------------------------------------------------------------------------------
# 2. SYNCHRONIZATION CSV FORMAT (one row = one radar chirp)
# --------------------------------------------------------------------------------

# Each experiment has one synchronization file at:
#         lidar/<day>/<day>_expX_radar_synchronized.csv

# Minimal required columns (indices from dataset documentation):
#     [0] chirp_id           # global chirp index within the experiment
#     [1] radar_timestamp    # unused here
#     [2] lidar_frame_index  # index of the closest LiDAR PointCloudFrame<i>

# Example row:
#     chirp_id, timestamp, lidar_frame_index
#     64,       ...,        12

# Important: the CSV maps *chirps* to LiDAR frames. But radar files are not stored
# per chirp nor per timestamp — they are stored per:
#         (loop_id, frame_id)
# thus a reconstruction step is mandatory.

# --------------------------------------------------------------------------------
# 3. THE CORE INDEXING PROBLEM
# --------------------------------------------------------------------------------

# LiDAR:
#     Filenames contain the actual frame index:
#           PointCloudFrame<i>.pcd
#     → easy: lidar_frame_index = i

# Radar:
#     Filenames DO NOT contain chirp indices. Instead they use:
#           <exp>_master_loop_<L>_frm_<F>_RA.npz
#     where:
#         L = loop number (1,2,...)
#         F = frame number within loop

# However, the synchronization CSV never mentions L or F.  
# It only gives chirp_id.  

# Thus we must invert the mapping:
#         chirp_id  → (loop_id, frame_id)

# The authors specify two constants:

#     CHIRPS_PER_FRAME          = 32
#     MAX_RADAR_FRAMES_PER_LOOP = 340

# Given chirp_id c:
#     frame_count = c // 32
#     frame_id    = (frame_count % 340) + 2     # consistent with filenames
#     loop_id     = (frame_count // 340) + 1

# This exactly reproduces the radar filenames found in the dataset.

# This inversion step is essential: without it, radar → LiDAR matching is impossible.

# --------------------------------------------------------------------------------
# 4. CROSS-MODAL FRAME MATCHING: WHAT WE ASSUME
# --------------------------------------------------------------------------------

# Given one row in the synchronization CSV:
#         chirp_id = c
#         lidar_frame_index = k

# We associate:
#         radar_file = radar/<day>/<day>_expX/<exp>_master_loop_L_frm_F_RA.npz
#         lidar_file = lidar/<day>/<day>_expX/PointCloudFrame<k>.ply/.pcd

# Assumptions derived from original dataset authors:
#     • chirp_id is globally monotonic within each experiment.
#     • All radar frames can be reconstructed through the provided constants 
#       (CHIRPS_PER_FRAME, MAX_RADAR_FRAMES_PER_LOOP).
#     • The closest LiDAR frame has already been computed by the dataset authors 
#       and stored in the CSV; we do NOT recompute temporal offsets.
#     • Only the *first chirp of each radar frame* is used to avoid redundancy 
#       (i.e., the chirp_id values are always multiples of 32 in our loop).

# --------------------------------------------------------------------------------
# 5. WHY CONVERSION STEPS ARE REQUIRED (implementation-specific)
# --------------------------------------------------------------------------------

# Radar RA files contain:
#         RA_mc ∈ ℂ^(AZIM_BINS × CHIRPS_PER_FRAME × RANGE_BINS)
# One .npz file = one radar frame (not one chirp).  
# The RA tensor must be compressed to a single frame representation:
#         RA_mc[:,0,:]  # take chirp 0 of the frame (consistent across dataset)

# We then convert RA → point-cloud-like structure → polar raster:
#         RA domain → threshold → (x,y,intensity) → (r,θ,intensity)
#         [AZIM_BINS × RANGE_BINS]  matrix of complex intensities → [AZIM_BINS*RANGE_BINS, AZIM_BINS*RANGE_BINS, AZIM_BINS*RANGE_BINS] list of vectors with physical dimensions and associated real intensities → [AZIM_BINS, RANGE_BINS, AZIM_BINS*RANGE_BINS] list for final polar image creation
# because LiDAR images will also be represented in the same polar grid.

# LiDAR PLY/PCD files contain raw 3D points. For alignment:
#         • ignore z beyond a predefined slice
#         • restrict to radar FoV limits
#         • convert Cartesian → (r, θ)
#         • discretize to same polar grid as radar

# The resulting pair of images:
#         radar/<day>/<exp>_<chirp>.png
#         lidar/<day>/<exp>_<chirp>.png
# is geometrically comparable and indexed consistently.

# --------------------------------------------------------------------------------
# 6. SUMMARY OF THE REAL PROBLEMS SOLVED BY THIS SCRIPT
# --------------------------------------------------------------------------------

# ✓ Radar files use loop/frame indexing; synchronization uses chirp indexing.  
#   → Must reconstruct loop_id and frame_id from chirp_id.

# ✓ Synchronization is based on chirps, not radar frames.  
#   → Must downsample to the first chirp of each frame.

# ✓ Radar RA tensors contain multiple chirps per frame, but filenames represent 
#   only the frame.  
#   → Must extract a single representative chirp (chirp 0).

# ✓ LiDAR and Radar have incompatible coordinate systems and FoVs.  
#   → Must convert both to a shared polar grid (RBINS × ABINS_*).

# ✓ Directory structure places sync CSVs at <day>/, while frames are inside 
#   <day>_expX/.  
#   → Must dynamically resolve experiment numbers and construct file paths.

# This documentation is aligned with:
#     https://www.notion.so/SLAM-RF-dataset-description-2b57344603d980f0b21ad19fc1d22638

# ================================================================================
# 



# ------------------------------------------------------------------------------
# Command-line interface (CLI) usage examples
# ------------------------------------------------------------------------------
#
# Required argument
# -----------------------
# --folder / -fo
# Selects the user working directory. Must be one of:
#     {moh, ste, alex}
#
# Example:
#     python sync_slam_rf.py --folder ste
#
#
# Optional verbosity
# ------------------
# --verbose / -v
# Enables verbose logging during processing.
#
# Example:
#     python sync_slam_rf.py -fo ste --verbose
#
#
# Output directory override
# -------------------------
# --out-folder <name>
# Overrides the default output folder (processed_SLAM_12005).
# The folder is created relative to the project root.
#
# Example:
#     python sync_slam_rf.py -fo ste --out-folder processed_SLAM_debug
#
#
# Data augmentation / enhancement
# -------------------------------
# --enhancement {none, low, high, vhigh, vvhigh, vvvhigh}
#
# Controls image augmentation applied to both radar and LiDAR polar images:
#   - none     : no augmentation (single image)
#   - low      : flips only
#   - high     : shifts only
#   - vhigh    : shifts + flips
#   - vvhigh   : hardcoded shifts + flips
#   - vvvhigh  : stronger hardcoded shifts + flips
#
# Example:
#     python sync_slam_rf.py -fo ste --enhancement vhigh
#
#
# Radar intensity thresholding (magnitude-based)
# ----------------------------------------------
# --mag-threshold [VALUE]
#
# Enables magnitude-based thresholding for radar intensities.
# If VALUE is provided, it is used explicitly.
# If the flag is provided without VALUE, the internal default MAG_THRESHOLD
# defined in the script is used.
#
# Examples:
#     # Use internal default magnitude threshold
#     python sync_slam_rf.py -fo ste --mag-threshold
#
#     # Use a custom magnitude threshold
#     python sync_slam_rf.py -fo ste --mag-threshold 0.03
#
#
# Radar intensity thresholding (CFAR-based)
# ----------------------------------------
# --cfar-threshold [VALUE]
#
# Enables CFAR-based thresholding (range-only CFAR).
# Same semantics as --mag-threshold:
#   - no VALUE  -> use internal default CFAR_THRESHOLD
#   - VALUE     -> use provided CFAR threshold
#
# Note:
# If both --mag-threshold and --cfar-threshold are provided,
# magnitude thresholding takes precedence.
#
# Examples:
#     python sync_slam_rf.py -fo ste --cfar-threshold
#     python sync_slam_rf.py -fo ste --cfar-threshold 6.0
#
#
# Binary intensity mode
# ---------------------
# --binary-intensity
#
# When enabled, the final polar images store binary occupancy
# (a bin is set to 1 if at least one point falls into it).
# This affects the final rasterization step and does NOT disable
# radar thresholding itself.
#
# Example:
#     python sync_slam_rf.py -fo ste --binary-intensity
#
#
# Combined examples
# -----------------
# 1) CFAR thresholding + binary intensity + augmentation:
#     python sync_slam_rf.py \
#         -fo ste \
#         --cfar-threshold 5 \
#         --binary-intensity \
#         --enhancement high \
#         --out-folder processed_SLAM_cfar_bin
#
# 2) Magnitude thresholding with default value, verbose mode:
#     python sync_slam_rf.py \
#         -fo ste \
#         --mag-threshold \
#         --verbose
#
# ------------------------------------------------------------------------------

import sys
import os
import pandas as pd
import numpy as np
import glob as glob
from PIL import Image

from plyfile import PlyData

import argparse
import gc
from concurrent.futures import ProcessPoolExecutor
import time

# folder in the working derictory where the generated files will be saved
# naming convention is: preprocessing: 1-6 (none->vvvhigh), intensity handling: 1 or 0 for binary, followed by 1 for no thresholding, 2 for magnitude thresh and 3 for cfar, followed by the value used for thresholding if applicable
GENERATED_FOLDER = "processed_SLAM_112005"

scratch_path = "/mnt/course-cs-433-group03/scratch"

LIDAR_PATH = scratch_path + "/SLAM-RF/lidar"
RADAR_PATH = scratch_path + "/SLAM-RF/radar"

lidar_frame_name = "PointCloudFrame"
PLY = ".ply"
PCD = ".pcd"
NPZ = ".npz"

index_length = 6

# in our dataset the frames go from loop_1_frm_2 to loop_1_frm_341, so 340 frames before loop changes to loop_2_frm_2
max_radar_frames_per_loop = 340
radar_chirp_per_frame = 32

# CARTESIAN dimension, same as the ones used in the pcl data
X_MAX = 10
Y_MAX = 10
Z_MIN = -0.3
Z_MAX = 0.3

R_MAX = 10.8  # maximum range for both radar and lidar
A_MAX_RADAR = 90  # maximum aperture for radar, clip inside -A_MAX_RADAR < azimuth < A_MAX_RADAR
A_MAX_LIDAR = 90  # used to create the final image, there are ABINS_LIDAR pixels that go from -A_MAX_LIDAR to A_MAX_LIDAR
E_MAX_LIDAR = 30  # points outside the elevation range of -E_MAX_LIDAR < elevation < E_MAX_LIDAR are discarded

# final image resolution
RBINS = 256
ABINS_LIDAR = 512
ABINS_RADAR = 180
# grids used to save the final polar images for radar and lidar
# this is wha we WISH we had, and what will go to the model
# physical coordinates are 'snapped' to this grid by putting r_physical inside the first r_grid which is bigger, same for azimuth
R_GRID = np.linspace(0, R_MAX, RBINS)
A_GRID_RADAR = np.linspace(-A_MAX_RADAR, A_MAX_RADAR, ABINS_RADAR)
A_GRID_LIDAR = np.linspace(-A_MAX_LIDAR, A_MAX_LIDAR, ABINS_LIDAR)

# values regarding raw data for radar
RANGE_FFT = 512
AZIM_FFT = 180
MAX_RANGE = 21.59

#conventions used in naming folders reagrding data, from now on will follow the convention:
# <stage>_SLAM_<specifier>
# -stage is one of {processed, dataset} 
# -specifier is: xymzzz, x:1-6 enhancement level, y:0/1  binary intensity , m:1-3 thresholding method (1 no, 2 mag thresh, 3 cfar) zzz: threshold value used (000 if none)
MAG_THRESHOLD = 0.05
RANGE_GUARD = 5
CFAR_THRESHOLD = 5
METHOD = 'mag'
BINARY_INTENSITY = False

# values used topass from FOURIER domain to PHYSICAL domain x,y,z
# used inside of threshold() function
# these are the physical values corresponding to one a_bin and r_bin as found in our starting object RA_mc
# Note that in our case a_bin is NOT in the Fourier domain but is already an angle since we are using beamforming!
# this is why the spacing is linear in radiant angles and not in sin(theta) as it usually is for Fourier domain
theta = np.linspace(-np.pi / 2, np.pi / 2, AZIM_FFT)  # we are translating original data where theta goes from 0 to pi
sin_theta = np.sin(theta)  # goes from -1 to 1 in this order
cos_theta = np.cos(theta)  # goes from 0 to 1 in this order
range_d, sine_theta_mat = np.meshgrid(np.linspace(0, MAX_RANGE, RANGE_FFT), sin_theta)
_, cos_theta_mat = np.meshgrid(np.linspace(0, MAX_RANGE, RANGE_FFT), cos_theta)
x_axis = np.multiply(range_d, cos_theta_mat)
y_axis = np.multiply(range_d, sine_theta_mat)

# used when enhancing images, controls the parameters for the "high" mode of enhancement, other modes are hardcoded
ENHANCEMENT = "none"
NUM_SHIFTS = 2*2 #this is the total number of shifts we do on the azimuth axis after shifting ones on the range axis by SHIFT_Y, the azimuth shifts are performed half on the left and half on the right, all by the same amount of SHIFT_X pixels
SHIFT_X = 10 #shifts in azimuth bins, we alway do one shift to the right and one to the left
SHIFT_Y = 25 #shifts in range bins going further away

#the levels of preprocessing result in a multiplier of:
# -none=1
# -low=4
# -high=5 
# -vhigh=3*5
# -vvhigh=(6+1)*4
# -vvvhigh=(8+1)*4
valid_enhancements=["none","low","high","vhigh","vvhigh","vvvhigh"]
#Important: the values for high and vhigh will be wrong if the user sets the shifts manually in the sync_slam code
augmentation_map = {
    "none": 1,
    "low": 4, 
    "high": 5,
    "vhigh": 20,
    "vvhigh": 28, 
    "vvvhigh": 36,
   }

def generate_flips(img):
    """Return [original, flip_lr, flip_ud, flip_both]."""
    return [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.ROTATE_180)
    ]

def generate_shift_values(shift_x, shift_y, num_shifts):
    """
    Generate shifts as:
      - original (0, 0)
      - for each y-level j (there are num_shifts/2 levels):
          (+shift_x, j*shift_y)
          (-shift_x, j*shift_y)
          in the eng you will get num_shifts shifts + original

    Requires num_shifts to be even.
    """

    shifts = [(0, 0)]  # original

    half = num_shifts // 2
    for j in range(1, half + 1):
        dy = j * shift_y
        shifts.append((+shift_x, dy))
        shifts.append((-shift_x, dy))

    return shifts

# note that dx and dy represent the number of pixels we want to shift
def shift_image(img, dx, dy):
    """
    Shift dx right, dy up. Missing regions filled with black.
    dx > 0 => shift right
    dy > 0 => shift up
    """
    arr = np.array(img)
    H, W = arr.shape

    shifted = np.zeros_like(arr)

    x_from = max(0, dx)
    x_to = W
    x_src = 0
    x_src_end = W - dx
    if dx < 0:  # shift left
        x_from = 0
        x_to = W + dx
        x_src = -dx
        x_src_end = W

    y_from = max(0, dy)
    y_to = H
    y_src = 0
    y_src_end = H - dy
    if dy < 0:  # shift down
        y_from = 0
        y_to = H + dy
        y_src = -dy
        y_src_end = H

    # Check
    if x_src_end <= 0 or y_src_end <= 0:
        return Image.fromarray(shifted)

    shifted[y_from:y_to, x_from:x_to] = arr[y_src:y_src_end, x_src:x_src_end]
    return Image.fromarray(shifted)


# x and y are the dimensions of the image so azimuth and radius respectively
def enhance_image(img, enhancement, shift_x, shift_y, num_shifts):
    """Return a list of augmented images according to enhancement level."""

    if enhancement == "none":
        return [img]

    # LOW → original + 3 flip
    if enhancement == "low":
        return generate_flips(img)

    # HIGH → original + shifts (as set at the begginning of this script)
    if enhancement == "high":
        result = []
        #check genreate shift values function for specific method of shifting
        shift_values = generate_shift_values(shift_x, shift_y, num_shifts)        # k=0 includes original image

        for dx, dy in shift_values:
            shifted = shift_image(img, dx, dy)
            result.append(shifted)
        return result
    # VHIGH → original + shifting + flipping for every shift and original (shifting as set at the beginning this script)
    if enhancement == "vhigh":
        result = []

        #check genreate shift values function for specific method of shifting
        shift_values = generate_shift_values(shift_x, shift_y, num_shifts)        # k=0 includes original image
        # k=0 includes original image

        for dx, dy in shift_values:
            shifted = shift_image(img, dx, dy)
            flips = generate_flips(shifted)#multiplies everything by 4
            result.extend(flips)
        return result
    # VVHIGH → original + shifting + flipping for every shift and original (shifting as hardcoded below)
    if enhancement == "vvhigh":
        num_shifts = 6
        shift_x = 10
        shift_y = 20
        result = []

        #check genreate shift values function for specific method of shifting
        shift_values = generate_shift_values(shift_x, shift_y, num_shifts)        # k=0 includes original image

        for dx, dy in shift_values:
            shifted = shift_image(img, dx, dy)
            flips = generate_flips(shifted) #multiplies everything by 4
            result.extend(flips)
        return result
    # VVVHIGH → original + shifting + flipping for every shifting (shifting as hardcoded below)
    if enhancement == "vvvhigh":
        num_shifts = 8
        shift_x = 15
        shift_y = 15
        result = []

#check genreate shift values function for specific method of shifting
        shift_values = generate_shift_values(shift_x, shift_y, num_shifts)        # k=0 includes original image

        for dx, dy in shift_values:
            shifted = shift_image(img, dx, dy)
            flips = generate_flips(shifted)
            result.extend(flips)

        return result


def list_sync_files(index):
    """
    returns the names of all the synchronized csv files in the directory index.

    Parameters:
        index (str): number corresponding to the directory name.
    Returns:
        list: list containing the names of the synchronzed csv files.
    """

    index = str(index)
    target_path = LIDAR_PATH + "/" + index

    csv_list = []

    with os.scandir(target_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith(index + "_exp") and entry.name.endswith(
                    "radar_synchronized.csv"):
                csv_list.append(target_path + "/" + entry.name)

    return sorted(csv_list)


def get_matching_frames(sync_files, index):
    """
    Given the paths to the radar_synchronized.csv files, returns a list of tuples that match a radar frame to the closest lidar frame.
    All the radar_synchronized.csv files must belong to the same index.

    Parameters:
        sync_files : list of paths to the radar_synchronized.csv files.
        index: index to which all the sync_files belong.
    Returns:
        list of tuples, each one containing in order: a radar frame, its closest
        lidar frame, the index and experiment number they both belong to and the radar chirp.
    """

    for sync_file in sync_files:
        if not sync_file.split("/")[-1].startswith(index):
            raise Exception("sync files must have same given index.")

    synced_frames = []
    for sync_file in sync_files:
        exp_num = sync_file.split("/")[-1].split("_")
        # exp_num = "exp[index]"
        exp_num = exp_num[1]
        data = np.genfromtxt(sync_file, delimiter=',', skip_header=1)
        total_rows = data.shape[0]

        for i in range(0, total_rows, radar_chirp_per_frame):
            row = data[i]
            lidar_frame_index = int(row[2])
            radar_chirp = int(row[0])

            lidar_file_path = f"{LIDAR_PATH}/{index}/{index}_{exp_num}/{lidar_frame_name}{lidar_frame_index}"

            radar_frame_index = ((radar_chirp // 32) % max_radar_frames_per_loop) + 2
            radar_frame_loop_index = ((radar_chirp // 32) // max_radar_frames_per_loop) + 1

            radar_frame_name = f"{exp_num}_master_loop_{radar_frame_loop_index}_frm_{radar_frame_index}_RA{NPZ}"
            radar_file_path = f"{RADAR_PATH}/{index}/{index}_{exp_num}/{radar_frame_name}"

            synced_frames.append((radar_file_path, lidar_file_path, index, exp_num, radar_chirp))

    return synced_frames


# takes as input a list with 4 vectors x y z and real valued intensities, ideally they are of shape A_BINS*R_BINS
# but after thresholding for radar, it is possible that some entries are dropped so the dimensions might be smaller
def pcl_to_polar(pcl_data):
    # Filter pcl data based on x y z values
    x, y, z = pcl_data[:, 0], pcl_data[:, 1], pcl_data[:, 2]
    mask = ((x > 0) & ((x <= X_MAX)
                       & ((z >= Z_MIN) & ((z <= Z_MAX) &
                                          ((y >= -Y_MAX) & (y <= Y_MAX))))))
    pcl_data = pcl_data[mask, :]

    x, y, z = pcl_data[:, 0], pcl_data[:, 1], pcl_data[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    a = np.rad2deg(np.arctan2(y, x))
    e = np.rad2deg(np.arcsin(z / r))
    polar_data = np.column_stack([r, a, e, pcl_data[:, 3]])

    return polar_data


def create_image_polar(polar_data, is_lidar, is_binary):
    r = polar_data[:, 0]
    a = polar_data[:, 1]
    e = polar_data[:, 2] if is_lidar else None

    mask = ((r > 0) & (r <= R_MAX))

    if not is_lidar:
        mask &= np.abs(a) <= A_MAX_RADAR
    else:
        mask &= np.abs(a) <= A_MAX_LIDAR
        mask &= np.abs(e) <= E_MAX_LIDAR

    polar_data = polar_data[mask, :]
    r = polar_data[:, 0]
    a = polar_data[:, 1]

    # if is_lidar:
    #     e = polar_data[:, 2]
    #     polar_data[:,0] = r*np.cos(np.deg2rad(e))#taking the 2D radius by projecting elevation on the ground plane
    #     r = polar_data[:,0]

    intensity = polar_data[:, 3]

    # creating image with correct dimensions
    if is_lidar:
        image = np.zeros((RBINS, ABINS_LIDAR))
        dr = R_GRID[1] - R_GRID[0]
        da = A_GRID_LIDAR[1] - A_GRID_LIDAR[0]
        a_min = A_GRID_LIDAR[0]


    else:
        image = np.zeros((RBINS, ABINS_RADAR))
        epsilon = 1e-6

        intensity = np.maximum(intensity, epsilon)
        intensity = 10 * np.log10(intensity)
        dr = R_GRID[1] - R_GRID[0]
        da = A_GRID_RADAR[1] - A_GRID_RADAR[0]
        a_min = A_GRID_RADAR[0]

    if intensity.size != 0:
        min_intensity = np.min(intensity)
        max_intensity = np.max(intensity)

        if not is_lidar:
            den = max_intensity - min_intensity
            if den < 1e-6:
                intensity[:] = 0.0
            else:
                intensity = (intensity - min_intensity) / den

        else:
            intensity = (intensity - min_intensity) / (max_intensity - min_intensity)

        # forcing radius and azimuth to allign with our predetermined grids
        # we are going from physical dimensions r = [m] a = [deg] to indices for R_GRID and A_GRID
        r_index = np.floor(r / dr).astype(int)
        a_index = np.floor((a - a_min) / da).astype(int)

        r_index = np.clip(r_index, 0, RBINS - 1)
        a_index = np.clip(a_index, 0, (ABINS_LIDAR if is_lidar else ABINS_RADAR) - 1)

        # compute flattened indices, row-major order: one index per polar point in the grid, instead of two.
        flat_indices = r_index * image.shape[1] + a_index

        if is_binary:
            # set all bins that contain at least one point
            image.ravel()[flat_indices] = 1
        else:
            # take the maximum intensity per bin
            # For each point k:
            #   - flat_indices[k] identifies the bin (r_bin, a_bin) in the flattened image
            #   - np.maximum.at(...) updates that bin using:
            #         image_flat[idx] = max(image_flat[idx], intensity[k])
            # This accumulates all points falling into the same polar bin
            # by keeping the maximum intensity for that bin.
            np.maximum.at(image.ravel(), flat_indices, intensity)

        if is_binary:
            image = image.astype(np.bool_)

    return image


#  This can be CFAR instead
# takes as input a matrix of real valued intensities with dimsensions [A_BINS,R_BINS] (it should be RA_mc from our SLAM data)
# returns as output a list with fielsd 'x','y','z','intensity' all with the same size A_BINS_RADAR*R_BINS
# meaning that intensity is flattended, at each index you see intensity[i] from phyisical reflector at x[i],y[i],z[i]
def threshold(frame_fft):
    if METHOD == 'mag':
        # Magnitude only thresholding
        m = np.max(
            frame_fft[:, 6:])  # removing the first 6 bins for range because they are polluted by strong reflections
        idx = (frame_fft[:, 6:] >= MAG_THRESHOLD * m)
        idx = np.concatenate((np.zeros((AZIM_FFT, 6), dtype=bool), idx), axis=1)
    elif METHOD == 'cfar':
        # CFAR range only thresholding
        idx = np.zeros((AZIM_FFT, RANGE_FFT), dtype=bool)
        frame_fft = 10 * np.log10(np.maximum(frame_fft, 1e-12))
        for i in range(10, RANGE_FFT - RANGE_GUARD):
            cut = frame_fft[:, i]
            guard = np.sum(frame_fft[:, i - RANGE_GUARD:i + RANGE_GUARD + 1], axis=1)
            guard = (guard - cut) / (2 * RANGE_GUARD)
            idx[:, i] = ((cut - guard) > CFAR_THRESHOLD)
    elif METHOD == 'no':
        idx = np.concatenate((np.zeros((AZIM_FFT, 6), dtype=bool), np.ones((AZIM_FFT, RANGE_FFT - 6), dtype=bool)),
                             axis=1)
    

    x = x_axis[idx].reshape(-1, 1)  # x axis and y axis are matrices resulting from a meshgrid
    y = y_axis[idx].reshape(-1,
                            1)  # they representa physical coordinates associated with each [i,j] entry of the raw data matrix frame_fft (for us RA_mc)
    z = np.zeros(x.shape)
    intensity = frame_fft[idx].reshape(-1, 1)

    frame_pcl = np.concatenate((x, y, z, intensity), axis=1)

    return frame_pcl


# assuming we are only getting the matrix "RA_mc" = [180,512] as input ( it comes from from the list containing also "theta" = [,180] , "dist" = [512]  )
# where theta and dist map the entries to physical dimensions of angle and range, and RA_mc contains complex values for each intensity reflected by (angle, range) bin
def convert_radar_to_pcl(frame):
    frame_fft = np.abs(frame)  # taking the intensity of the complex values
    frame_pcl = threshold(frame_fft)

    return frame_pcl


def get_radar_polar_img_from_frame(frame_path, chirp_num):
    file_name = frame_path.split("/")[-1].split(".")[0]

    with np.load(frame_path) as data:
        # for key in data.files:
        #     array = data[key]
        #     print(f"{key}: shape={array.shape}, dtype={array.dtype}")
        frame = data["RA_mc"][:, chirp_num, :]

        # curr_radar_data = convert_radar_to_pcl(frame)
        curr_radar_data = convert_radar_to_pcl(frame)
        polar_radar_data = pcl_to_polar(curr_radar_data)

        radar_img = create_image_polar(polar_radar_data, is_lidar=False, is_binary=False)
        im = Image.fromarray((radar_img * 255).astype(np.uint8))
        del frame, curr_radar_data, polar_radar_data, radar_img
        return im


# applyied changes to flip y-axis since lidar .pcl use only negative values for y and rotate 90 degree clockwise since original reserchears work with x axis clipped from 0 to R_MAX
def get_lidar_polar_img_from_frame(file_path):
    def read_pcd_with_pandas(pcd_path):
        with open(pcd_path, 'r') as f:
            lines = f.readlines()

        data_start = 0
        fields = []
        for i, line in enumerate(lines):
            if line.startswith('FIELDS'):
                fields = line.strip().split()[1:]
            elif line.startswith('DATA ascii'):
                data_start = i + 1
                break

        if data_start == 0:
            raise ValueError("Could not find data section or not ASCII format")

        df = pd.read_csv(
            pcd_path,
            skiprows=data_start,
            sep='\s+',
            header=None,
            names=fields
        )
        return df

    lidar_file_path_ply = file_path + PLY
    lidar_file_path_pcd = file_path + PCD

    if os.path.exists(lidar_file_path_pcd):
        curr_lidar_data = read_pcd_with_pandas(lidar_file_path_pcd).to_numpy()
        curr_lidar_data = curr_lidar_data[:, (0, 1, 2, 5)]
    else:
        curr_lidar_data = PlyData.read(lidar_file_path_ply)
        curr_lidar_data = np.asarray(curr_lidar_data.elements[0].data[:, (0, 1, 2, 5)])

    old_x = curr_lidar_data[:, 0].copy()
    old_y = -curr_lidar_data[:, 1].copy()  # flip the y axis becaus we are given only negative values

    curr_lidar_data[:, 0] = old_y  # rotating 90° clockwise because of our lidar convention
    curr_lidar_data[:, 1] = -old_x

    polar_lidar_data = pcl_to_polar(curr_lidar_data)
    lidar_img = create_image_polar(polar_lidar_data, is_lidar=True, is_binary=BINARY_INTENSITY)

    file_name = file_path.split("/")[-1].split('.')[0] + '.png'

    im = Image.fromarray((lidar_img * 255).astype(np.uint8))
    del curr_lidar_data, polar_lidar_data, lidar_img
    return im


def parallel_process_index(index_list, storage_path, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(generate_processed_lidar_radar_images, index_list, [storage_path] * len(index_list)))


def parallel_process_frame(index_list, storage_path, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(generate_processed_lidar_radar_image, index_list, [storage_path] * len(index_list)))


def generate_processed_lidar_radar_image(matching_frame, storage_path, verbose=False):
    for chirp_num in range(0, 2):
        radar_frame = matching_frame[0]
        lidar_frame = matching_frame[1]
        index = matching_frame[2]  # day of the experiment e.g. 020925
        exp = matching_frame[3]  # experiment number e.g. exp1
        chirp_idx = int(matching_frame[
                            4])  # radar chirp number, note that we are taking the first chirp per frame, so these numbers are all multiples of 32 starting from 0

        exist_radar_path = os.path.exists(radar_frame)
        exist_lidar_path = os.path.exists(lidar_frame + PLY) or os.path.exists(lidar_frame + PCD)
        if (not exist_radar_path and not exist_lidar_path):
            print(f"This radar-lidar couple of matching frames doesn not exist (skipping): {matching_frame} ")
            continue
        if not exist_radar_path:
            print(f"Radar frame of this couple does not exist (skipping): {matching_frame} ")
            continue
        if not exist_lidar_path:
            print(f"Lidar frame of this couple does not exist (skipping): {matching_frame} ")
            continue

        chirp_idx_in_frame = int(radar_chirp_per_frame / 4) + int(radar_chirp_per_frame / 2) * chirp_num  # we are taking either the 8th or the 24th chirp in the frame

        original_radar_img = get_radar_polar_img_from_frame(radar_frame, chirp_idx_in_frame)
        original_lidar_img = get_lidar_polar_img_from_frame(lidar_frame)

        radar_imgs = []
        lidar_imgs = []
        if not ((ENHANCEMENT == 'none') or (ENHANCEMENT is None)):
            radar_imgs = enhance_image(original_radar_img, ENHANCEMENT, SHIFT_X, SHIFT_Y, NUM_SHIFTS)
            lidar_imgs = enhance_image(original_lidar_img, ENHANCEMENT, SHIFT_X, SHIFT_Y, NUM_SHIFTS)
            enhancement_numbers = len(radar_imgs)
        else:
            enhancement_numbers = 1 # only original image

        img_idx = int(chirp_idx / 32) * 2 * enhancement_numbers + chirp_num * enhancement_numbers

        original_radar_image_path = f"{storage_path}/radar/{index}/{exp}_{img_idx}.png"
        original_lidar_image_path = f"{storage_path}/lidar/{index}/{exp}_{img_idx}.png"

        radar_directory = os.path.dirname(original_radar_image_path)
        lidar_directory = os.path.dirname(original_lidar_image_path)

        # Create directories if they don't exist
        if radar_directory and not os.path.exists(radar_directory):
            os.makedirs(radar_directory, exist_ok=True)
        if lidar_directory and not os.path.exists(lidar_directory):
            os.makedirs(lidar_directory, exist_ok=True)

        if (ENHANCEMENT == 'none') or (ENHANCEMENT is None):
            original_radar_img.save(f"{original_radar_image_path}")
            original_lidar_img.save(f"{original_lidar_image_path}")
            if (verbose):
                print(f"Saved radar image to: {original_radar_image_path}")
                print(f"Saved lidar image to: {original_lidar_image_path}")

            del original_radar_img, original_lidar_img
            gc.collect()


        # save enhanced images with continuous numbering
        for enhanced_radar_img, enhanced_lidar_img in zip(
                radar_imgs,
                lidar_imgs
        ):
            img_idx += 1  # advance global counter

            radar_image_path_n = f"{storage_path}/radar/{index}/{exp}_{img_idx}.png"
            lidar_image_path_n = f"{storage_path}/lidar/{index}/{exp}_{img_idx}.png"

            enhanced_radar_img.save(radar_image_path_n)
            enhanced_lidar_img.save(lidar_image_path_n)

            if verbose:
                print(f"Saved radar image: {radar_image_path_n}")
                print(f"Saved lidar image: {lidar_image_path_n}")

            del enhanced_radar_img, enhanced_lidar_img

        del radar_imgs, lidar_imgs
        gc.collect()


def generate_processed_lidar_radar_images(matching_frames, storage_path, verbose=False):
    # this is continuous and loops back to zero after each experiment and day
    # incremental_idx = 0 index used to name the saved images. e.g.  storagepath/lidar/020925/exp1/{incremental_idx}
    previous_day = "000000"
    previous_exp = "exp-1"
    incremental_idx = 0
    parallel_process_frame(matching_frames, storage_path, 6)


def main():
    args = parse_cli()
    user = args.folder
    verbose = args.verbose

    # --------------------------------------------------
    # CHECK WORKING DIRECTORY
    # --------------------------------------------------

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level above create_dataset/
    SLAM_RF_DIR = os.path.dirname(PROJECT_ROOT)  # one level above <user>/

    expected_suffix = os.path.join(user, "project-2-radar4k", "create_dataset")

    if not SCRIPT_DIR.endswith(expected_suffix):
        raise ValueError(
            f"Please run the script from inside the '{expected_suffix}' folder.\n"
            f"Current directory: {SCRIPT_DIR}"
        )
    # --------------------------------------------------

    # assuming sync_slam is saved in <user>/radarhd/creasync_slam_rf.py
    # assuming SLAM-RF raw dataset is saved at the same level as <user>
    # All paths resolved relative to the repo structure

    # override generated folder
    global GENERATED_FOLDER
    if args.out_folder is not None:
        GENERATED_FOLDER = args.out_folder

    # override enhancement
    global ENHANCEMENT
    if args.enhancement is not None:
        ENHANCEMENT = args.enhancement
    # Folder where processed images will be saved

    #override magnitude_thresholding value if one is provided and do the same for cfar thresholding, when both are specified go with magnitude
    global METHOD, MAG_THRESHOLD, CFAR_THRESHOLD

    if args.mag_threshold is not None:
        MAG_THRESHOLD = args.mag_threshold
        METHOD = "mag"

    elif args.cfar_threshold is not None:
        CFAR_THRESHOLD = args.cfar_threshold
        METHOD = "cfar"
    
    global BINARY_INTENSITY #when provided, threshold values are still applied but final images are binary occupancy
    if args.binary_intensity:
        BINARY_INTENSITY = args.binary_intensity

    
    if args.no_threshold:
        METHOD = 'no'
     
    STORAGE_DIR = os.path.join(PROJECT_ROOT, GENERATED_FOLDER)

    print(f"-------------------------------------- sync_slam_rf.py --------------------------------------")
    print(f"Processing ALL days inside {user}'s working directory")
    print(f"Saving results inside: {GENERATED_FOLDER}")

    start = time.time()
    matching_frames = []
    # Iterate over all day folders in SLAM-RF/lidar/
    with os.scandir(LIDAR_PATH) as entries:
        for entry in entries:
            if entry.is_dir() and len(entry.name) == index_length:
                index = entry.name
                print(f"\n=== Processing day: {index} ===")

                sync_files = list_sync_files(index)
                if len(sync_files) == 0:
                    print(f"No synchronization CSV found for day {index} → skipping.")
                    continue

                matching_frames_per_index = get_matching_frames(sync_files,
                                                                index)  # one single call to get_matching_frames generates one continuous indexing of images, it should be called on one day at a time
                matching_frames.append(matching_frames_per_index)
                # print(" First 5 matches:", matching_frames[:5])

        parallel_process_index(matching_frames, STORAGE_DIR, 6)
        # generate_processed_lidar_radar_images(
        #     matching_frames,
        #     storage_path=STORAGE_DIR,
        #     verbose=verbose
        # )

    print("\nAll days processed.")
    end = time.time()
    print(f"time.time(): {end - start:.6f}s")




# -------------------------------------------------------
# Inline argument parser to select folder
# -------------------------------------------------------

valid_folders = {"moh", "ste", "alex"}


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Select user folder (moh, ste, alex)."
    )
    parser.add_argument(
        "-fo", "--folder",
        choices=valid_folders,
        required=True,
        help="Folder name (must be one of: moh, ste, alex)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    # enhancement level for image flipping and shifting
    parser.add_argument(
        "--enhancement",
        choices=valid_enhancements,
        default=None,
        help="Enhancement level: none (default), low (4 images), high ( shifts ), vhigh ( shifts + flips ), vvhigh ( hardcoded shifts + flips), vvvhigh ( hardcoded shifts + flips)."
    )
    parser.add_argument(
        "--out-folder",
        type=str,
        default=None,
        help="Optional output folder name. If not provided, the default internal folder is used."
    )
    parser.add_argument(
    "--binary-intensity",
    action="store_true",
    required=False,
    help="Use binary intensity (still paired with magnitude/CFAR thresholding)."
)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
    "--mag-threshold",
    type=float,
    nargs="?",
    const=MAG_THRESHOLD,
    default=None,
    help="Magnitude threshold (default: internal value)."
    )

    group.add_argument(
    "--cfar-threshold",
    type=float,
    nargs="?",
    const=CFAR_THRESHOLD,
    default=None,
    help="CFAR threshold (default: internal value)."
    )
    group.add_argument(
    "--no-threshold",
    action="store_true",
    required=False,
    help="Use no thresholding (paired with either binary or continuous intensity)."
)
    
    

    return parser.parse_args()


# ------------------------------------------------------


if __name__ == "__main__":
    main()
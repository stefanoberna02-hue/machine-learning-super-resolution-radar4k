# Converts polar test images into Cartesian 2D occupancy maps
# and arranges output in RadarHD-style folders for MATLAB visualization.

import os
import glob
import numpy as np
from PIL import Image

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

params = {
    'model_name': '13',
    'expt': 1,
    'dt': '20251211-190417',
    'epoch_num': 120,
}

RMAX = 10.8
RBINS = 256
ABINS = 512

# ---------------------------------------------------------------
# POLAR → CARTESIAN CONVERSION
# ---------------------------------------------------------------

def convert_pol2cart(a):
    b = np.zeros((RBINS, ABINS))

    loc = np.argwhere(a > 0)
    xloc = loc[:, 0]     # r index
    yloc = loc[:, 1]     # theta index

    x = x_axis[xloc, yloc]
    y = y_axis[xloc, yloc]

    new_xloc = np.searchsorted(x_axis_grid, x)
    new_yloc = np.searchsorted(y_axis_grid, y)

    new_xloc = np.clip(new_xloc, 0, RBINS - 1)
    new_yloc = np.clip(new_yloc, 0, ABINS - 1)

    b[new_xloc, new_yloc] = a[xloc, yloc]
    return b


# ---------------------------------------------------------------
# INPUT DIRS
# ---------------------------------------------------------------

name_str = params['model_name'] + '_' + str(params['expt']) + '_' + params['dt']

img_dir = f"../logs/{name_str}/test_imgs/"
save_path = f"./processed_imgs_{name_str}_test_imgs/"

os.system(f"rm -rf {save_path}")
os.makedirs(save_path, exist_ok=True)


# ---------------------------------------------------------------
# GRID DEFINITION
# ---------------------------------------------------------------

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


# ---------------------------------------------------------------
# PROCESS ALL PRED + LABEL PNG FILES
# ---------------------------------------------------------------

files = sorted(glob.glob(img_dir + "*.png"))

for file in files:
    # Example filename:
    #   040_100925_exp1_3_pred.png
    name = os.path.basename(file).replace(".png", "")
    tokens = name.split("_")

    epoch = tokens[0]
    day   = tokens[1]
    exp   = tokens[2]   # ex: exp1
    idx   = tokens[3]   # frame index
    kind  = tokens[4]   # pred or label

    traj_folder = f"{day}_{exp}"

    # Load polar image
    a = np.asarray(Image.open(file))
    a = convert_pol2cart(a).astype(np.uint8)
    a = Image.fromarray(a)

    # Output directory structure:
    # processed_imgs_xxx/day_exp/epoch/{pred,label}/
    outdir = f"{save_path}/{traj_folder}/{epoch}/{kind}/"
    os.makedirs(outdir, exist_ok=True)

    save_name = f"{epoch}_{day}_{exp}_{idx}_{kind}.png"
    a.save(os.path.join(outdir, save_name))


print("Conversion completed.")
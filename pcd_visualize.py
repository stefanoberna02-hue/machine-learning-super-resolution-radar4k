#This function allows to plot and visualize (only locally) lidar pointclouds from .pcd files
#note that when working on the cluster you necessarily have to save the output and visualize later if you want the interactive plot
#Visualization options:
# 1) --save gif , Save rotating GIF
# 2) --save int , interactive HTML plot
# 3) --save multi , Save multiple static views

#example usage:
#pcd_visualize.py --save -i 050925 -f 1

import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")  # Safe backend for SSH or headless environments
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import imageio.v2 as imageio
import io




# -------------------------------------------------------------------------
# Fixed paths for the SLAM-RF dataset on the EPFL cluster
# -------------------------------------------------------------------------
#example path: "/mnt/course-cs-433-group03/scratch/SLAM-RF/lidar/020925/020925_exp1/PointCloudFrame0.pcd"

SCRATCH = "/mnt/course-cs-433-group03/scratch"
LIDAR_ROOT = f"{SCRATCH}/SLAM-RF/lidar"
FRAME_PREFIX = "PointCloudFrame"
DUMP_DIR = "imgs"
os.makedirs(DUMP_DIR, exist_ok=True)


def voxel_downsample(points, voxel_size=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_ds.points)

def save_rotation_gif(fig, ax, out_file="rotation.gif", frames=30):
    out_path = os.path.join(DUMP_DIR, out_file)
    imgs = []
    for azim in np.linspace(0, 360, frames):
        ax.view_init(elev=30, azim=azim)

        # salva frame in buffer rapido
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=50)
        buf.seek(0)
        img = imageio.imread(buf)
        imgs.append(img)

    imageio.mimsave(out_path, imgs, fps=15)
    print(f"Saved rotating GIF: {out_path}")



def save_interactive_plot(points, out_file="interactive.html"):
    out_path = os.path.join(DUMP_DIR, out_file)
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode="markers",
        marker=dict(size=2, opacity=0.7),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        width=900,
        height=700,
        uirevision=True,      
    )

    # Renderer più stabile in VS Code webview
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive HTML: {out_path}")

def save_multiview(fig, ax, points, index, frame, exp):
    """
    Saves multiple 3D views of a point cloud into a specified directory.
    """
    out_dir = os.path.join(DUMP_DIR, f"{index}_{exp}_frame{frame}")
    os.makedirs(out_dir, exist_ok=True)

    # List of (elev, azim, filename suffix)
    views = [
        (10,   85,  "front"),
        (30,   85,  "front2"),
        #(80,   85,  "front3"),
        #(0,  -15,  "sideR"),
        #(0, 105,  "back"),
        #(0,  195,  "sideL"),
        #(-45,  135,  "isometric"),
        (45,  135,  "isometric2"),
        (90,   -90,  "top"),
        
    ]

    for elev, azim, name in views:
        ax.view_init(elev=elev, azim=azim)
        filepath = f"{out_dir}/{index}_{exp}_{frame}_{name}.png"
        fig.savefig(filepath, dpi=200)
        print(f"Saved: {filepath}")

# -------------------------------------------------------------------------
# Utility: load .pcd or .ply automatically
# -------------------------------------------------------------------------
def load_pointcloud(base_path):
    """
    Attempts to load <base_path>.pcd first, then <base_path>.ply.
    Returns Nx3 numpy array of XYZ points.
    """
    pcd_path = base_path + ".pcd"
    ply_path = base_path + ".ply"

    if os.path.exists(pcd_path):
        print(f"Loading PCD: {pcd_path}")
        pc = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pc.points)

    if os.path.exists(ply_path):
        print(f"PCD not found, loading PLY: {ply_path}")
        ply = o3d.io.read_point_cloud(ply_path)
        return np.asarray(ply.points)

    raise FileNotFoundError(
        f"Neither {pcd_path} nor {ply_path} exists."
    )


# -------------------------------------------------------------------------
# Main visualization function
# -------------------------------------------------------------------------
def visualize_pointcloud(index, frame, exp, show, save_mode=None):
    """
    Visualizes one lidar frame as a raw 3D scatter.
    No axis normalization, no centering, no scaling.
    """

    # Build correct path 
    base_path = f"{LIDAR_ROOT}/{index}/{index}_{exp}/{FRAME_PREFIX}{frame}"

    points = load_pointcloud(base_path)
    points = points[abs(points[:, 2])<0.3]  # Discard intensity if present
    print(f"Loaded {points.shape[0]} points.")
    print(f"Min XYZ: {points.min(axis=0)}")
    print(f"Max XYZ: {points.max(axis=0)}")

    # ----- 3D scatter, raw coordinates, no adjustments -----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"{index} - frame {frame}" + (f" ({exp})" if exp else ""))

        
    # ---- Set axis limits based on point cloud ----
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)

    # Optional 5% padding for readability
    px = 0.05 * (xmax - xmin)
    py = 0.05 * (ymax - ymin)
    pz = 0.05 * (zmax - zmin)

    ax.set_xlim(xmin - px, xmax + px)
    ax.set_ylim(ymin - py, ymax + py)
    ax.set_zlim(zmin - pz, zmax + pz)

    
    if save_mode == "gif":
        pc = voxel_downsample(points, voxel_size=0.15)
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=0.5)
        save_rotation_gif(fig, ax, f"{index}_{exp}_frame{frame}_rotate.gif")

    elif save_mode == "int":
        pc = voxel_downsample(points, voxel_size=0.15)
        save_interactive_plot(pc, f"{index}_{exp}_frame{frame}_interactive.html")

    elif save_mode == "multi":
        save_multiview(fig, ax, points, index, frame, exp)

        

    if show:
        plt.show()
    else:
        plt.close(fig)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D visualization of SLAM-RF lidar point clouds (.pcd/.ply)."
    )

    # Defaults allow running with minimal arguments
    parser.add_argument(
        "-i", "--index",
        default="020925",
        help="Recording index folder, e.g. 020925 (default: 020925)."
    )
    parser.add_argument(
        "-f", "--frame",
        type=int,
        default=2,
        help="Frame number, e.g. 7 for PointCloudFrame7 (default: 0)."
    )
    parser.add_argument(
        "-e", "--exp",
        default="exp1",
        help="Experiment name, e.g. exp1. "
             "default exp1."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (default: no interactive display)."
    )
    parser.add_argument(
    "--save",
    type=str,
    choices=["gif", "int", "multi"],
    nargs="?",
    const="multi",
    default=None,
    help=(
        "Saving mode: gif | int | multi. "
        "If --save is passed without value → multi. "
        "If --save is omitted → None."
    )
    )



    args = parser.parse_args()

    visualize_pointcloud(args.index, args.frame, args.exp, args.show, args.save)

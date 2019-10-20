import glob

import scipy.io as io
import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt


def convert_mat_to_voxels(mat_path: str, cube_dimension: int = 64):
    voxels = io.loadmat(mat_path)["instance"]
    voxels = np.pad(voxels, (1, 1), mode="constant", constant_values=(0, 0))
    return (
        nd.zoom(voxels, (2, 2, 2), mode="constant", order=0)
        if cube_dimension == 64
        else voxels
    )


def get_three_d_images(directory: str):
    files = np.random.choice(glob.glob(directory), size=10)
    volumes = np.asarray([convert_mat_to_voxels(x) for x in files], dtype=np.bool)
    return volumes


def save_voxels(voxels, save_path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='blue')
    plt.savefig(save_path)


def plot_save_voxel(save_path, voxel):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(voxel, edgecolor="blue")
    plt.savefig(save_path)



import hashlib
from enum import Enum

import numpy as np
from h5py import File
from matplotlib import pyplot as plt
from numba.cuda.cudadrv.devicearray import DeviceNDArray


class ParamToIdx(Enum):
    ALPHA = 0
    BETA = 1
    GAMMA = 2
    THETA = 3
    KAPPA = 4
    MU = 5
    RHO = 6


def get_flake_hash_name(alpha, beta, gamma, theta, kappa, mu, rho):
    """
    Generate a short, deterministic filename based on the parameters of the snowflake.
    """
    # Concatenate parameters into a single string
    params_str = f"{alpha}-{beta}-{gamma}-{theta}-{kappa}-{mu}-{rho}"
    # Create a SHA-256 hash of the string
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()
    # Use the first 10 characters of the hash for the filename
    filename = f"flake_{params_hash[:10]}"
    return filename


def plot_flake_masses(flake):
    fig, axs = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(10, 10))
    # select only the top-right quadrant of the flake
    axs[0, 0].imshow(flake[:, :, 0], origin='lower', clim=(0.0, 1.0))
    axs[0, 1].imshow(flake[:, :, 1], origin='lower', clim=(0.0, 1.0))
    axs[1, 0].imshow(flake[:, :, 2], origin='lower', clim=(0.0, 1.0))
    i = axs[1, 1].imshow(flake[:, :, 3], origin='lower', clim=(0.0, 1.0))
    axs[0, 0].set_title("Attachment")
    axs[0, 1].set_title("Boundary mass")
    axs[1, 0].set_title("Ice")
    axs[1, 1].set_title("Vapour")
    cb = fig.colorbar(i, ax=axs)
    plt.show()


def save_flake_to_hdf5(flake: np.ndarray, flake_device: DeviceNDArray, step: int, h5_file: File):
    # Copy data from device to host
    flake_device.copy_to_host(flake)

    dt = np.dtype([('ice', np.float32), ('attachment', np.bool_)])
    data = np.zeros((flake.shape[0], flake.shape[1]), dtype=dt)
    data['ice'] = flake[:, :, 2]
    data['attachment'] = flake[:, :, 0].astype(np.bool_)

    # Create a dataset for this step
    dataset = h5_file.create_dataset(f'step_{step}', data=data, compression='gzip', compression_opts=9)
    dataset.attrs['step'] = step

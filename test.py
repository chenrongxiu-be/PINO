# -*- coding: utf-8 -*-
"""
Test script for physics-informed neural operator.
@author: Chen Rongxiu

This script:
    - Loads a pretrained TFNO model (same architecture as in train.py),
    - Loads test data for bridge response under random vehicle parameters,
    - Runs auto-regressive prediction,
    - Visualizes the predicted displacement field versus reference.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import warnings
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import repeat

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------------------- #
# Configuration (must be consistent with train.py)
# ------------------------------------------------------------------------- #

title: str = "pretrained"  # experiment name (i.e., folder name in `model/`)

# Data
ip_timestep: int = 340
op_timestep: int = 340
batchsize: int = 1

# Model
embed_dim: int = 512
# Transformer block
trans_layernum: int = 1
trans_headnum: int = 4
trans_hiddendim: int = 1024
trans_dropout: float = 0.0
# FNO block
lift_channels: List[int] = [32]
lift_dropout: float = 0.0
fourier_layernum: int = 4
n_modes: List[int] = [4, 64]
proj_channels: List[int] = [128, 1]
proj_dropout: float = 0.0

# Data-related configuration (fixed for this project)
bridge: str = "ssb"
node_num: int = 11
target_datasets: List[int] = [1, 2]
delta_t: float = 0.005


# ===================================================================== #
# Data utilities
# ===================================================================== #
class Data:
    """
    Data loader and normalization wrapper for TFNO testing.

    Parameters
    ----------
    bridge : str
        Bridge identifier (used in path construction).
    maxmin_dataset : int
        Dataset index used for computing min/max statistics.
    batchsize : int
        Batch size for the test DataLoader.
    device : torch.device
        Device for inference.
    """

    def __init__(
        self,
        bridge: str,
        maxmin_dataset: int,
        batchsize: int,
        device: torch.device,
    ) -> None:
        self.bridge = bridge

        temp = f"{maxmin_dataset:03d}"
        self.path_maxmin = os.path.join("data", bridge, f"dataset_{temp}")

        self.bs = batchsize
        self.device = device

    def load_data(
        self,
        dsno: int,
    ) -> Tuple[torch.utils.data.DataLoader, torch.Tensor, List[torch.Tensor]]:
        """
        Load test dataset `dsno` and construct a DataLoader.

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader for testing.
        data_nodegrid : torch.Tensor
            Node coordinates (normalized).
        dis_dist : list[torch.Tensor]
            Normalization min/max for displacement.
        """
        temp = f"{dsno:03d}"
        path = os.path.join("data", self.bridge, f"dataset_{temp}")

        # Node coordinates
        temp_arr = np.load(os.path.join(path, "grid.npy"), allow_pickle=True)
        data_nodegrid = torch.from_numpy(temp_arr)

        # Bridge response (displacement)
        temp_arr = np.load(os.path.join(path, "testdata_dis.npy"), allow_pickle=True)
        testdata_dis = torch.from_numpy(temp_arr)

        # Vehicle parameters
        temp_arr = np.load(os.path.join(path, "testdata_veh.npy"), allow_pickle=True)
        testdata_veh = torch.from_numpy(temp_arr)

        # Others
        temp_arr = np.load(
            os.path.join(path, "testdata_caselabel.npy"),
            allow_pickle=True,
        )
        testdata_caselabel = temp_arr.tolist()

        temp_arr = np.load(
            os.path.join(path, "testdata_timestep.npy"),
            allow_pickle=True,
        )
        testdata_timestep = torch.from_numpy(temp_arr)

        # Normalization statistics
        temp_arr = np.load(
            os.path.join(self.path_maxmin, "dis_maxmin.npy"),
            allow_pickle=True,
        )
        dis_maxmin = torch.from_numpy(temp_arr)

        temp_arr = np.load(
            os.path.join(self.path_maxmin, "veh_maxmin.npy"),
            allow_pickle=True,
        )
        veh_maxmin = torch.from_numpy(temp_arr)

        # Normalization
        data_nodegrid = self._norm_init_data(
            data_nodegrid,
            data_idx="grid",
            data_minmax=None,
        )
        testdata_dis_tarnode_norm, dis_dist = self._norm_init_data(
            testdata_dis[:, 0::2, :],
            data_idx="res",
            data_minmax=dis_maxmin,
        )
        testdata_veh, _ = self._norm_init_data(
            testdata_veh,
            data_idx="veh",
            data_minmax=veh_maxmin,
        )

        # Build DataLoader
        dataset = CustomDataset(
            testdata_caselabel,
            testdata_dis,
            testdata_dis_tarnode_norm,
            testdata_veh,
            testdata_timestep,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs,
            shuffle=True,
            generator=torch.Generator(self.device),
        )

        return loader, data_nodegrid, dis_dist

    @staticmethod
    def _norm_init_data(
        data: torch.Tensor,
        data_idx: str,
        data_minmax: Optional[torch.Tensor],
    ):
        """
        Normalize data according to the specified mode.

        Parameters
        ----------
        data : torch.Tensor
            Raw data tensor.
        data_idx : {'grid', 'res', 'veh'}
            Switch for different normalization strategies.
        data_minmax : torch.Tensor or None
            Min/max values used for normalization.

        Returns
        -------
        For 'grid'
            data_norm : torch.Tensor
        For 'res' and 'veh'
            data_norm : torch.Tensor
            data_dist : list[torch.Tensor]
        """
        a, b = 0.0, 1.0

        if data_idx == "grid":
            data_max = torch.max(data)
            data_min = torch.min(data)
            # Only normalize the spatial coordinate (assumed to be column 0)
            data[:, 0] = (b - a) * (data[:, 0] - data_min) / (data_max - data_min + 1e-8) + a
            return data

        if data_idx == "res":
            assert data_minmax is not None, "data_minmax must not be None for 'res'."
            data_min, data_max = data_minmax[0], data_minmax[1]
            normed_data = (b - a) * (data - data_min) / (data_max - data_min + 1e-8) + a
            data_dist = [data_min, data_max]
            return normed_data, data_dist

        if data_idx == "veh":
            assert data_minmax is not None, "data_minmax must not be None for 'veh'."
            # data_minmax shape: (p, 2)
            data_min = repeat(data_minmax[:, 0], "p -> a p b", a=1, b=1)
            data_max = repeat(data_minmax[:, 1], "p -> a p b", a=1, b=1)

            temp_shape = data.shape
            data_min_exp = data_min.expand(temp_shape[0], data_min.shape[1], temp_shape[2])
            data_max_exp = data_max.expand(temp_shape[0], data_max.shape[1], temp_shape[2])
            normed_data = (b - a) * (data - data_min_exp) / (data_max_exp - data_min_exp + 1e-8) + a
            data_dist = [data_min, data_max]
            return normed_data, data_dist

        raise ValueError(f"Unknown data_idx: {data_idx!r}")


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset wrapper for test bridge-vehicle data.

    Each item returns:
        (case_label,
         dis, dis_norm,
         veh_param,
         timestep)
    """

    def __init__(self, label, *data) -> None:
        super().__init__()
        self.label = label
        self.data = [d for d in data]

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int):
        result = [self.label[idx]]
        for d in self.data:
            result.append(d[idx, ...])
        return tuple(result)


# ===================================================================== #
# Helper functions
# ===================================================================== #
def get_ip(
    i_iter: int,
    ip_def: Optional[torch.Tensor],
    pred_def: Optional[torch.Tensor],
    stride: int,
    def_dist: List[torch.Tensor],
    batch_num: int,
    ip_timestep: int,
    op_timestep: int,
    delta_t: float,
    ts_idx: Optional[List[int]],
    device: torch.device,
    node_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare input deflection and timestamp for the current iteration.

    Parameters
    ----------
    i_iter : int
        Current iteration index (0-based).
    ip_def : torch.Tensor or None
        Previous input sequence (ignored when i_iter == 0).
    pred_def : torch.Tensor or None
        Previous predicted sequence (used when i_iter > 0).
    stride : int
        Length of prediction window (op_timestep).
    def_dist : list[torch.Tensor]
        Min/max used for displacement normalization.
    batch_num : int
        Batch size.
    ip_timestep : int
        Length of input window.
    op_timestep : int
        Length of output window (not used directly, kept for consistency).
    delta_t : float
        Time step size.
    ts_idx : list[int] or None
        Time index range of the last prediction window.
    device : torch.device
        Device used for inference.
    node_num : int
        Number of bridge nodes.

    Returns
    -------
    ip_def : torch.Tensor
        Input deflection sequence, shape (batch, node, ip_timestep).
    timestamp : torch.Tensor
        Time stamps, shape (batch, 1, ip_timestep).
    """
    if i_iter == 0:
        # Initialize input deflection with small random noise around zero
        a, b = 0.0, 1.0
        data_min, data_max = def_dist[0], def_dist[1]
        ini_def = (b - a) * (0.0 - data_min) / (data_max - data_min + 1e-8) + a
        ini_def = ini_def.to(device)

        noise = (1.0 / 1e6) * torch.randn(batch_num, node_num, ip_timestep, device=device)
        ip_def = noise + ini_def * torch.ones(batch_num, node_num, ip_timestep, device=device)

        timestamp_1d = torch.arange(0, ip_timestep, 1, dtype=torch.float64, device=device) * delta_t
    else:
        # Closed-loop: directly feed previous prediction as input
        assert pred_def is not None, "pred_def must not be None when i_iter > 0."
        ip_def = pred_def
        assert ts_idx is not None, "ts_idx must not be None when i_iter > 0."
        timestamp_1d = torch.arange(
            ts_idx[0],
            ts_idx[0] + ip_timestep,
            1,
            dtype=torch.float64,
            device=device,
        ) * delta_t

    timestamp = repeat(timestamp_1d, "t -> b k t", b=batch_num, k=1)
    return ip_def, timestamp


def denorm_data(
    data: torch.Tensor,
    data_min: torch.Tensor,
    data_max: torch.Tensor,
) -> torch.Tensor:
    """
    De-normalize data from [0, 1] back to physical space.
    """
    a, b = 0.0, 1.0
    data_min = data_min.to(data.device)
    data_max = data_max.to(data.device)
    return (data - a) * (data_max - data_min) / (b - a + 1e-8) + data_min


def norm_data(
    data: torch.Tensor,
    data_min: torch.Tensor,
    data_max: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize data into [0, 1].
    """
    a, b = 0.0, 1.0
    data_min = data_min.to(data.device)
    data_max = data_max.to(data.device)
    return (b - a) * (data - data_min) / (data_max - data_min + 1e-8) + a


def plot_cm(
    pred_def: torch.Tensor,
    tar_def: torch.Tensor,
    delta_t: float,
    factor: float = 1000.0,
) -> plt.Figure:
    """
    Plot reference, prediction and absolute error of displacement field.

    Parameters
    ----------
    pred_def : torch.Tensor
        Predicted displacement, shape (node, timestep).
    tar_def : torch.Tensor
        Reference displacement, shape (node, timestep).
    delta_t : float
        Time step size.
    factor : float, optional
        Scaling factor for displacement (e.g., to convert m -> mm).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    """
    node_num_local = pred_def.shape[0]
    timestep = pred_def.shape[1]

    def_lim = [-1.0, 4.0]
    def_err_lim = [0.0, 0.1]

    colors = [
        (0.12, 0.27, 0.43),
        (0.22, 0.40, 0.58),
        (0.32, 0.56, 0.68),
        (0.45, 0.74, 0.84),
        (0.67, 0.86, 0.88),
        (1.00, 0.90, 0.72),
        (1.00, 0.82, 0.44),
        (0.97, 0.67, 0.35),
        (0.94, 0.54, 0.28),
        (0.91, 0.38, 0.33),
    ]
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    colors = [(1.0, 1.0, 1.0), (0.91, 0.38, 0.33)]
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    fig, axs = plt.subplots(
        1,
        3,
        layout="constrained",
        figsize=(15, 2),
        dpi=800,
    )

    t = np.arange(0, timestep, 1) * delta_t
    X, Y = np.meshgrid(t, np.arange(1, node_num_local + 1, 1))

    # Reference
    c = axs[0].pcolor(
        X,
        Y,
        factor * tar_def.detach().cpu().numpy(),
        vmin=def_lim[0],
        vmax=def_lim[1],
        cmap=cmap1,
    )
    fig.colorbar(c, ax=axs[0])

    # Prediction
    c = axs[1].pcolor(
        X,
        Y,
        factor * pred_def.detach().cpu().numpy(),
        vmin=def_lim[0],
        vmax=def_lim[1],
        cmap=cmap1,
    )
    fig.colorbar(c, ax=axs[1])

    # Absolute error
    mae = factor * torch.abs(tar_def.cpu() - pred_def.cpu())
    c = axs[2].pcolor(
        X,
        Y,
        mae.detach().numpy(),
        vmin=def_err_lim[0],
        vmax=def_err_lim[1],
        cmap=cmap2,
    )
    fig.colorbar(c, ax=axs[2])

    title = ["Deflection"]
    subtitle = ["ref.", "pred.", "absolute error"]
    unit = ["mm"]
    tlim = timestep * delta_t
    for j in range(3):
        axs[j].set_title(f"{title[0]} {subtitle[j]} ({unit[0]})")
        axs[j].set_xlim(0.0, tlim)
        axs[j].set_yticklabels(axs[j].get_yticks().astype(int))

    return fig


# ===================================================================== #
# Main script
# ===================================================================== #
def main() -> None:
    # ------------------------------------------------------------------ #
    # Device setup
    # ------------------------------------------------------------------ #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    data_module = Data(
        bridge=bridge,
        maxmin_dataset=target_datasets[-1],
        batchsize=batchsize,
        device=device,
    )

    dataloaders: List[torch.utils.data.DataLoader] = []
    nodegrid: Optional[torch.Tensor] = None
    def_dist: Optional[List[torch.Tensor]] = None

    for ds_no in target_datasets:
        temp_loader, nodegrid, def_dist = data_module.load_data(ds_no)
        dataloaders.append(temp_loader)

    assert nodegrid is not None, "Node grid must not be None."
    assert def_dist is not None, "def_dist must not be None."

    # ------------------------------------------------------------------ #
    # Load model (from train.py)
    # ------------------------------------------------------------------ #
    print("Inference...")

    spec = importlib.util.spec_from_file_location("train", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train"] = train_module
    assert spec.loader is not None
    spec.loader.exec_module(train_module)

    model = train_module.Build_model(
        nodegrid=nodegrid,
        ip_timestep=ip_timestep,
        op_timestep=op_timestep,
        embed_dim=embed_dim,
        node_num=node_num,
        trans_layernum=trans_layernum,
        trans_headnum=trans_headnum,
        trans_hiddendim=trans_hiddendim,
        trans_dropout=trans_dropout,
        lift_channels=lift_channels,
        lift_dropout=lift_dropout,
        fourier_layernum=fourier_layernum,
        n_modes=n_modes,
        proj_channels=proj_channels,
        proj_dropout=proj_dropout,
    ).to(device)

    model_path = os.path.join("model", title, "model.pth")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ------------------------------------------------------------------ #
    # Randomly pick a sample from the first test loader
    # ------------------------------------------------------------------ #
    first_loader = dataloaders[0]
    dataset = first_loader.dataset
    num_samples = len(dataset)
    assert num_samples > 0, "Test dataset is empty."

    # Randomly select one index and fetch the sample directly
    rand_idx = random.randrange(num_samples)
    (
        data_caselabel,
        data_dis,
        data_dis_tarnode_norm,
        data_veh,
        data_timestep,
    ) = dataset[rand_idx]

    # Add batch dimension
    data_dis = data_dis.unsqueeze(0)
    data_dis_tarnode_norm = data_dis_tarnode_norm.unsqueeze(0)
    data_veh = data_veh.unsqueeze(0)
    data_timestep = data_timestep.unsqueeze(0)

    max_timestep = int(torch.max(data_timestep).item())
    iter_num = int(np.ceil(max_timestep / ip_timestep))
    stride = op_timestep
    ip_def: Optional[torch.Tensor] = None
    pred_def: Optional[torch.Tensor] = None
    ts_idx: Optional[List[int]] = None
    batch_num = data_veh.shape[0]

    data_veh = data_veh.to(device)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        for i_iter in range(iter_num):
            # Prepare input and timestamps
            ip_def, timestamp = get_ip(
                i_iter=i_iter,
                ip_def=ip_def,
                pred_def=pred_def,
                stride=stride,
                def_dist=def_dist,
                batch_num=batch_num,
                ip_timestep=ip_timestep,
                op_timestep=op_timestep,
                delta_t=delta_t,
                ts_idx=ts_idx,
                device=device,
                node_num=node_num,
            )

            # Prepare timestep index
            ts_idx = [i_iter * stride + 1, i_iter * stride + 1 + op_timestep]
            delta_ts = ts_idx[1] - max_timestep
            if delta_ts == 0:
                break
            if delta_ts > 0:
                ts_idx[1] = int(max_timestep)

            # Predict
            pred_def = model(ip_def, data_veh[:, :, ts_idx[0] - 1], timestamp)

            # Denormalize and concatenate predictions
            pred_def_dn = denorm_data(pred_def, def_dist[0], def_dist[1])
            if i_iter == 0:
                b, n, t = pred_def.shape
                pred_def_cat = torch.cat(
                    (torch.zeros(b, n, 1, device=device, dtype=pred_def_dn.dtype), pred_def_dn),
                    dim=-1,
                )
            else:
                pred_def_cat = torch.cat((pred_def_cat, pred_def_dn), dim=-1)

    # ------------------------------------------------------------------ #
    # Plot prediction vs reference
    # ------------------------------------------------------------------ #
    print("Plot ...")
    os.makedirs("plot", exist_ok=True)
    
    max_timestep = min(
        max_timestep, 
        data_dis[0, 0::2, :max_timestep].shape[-1], 
        pred_def_cat[0, :, :max_timestep].shape[-1]
    )
    prediction = pred_def_cat[0, :, :max_timestep]
    target = data_dis[0, 0::2, :max_timestep]
    fig = plot_cm(prediction, target, delta_t)
    fig.savefig(os.path.join("plot", f"DisplacementField_ModelTitle.{title}.png"))
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()

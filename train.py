# -*- coding: utf-8 -*-
"""
Train a physics-informed neural operator for forecasting bridge response under random vehicle parameters.
@author: Chen Rongxiu

"""

from __future__ import annotations

import math
import os
import shutil
import sys
import warnings
from timeit import default_timer
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import repeat

warnings.filterwarnings("ignore")
torch.manual_seed(2)
torch.set_default_dtype(torch.float64)

def main() -> None:
    """
    Main entry point for training the physics-informed TFNO model.

    This function:
    1. Sets configuration and paths.
    2. Loads data and normalization statistics.
    3. Builds the TFNO model.
    4. Runs the training loop and saves the trained model.
    """
    
    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    title = "251001_trainPINO"  # experiment title (also used as folder name)

    # Data
    ip_timestep: int = 340          # timesteps of input data window
    op_timestep: int = 340          # timesteps predicted per iteration

    # Model hyperparameters
    embed_dim: int = 512            # embedding dimension

    # Transformer block
    trans_layernum: int = 1         # number of Transformer layers
    trans_headnum: int = 4          # number of attention heads
    trans_hiddendim: int = 1024     # hidden dimension in Transformer FFN
    trans_dropout: float = 0.0      # dropout in Transformer FFN and MHA

    # FNO block
    lift_channels: List[int] = [32]  # channel sizes of lifting MLP
    lift_dropout: float = 0.0        # dropout in lifting MLP
    fourier_layernum: int = 4        # number of Fourier layers
    n_modes: List[int] = [4, 64]     # truncation modes [spatial, temporal]
    proj_channels: List[int] = [128, 1]  # channel sizes of projection MLP
    proj_dropout: float = 0.0        # dropout in projection MLP

    # Training hyperparameters
    batchsize: int = 8               # batch size
    epoch_num: int = 600             # number of epochs
    adam_lr: float = 1e-3            # learning rate
    adam_weight_decay: float = 1e-5  # weight decay
    scheduler_step: int = 100        # period of LR decay
    scheduler_gamma: float = 0.5     # LR decay factor

    # Loss weights
    w_dataloss: float = 1.0
    w_freqloss_def: float = 1.0
    w_bcloss_def: float = 1.0
    w_velloss: float = 1.0
    w_freqloss_vel: float = 1.0
    w_bcloss_vel: float = 1.0
    w_accloss: float = 1.0
    w_freqloss_acc: float = 1.0
    w_bcloss_acc: float = 1.0
    w_geloss: float = 0.25
        
    # ------------------------------------------------------------------ #
    # Fixed configuration
    # ------------------------------------------------------------------ #
    bridge: str = "ssb"
    node_num: int = 11
    target_datasets: List[int] = [1, 2]
    delta_t: float = 0.005
    
    # Loss weights as a simple list (kept for backward compatibility)
    loss_weight = [
        w_dataloss,
        w_freqloss_def,
        w_bcloss_def,
        w_velloss,
        w_freqloss_vel,
        w_bcloss_vel,
        w_accloss,
        w_freqloss_acc,
        w_bcloss_acc,
        w_geloss,
    ]
    
    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    model_path = os.path.join("model", title)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    
    # ------------------------------------------------------------------ #
    # Device setup
    # ------------------------------------------------------------------ #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------ #
    # Load data & normalization statistics
    # ------------------------------------------------------------------ #
    data_module = Data(bridge=bridge,
                       maxmin_dataset=target_datasets[-1],
                       batchsize=batchsize,
                       device=device)

    train_loaders: List[torch.utils.data.DataLoader] = []
    nodegrid: Optional[torch.Tensor] = None
    def_dist = vel_dist = acc_dist = p_dist = None

    for ds_no in target_datasets:
        (
            temp_train_loader,
            nodegrid,
            def_dist,
            vel_dist,
            acc_dist,
            p_dist,
        ) = data_module.load_data(ds_no)
        train_loaders.append(temp_train_loader)
        
    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model = Build_model(
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
    model.apply(init_weights)
        
    # ------------------------------------------------------------------ #
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=adam_lr,
        weight_decay=adam_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma,
    )
    
    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model_path=model_path,
        device=device,
        ip_timestep=ip_timestep,
        op_timestep=op_timestep,
        delta_t=delta_t,
        def_dist=def_dist,
        vel_dist=vel_dist,
        acc_dist=acc_dist,
        p_dist=p_dist,
        node_num=node_num,
        loss_weight=loss_weight,
    )
    trainer.train(
        model=model,
        epoch_num=epoch_num,
        train_loaders=train_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        batchsize=batchsize,
    )
    
    
# ===================================================================== #
# Data utilities
# ===================================================================== #
class Data:
    """
    Data loader and normalization wrapper for TFNO training.

    Parameters
    ----------
    bridge : str
        Bridge identifier (used in path construction).
    maxmin_dataset : int
        Dataset index used for computing min/max statistics.
    batchsize : int
        Batch size for the training DataLoader.
    device : torch.device
        Device for training.
    """

    def __init__(self,
                 bridge: str,
                 maxmin_dataset: int,
                 batchsize: int,
                 device: torch.device) -> None:
        self.bridge = bridge
        temp = f"{maxmin_dataset:03d}"
        self.path_maxmin = os.path.join("data", bridge, f"dataset_{temp}")

        self.bs_train = batchsize
        self.device = device
    
    def load_data(
        self, dsno: int
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
    ]:
        """
        Load dataset `dsno` and construct a DataLoader.

        Returns
        -------
        train_loader : DataLoader
            Pytorch DataLoader for training.
        data_nodegrid : torch.Tensor
            Node coordinates (normalized).
        dis_dist, vel_dist, acc_dist : list[torch.Tensor]
            Normalization min/max for displacement, velocity, acceleration.
        p_dist : torch.Tensor
            Normalization min/max for equivalent force vector.
        """
        temp = f"{dsno:03d}"
        path = os.path.join("data", self.bridge, f"dataset_{temp}")
    
        # Node coordinates
        temp_arr = np.load(os.path.join(path, "grid.npy"), allow_pickle=True)
        data_nodegrid = torch.from_numpy(temp_arr)
    
        # Bridge response
        temp_arr = np.load(os.path.join(path, "traindata_dis.npy"), allow_pickle=True)
        traindata_dis = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(path, "traindata_vel.npy"), allow_pickle=True)
        traindata_vel = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(path, "traindata_acc.npy"), allow_pickle=True)
        traindata_acc = torch.from_numpy(temp_arr)

        # Vehicle parameters
        temp_arr = np.load(os.path.join(path, "traindata_veh.npy"), allow_pickle=True)
        traindata_veh = torch.from_numpy(temp_arr)

        # Vehicle response
        temp_arr = np.load(os.path.join(path, "traindata_vehdis.npy"), allow_pickle=True)
        traindata_vehdis = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(path, "traindata_vehvel.npy"), allow_pickle=True)
        traindata_vehvel = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(path, "traindata_vehacc.npy"), allow_pickle=True)
        traindata_vehacc = torch.from_numpy(temp_arr)

        # Coefficient matrices and force vector
        temp_arr = np.load(os.path.join(path, "traindata_pgreff.npy"), allow_pickle=True)
        traindata_pgreff = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(path, "traindata_pgr.npy"), allow_pickle=True)
        traindata_pgr = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(path, "traindata_cgr.npy"), allow_pickle=True)
        traindata_cgr = torch.from_numpy(temp_arr).to(torch.float64)

        temp_arr = np.load(os.path.join(path, "traindata_mgr.npy"), allow_pickle=True)
        traindata_mgr = torch.from_numpy(temp_arr)
    
        # Others
        temp_arr = np.load(
            os.path.join(path, "traindata_caselabel.npy"),
            allow_pickle=True,
        )
        traindata_caselabel = temp_arr.tolist()
        
        temp_arr = np.load(os.path.join(path, "traindata_timestep.npy"), allow_pickle=True)
        traindata_timestep = torch.from_numpy(temp_arr)
        
        # Normalization statistics
        temp_arr = np.load(os.path.join(self.path_maxmin, "dis_maxmin.npy"), allow_pickle=True)
        dis_maxmin = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(self.path_maxmin, "vel_maxmin.npy"), allow_pickle=True)
        vel_maxmin = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(self.path_maxmin, "acc_maxmin.npy"), allow_pickle=True)
        acc_maxmin = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(self.path_maxmin, "veh_maxmin.npy"), allow_pickle=True)
        veh_maxmin = torch.from_numpy(temp_arr)

        temp_arr = np.load(os.path.join(self.path_maxmin, "pgreff_maxmin.npy"), allow_pickle=True)
        p_dist = torch.from_numpy(temp_arr)
        
        # Normalization
        data_nodegrid = self._norm_init_data(data_nodegrid, data_idx="grid", data_minmax=None)

        traindata_dis_tarnode_norm, dis_dist = self._norm_init_data(
            traindata_dis[:, 0::2, :],
            data_idx="res",
            data_minmax=dis_maxmin,
        )
        traindata_vel_tarnode_norm, vel_dist = self._norm_init_data(
            traindata_vel[:, 0::2, :],
            data_idx="res",
            data_minmax=vel_maxmin,
        )
        traindata_acc_tarnode_norm, acc_dist = self._norm_init_data(
            traindata_acc[:, 0::2, :],
            data_idx="res",
            data_minmax=acc_maxmin,
        )
        traindata_veh, _ = self._norm_init_data(
            traindata_veh,
            data_idx="veh",
            data_minmax=veh_maxmin,
        )
        
        # Build DataLoader
        dataset = CustomDataset(
            traindata_caselabel,
            traindata_dis,
            traindata_dis_tarnode_norm,
            traindata_vel,
            traindata_vel_tarnode_norm,
            traindata_acc,
            traindata_acc_tarnode_norm,
            traindata_veh,
            traindata_vehdis,
            traindata_vehvel,
            traindata_vehacc,
            traindata_pgreff,
            traindata_pgr,
            traindata_cgr,
            traindata_mgr,
            traindata_timestep,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs_train,
            shuffle=True,
            generator=torch.Generator(self.device),
        )

        return train_loader, data_nodegrid, dis_dist, vel_dist, acc_dist, p_dist
    
    @staticmethod
    def _norm_init_data(
        data: torch.Tensor,
        data_idx: str,
        data_minmax: Optional[torch.Tensor],
    ):
        """
        Normalize data according to the specified mode.
        Method: Min-max normalization
        Range: [a, b]
    
        Parameters
        ----------
        data : torch.Tensor
            Raw data tensor.
        data_idx : {'grid', 'res', 'veh'}
            Switch for different data treatment.
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
    
            # Only normalize the spatial coordinate here (assumed column 0)
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
    Custom dataset wrapper for multi-field bridge-vehicle data.

    Each item returns:
        (case_label,
         dis, dis_norm,
         vel, vel_norm,
         acc, acc_norm,
         veh_param,
         veh_dis, veh_vel, veh_acc,
         pgreff, pgr, cgr, mgr,
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
# Model definition
# ===================================================================== #
def init_weights(m: nn.Module) -> None:
    """
    Initialize weights of Linear and Conv2d layers with Xavier uniform.

    Parameters
    ----------
    m : nn.Module
        Module to be initialized (used via `apply`).
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        
        
class Build_model(nn.Module):
    """
    TFNO model for bridge response prediction.

    This model contains:
    1. Embedding layer
    2. Transformer block (stack of Transformer layers)
    3. FNO block (lifting MLP -> Fourier layers -> projection MLP)

    Inputs
    ------
    x_def : torch.Tensor
        Previous deflection field, shape (batch, node, timestep).
    x_veh : torch.Tensor
        Vehicle parameters, shape (batch, 1, para_num).
    timestamp : torch.Tensor
        Time stamps, shape (batch, 1, timestep).

    Output
    ------
    out : torch.Tensor
        Future deflection field, shape (batch, node, timestep).

    Code adopted from:
        - TFNO, https://github.com/chenrongxiu-be/TFNO
    """ 

    def __init__(
        self,
        nodegrid: torch.Tensor,
        ip_timestep: int,
        op_timestep: int,
        embed_dim: int,
        node_num: int,
        trans_layernum: int,
        trans_headnum: int,
        trans_hiddendim: int,
        trans_dropout: float,
        lift_channels: List[int],
        lift_dropout: float,
        fourier_layernum: int,
        n_modes: List[int],
        proj_channels: List[int],
        proj_dropout: float,
    ) -> None:
        super().__init__()

        self.nodegrid = nodegrid
        self.node_num = node_num

        self.modlist_trans = nn.ModuleList()
        self.modlist_lift = nn.ModuleList()
        self.modlist_fl = nn.ModuleList()
        self.modlist_proj = nn.ModuleList()

        # Embedding layer
        self.embedding_layer = Embedding_layer(nodegrid, ip_timestep, embed_dim)
    
        # Transformer layers
        for i in range(trans_layernum):
            self.modlist_trans.append(
                Transformer_layer(
                    embed_dim=embed_dim,
                    num_heads=trans_headnum,
                    hidden_dim=trans_hiddendim,
                    op_timestep=op_timestep,
                    dropout=trans_dropout,
                    last_layer=(i == trans_layernum - 1),
                    node_num=node_num,
                )
            )

        # Lifting MLP
        lift_layernum = len(lift_channels)
        for i in range(lift_layernum):
            in_channel = 3 if i == 0 else lift_channels[i - 1]
            out_channel = lift_channels[i]
            self.modlist_lift.append(nn.Linear(in_channel, out_channel))
            if i < lift_layernum - 1:
                self.modlist_lift.append(nn.GELU())
                if lift_dropout > 0.0:
                    self.modlist_lift.append(nn.Dropout(p=lift_dropout))

        # Fourier layers
        for i in range(fourier_layernum):
            self.modlist_fl.append(
                Fourier_layer(
                    in_channel=lift_channels[-1],
                    out_channel=lift_channels[-1],
                    n_modes=n_modes,
                    last_layer=(i == fourier_layernum - 1),
                )
            )

        # Projection MLP
        proj_layernum = len(proj_channels)
        for i in range(proj_layernum):
            in_channel = lift_channels[-1] if i == 0 else proj_channels[i - 1]
            out_channel = proj_channels[i]
            self.modlist_proj.append(nn.Linear(in_channel, out_channel))
            if i < proj_layernum - 1:
                self.modlist_proj.append(nn.GELU())
                if proj_dropout > 0.0:
                    self.modlist_proj.append(nn.Dropout(p=proj_dropout))
    
    def forward(
        self,
        x_def: torch.Tensor,
        x_veh: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the TFNO model.

        Parameters
        ----------
        x_def : torch.Tensor
            Previous deflection field, shape (batch, node, timestep).
        x_veh : torch.Tensor
            Vehicle parameters, shape (batch, 1, para_num).
        timestamp : torch.Tensor
            Time stamps, shape (batch, 1, timestep).

        Returns
        -------
        out : torch.Tensor
            Predicted deflection field, shape (batch, node, timestep).
        """
        # Embedding layer
        x = self.embedding_layer(x_def, x_veh)

        # Transformer block
        for layer in self.modlist_trans:
            x = layer(x)

        # Concatenate node coordinate and timestamp
        x = x.unsqueeze(-1)
        nodegrid_rep = repeat(
            self.nodegrid,
            "n k -> b n t k",
            b=x.shape[0],
            t=x.shape[2],
        ).to(x.device)
        ts_rep = repeat(timestamp, "b n t -> b (repeat n) t", repeat=x.shape[1]).unsqueeze(-1).to(x.device)
        x = torch.cat((x, nodegrid_rep, ts_rep), dim=-1)

        # FNO (lifting MLP -> Fourier layers -> projection MLP)
        for layer in self.modlist_lift:
            x = layer(x)

        # (b, n, t, c) -> (b, c, n, t)
        x = x.permute(0, 3, 1, 2)
        for layer in self.modlist_fl:
            x = layer(x)
        # (b, c, n, t) -> (b, n, t, c)
        x = x.permute(0, 2, 3, 1)
        for layer in self.modlist_proj:
            x = layer(x)

        out = x.squeeze(-1)
        return out
    
    
class Embedding_layer(nn.Module):
    """
    Embedding layer for bridge response and vehicle parameters.

    - Applies positional encoding based on node coordinates.
    - Encodes temporal deflection sequence and vehicle parameters.
    """

    def __init__(self, nodegrid: torch.Tensor, ip_timestep: int, embed_dim: int) -> None:
        super().__init__()

        self.nodegrid = nodegrid
        self.fc1 = nn.Linear(13, embed_dim)          # vehicle parameters
        self.fc2 = nn.Linear(ip_timestep, embed_dim)  # temporal deflection

        d = int(np.ceil(ip_timestep / 2))
        self.omega = nn.Parameter(torch.rand(d), requires_grad=True)

    def pos_encoder(self, data: torch.Tensor) -> torch.Tensor:
        """
        Positional encoding based on spatial node coordinate.

        Parameters
        ----------
        data : torch.Tensor
            Input deflection, shape (batch, node, timestep).

        Returns
        -------
        data_pe : torch.Tensor
            Data with added positional encoding.
        """
        bat_num, loc_num, timestep_num = data.shape
        assert timestep_num % 2 == 0, "timestep_num must be even for this encoder."

        pe = torch.zeros(bat_num, loc_num, timestep_num, device=data.device, dtype=data.dtype)

        self.nodegrid = self.nodegrid.to(data.device)
        # use x-coordinate (column 0) for positional encoding
        x_coord = self.nodegrid[:, 0][:, None]  # (node, 1)

        pe[:, :, 0::2] = torch.sin(self.omega[None, :] * x_coord)
        pe[:, :, 1::2] = torch.cos(self.omega[None, :] * x_coord)

        data = data + pe
        return data

    def forward(self, x_def: torch.Tensor, x_veh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.

        Parameters
        ----------
        x_def : torch.Tensor
            Deflection field, shape (batch, node, timestep).
        x_veh : torch.Tensor
            Vehicle parameters, shape (batch, 1, para_num).

        Returns
        -------
        out : torch.Tensor
            Embedded representation, shape (batch, node+1, embed_dim).
        """
        out_def = self.fc2(self.pos_encoder(x_def))
        out_veh = self.fc1(x_veh.to(out_def.dtype)).unsqueeze(1)
        out = torch.cat((out_veh, out_def), dim=1)
        return out
    
    
class Transformer_layer(nn.Module):
    """
    Single Transformer encoder layer with optional post-projection.

    If `last_layer=True`, this layer also:
      - truncates the vehicle dimension (keeps only node_num positions),
      - projects embedding dimension to op_timestep,
      - applies LayerNorm along timestep dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        op_timestep: int,
        dropout: float,
        last_layer: bool,
        node_num: int,
    ) -> None:
        super().__init__()

        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.FFN = nn.ModuleList(
            [
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
            ]
        )
        if dropout > 0.0:
            self.FFN.append(nn.Dropout(p=dropout))
        self.FFN.append(nn.Linear(hidden_dim, embed_dim))

        # Normalization
        self.last_layer = last_layer
        self.node_num = node_num
        self.norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])
        if last_layer:
            self.norm.append(nn.LayerNorm(op_timestep))

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Linear projection from embedding to time dimension
        self.fc = nn.Linear(embed_dim, op_timestep)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer layer.

        Parameters
        ----------
        q : torch.Tensor
            Input embeddings, shape (batch, seq_len, embed_dim).

        Returns
        -------
        out : torch.Tensor
            Output embeddings, shape:
                - (batch, seq_len, embed_dim) if not last layer;
                - (batch, node_num, op_timestep) if last layer.
        """
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        ):
            # Multi-head attention with residual connection
            x_skip = q
            q_norm = self.norm[0](q)
            attn_output, _ = self.mha(
                q_norm,
                q_norm,
                q_norm,
                average_attn_weights=False,
            )
            x = self.dropout(attn_output)
            x = x + x_skip

            # Feed-forward network with residual connection
            x_skip = x
            x_norm = self.norm[1](x)
            for layer in self.FFN:
                x_norm = layer(x_norm)
            x = self.dropout(x_norm)
            out = x + x_skip

            # If last layer, truncate vehicle dimension and project
            if self.last_layer:
                out = out[:, -self.node_num :, :]
                out = self.fc(out)
                out = self.norm[2](out)

            return out

        
class Fourier_layer(nn.Module):
    """
    2D Fourier layer used in FNO.

    Performs:
        - 2D FFT
        - spectral truncation and complex multiplication
        - inverse FFT
        - 1x1 Conv skip connection
        - GELU nonlinearity (if not the last layer)
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        n_modes: List[int],
        last_layer: bool = False,
    ) -> None:
        super().__init__()

        self.last_layer = last_layer
        self.out_channel = out_channel
        self.n_modes = n_modes

        # Spectral weights (complex)
        scale = 1.0 / (in_channel * out_channel)
        self.weight1, self.weight2 = [
            nn.Parameter(
                scale
                * torch.rand(
                    [in_channel, out_channel] + n_modes,
                    dtype=torch.cfloat,
                )
            )
            for _ in range(2)
        ]

        # Skip connection in physical space
        self.skip = nn.Conv2d(in_channel, in_channel, kernel_size=1)

        # Activation
        if not self.last_layer:
            self.nonlinearity = nn.GELU()
            
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Fourier layer.

        Parameters
        ----------
        x : torch.Tensor
            Input in physical space, shape (batch, in_channel, nx, nt).

        Returns
        -------
        torch.Tensor
            Output in physical space, shape (batch, out_channel, nx, nt).
        """
        # FFT to frequency domain
        x_ft = torch.fft.rfft2(x).to(torch.cfloat)

        batchsize = x.shape[0]
        nx, nt_half = x.size(-2), x.size(-1) // 2 + 1
        out_ft = torch.zeros(
            batchsize,
            self.out_channel,
            nx,
            nt_half,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.n_modes[0], : self.n_modes[1]] = self.compl_mul2d(
            x_ft[:, :, : self.n_modes[0], : self.n_modes[1]],
            self.weight1,
        )
        out_ft[:, :, -self.n_modes[0] :, : self.n_modes[1]] = self.compl_mul2d(
            x_ft[:, :, -self.n_modes[0] :, : self.n_modes[1]],
            self.weight2,
        )

        # Inverse FFT back to physical space
        out_ft = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        # Skip connection
        out_skip = self.skip(x)
        out = out_ft + out_skip

        if not self.last_layer:
            out = self.nonlinearity(out)

        return out
    
    
# ===================================================================== #
# Training utilities
# ===================================================================== #
class Trainer:
    """
    Trainer for TFNO-based PINO model.

    Parameters
    ----------
    model_path : str
        Path to save the trained model.
    device : torch.device
        Device used for training.
    ip_timestep : int
        Length of input window.
    op_timestep : int
        Length of predicted window per iteration.
    delta_t : float
        Time step size.
    def_dist, vel_dist, acc_dist : list[torch.Tensor]
        Min/max used for normalization of displacement, velocity, acceleration.
    p_dist : torch.Tensor
        Min/max used for normalization of equivalent force.
    node_num : int
        Number of bridge nodes.
    loss_weight : list[float]
        Weights for different loss terms.
    """
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        ip_timestep: int,
        op_timestep: int,
        delta_t: float,
        def_dist: List[torch.Tensor],
        vel_dist: List[torch.Tensor],
        acc_dist: List[torch.Tensor],
        p_dist: torch.Tensor,
        node_num: int,
        loss_weight: List[float],
    ) -> None:

        self.model_path = model_path
        self.device = device

        self.ip_timestep = ip_timestep
        self.op_timestep = op_timestep
        self.delta_t = delta_t

        self.def_dist = def_dist
        self.vel_dist = vel_dist
        self.acc_dist = acc_dist
        self.p_dist = p_dist

        self.node_num = node_num
        self.loss_weight = loss_weight

        self.list_trainloss: Optional[torch.Tensor] = None
        
    def train(
        self,
        model: nn.Module,
        epoch_num: int,
        train_loaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        batchsize: int,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        model : nn.Module
            TFNO model to be trained.
        epoch_num : int
            Number of training epochs.
        train_loaders : list of DataLoader
            Training data loaders (possibly from multiple datasets).
        optimizer : torch.optim.Optimizer
            Optimizer used for training.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            LR scheduler.
        batchsize : int
            Batch size (used only for counting).
        """
        # Count total batches (only for logging)
        n_train_sam = 0
        n_train_batch = 0
        for loader in train_loaders:
            n_train_sam += len(loader.dataset)
            temp = n_train_sam // batchsize + 1 if (len(loader.dataset) % batchsize) != 0 else len(loader.dataset) // batchsize
            n_train_batch += temp

        sys.stdout.flush()

        t1 = default_timer()
        for ep in range(epoch_num):
            print(f"\nEpoch: {ep + 1} / {epoch_num}")
            print("# Training:")
            t2 = default_timer()

            model.train()
            train_loss = self.batch_loop(
                model=model,
                dataloaders=train_loaders,
                optimizer=optimizer,
                ep=ep + 1,
                n_batch=n_train_batch,
            )

            ave_trainloss = torch.mean(train_loss, dim=0, keepdim=True)
            if self.list_trainloss is None:
                self.list_trainloss = train_loss.cpu()
            else:
                self.list_trainloss = torch.cat(
                    (self.list_trainloss, train_loss.cpu()),
                    dim=0,
                )
            
            # Step scheduler
            scheduler.step()

            epoch_time = default_timer() - t2
            print(f"- {ep + 1:4d}th epoch took {epoch_time:0.2f} sec.")
            print(f"- Average train_loss: {ave_trainloss}.\n")
            print("=========================================================\n")

        # Save model
        torch.save(
            {"model_state_dict": model.state_dict()},
            os.path.join(self.model_path, "model.pth"),
        )

        # Print training info
        total_time = default_timer() - t1
        time_per_epoch = round(total_time / epoch_num, 6)
        print(f"\nAll {epoch_num} epochs finished.")
        print(f"The entire training took {total_time:0.2f} sec in total.")
        print(f" ({time_per_epoch:0.2f} sec per epoch)")
            
    def batch_loop(
        self,
        model: nn.Module,
        dataloaders: List[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        ep: int,
        n_batch: int,
    ) -> torch.Tensor:
        """
        Loop over all training batches for one epoch.

        Returns
        -------
        loss_log : torch.Tensor
            Logged loss values for this epoch.
        """
        loss_log: Optional[torch.Tensor] = None
        count_batch = 0

        for dataloader in dataloaders:
            for (
                data_caselabel,
                data_def,
                data_def_tarnode_norm,
                data_vel,
                data_vel_tarnode_norm,
                data_acc,
                data_acc_tarnode_norm,
                data_veh,
                data_vehdis,
                data_vehvel,
                data_vehacc,
                data_pgreff,
                data_pgr,
                data_cgr,
                data_mgr,
                data_timestep,
            ) in dataloader:

                # Move tensors to device
                (
                    data_def,
                    data_def_tarnode_norm,
                    data_vel,
                    data_vel_tarnode_norm,
                    data_acc,
                    data_acc_tarnode_norm,
                    data_veh,
                    data_vehdis,
                    data_vehvel,
                    data_vehacc,
                    data_pgreff,
                    data_pgr,
                    data_cgr,
                    data_mgr,
                ) = (
                    data_def.to(self.device),
                    data_def_tarnode_norm.to(self.device),
                    data_vel.to(self.device),
                    data_vel_tarnode_norm.to(self.device),
                    data_acc.to(self.device),
                    data_acc_tarnode_norm.to(self.device),
                    data_veh.to(self.device),
                    data_vehdis.to(self.device),
                    data_vehvel.to(self.device),
                    data_vehacc.to(self.device),
                    data_pgreff.to(self.device),
                    data_pgr.to(self.device),
                    data_cgr.to(self.device),
                    data_mgr.to(self.device),
                )
                    
                # Maximum valid timestep in this batch 
                max_timestep = int(max(data_timestep))
                iter_num = int(np.ceil(max_timestep / self.ip_timestep))
                
                count_batch += 1
                stride = self.op_timestep
                batch_num = data_veh.shape[0]

                ip_def: Optional[torch.Tensor] = None
                pred_def: Optional[torch.Tensor] = None
                ts_idx: Optional[List[int]] = None
                
                for i_iter in range(iter_num):
                    # Prepare input sequence
                    ip_def, timestamp = self.get_ip(
                        i_iter=i_iter,
                        ip_def=ip_def,
                        pred_def=pred_def,
                        stride=stride,
                        def_dist=self.def_dist,
                        batch_num=batch_num,
                        ip_timestep=self.ip_timestep,
                        op_timestep=self.op_timestep,
                        delta_t=self.delta_t,
                        data_def_tarnode_norm=data_def_tarnode_norm,
                        ts_idx=ts_idx,
                    )
                    
                    # Prepare timestep indices for prediction window
                    ts_idx = [i_iter * stride + 1, i_iter * stride + 1 + self.op_timestep]
                    delta_ts = ts_idx[1] - max_timestep
                    if delta_ts == 0:
                        break
                    if delta_ts > 0:
                        ts_idx[1] = int(max_timestep)
                        if delta_ts > (self.op_timestep * 9 / 10):
                            break

                    # Predict
                    pred_def = model(ip_def, data_veh[:, :, ts_idx[0] - 1], timestamp)
                    if delta_ts > 0:
                        # Truncate when the last segment is shorter than op_timestep
                        pred_def_ = pred_def[..., :-delta_ts]
                    else:
                        pred_def_ = pred_def

                    # Build mask to discard invalid timesteps per sample
                    keep_mask = torch.ones_like(pred_def_, dtype=torch.bool)
                    batch_size = pred_def_.shape[0]
                    for i_b in range(batch_size):
                        if ts_idx[1] > int(data_timestep[i_b]):
                            delta_ts_b = ts_idx[1] - int(data_timestep[i_b])
                            if delta_ts_b >= self.op_timestep:
                                keep_mask[i_b, :, :] = False
                            else:
                                keep_mask[i_b, :, -delta_ts_b:] = False

                    # Compute loss
                    weighted_sum, loss_log = get_loss(
                        ts_idx=ts_idx,
                        pred_def=pred_def_,
                        keep_mask=keep_mask,
                        def_dist=self.def_dist,
                        vel_dist=self.vel_dist,
                        acc_dist=self.acc_dist,
                        p_dist=self.p_dist,
                        delta_t=self.delta_t,
                        data_def=data_def[..., :max_timestep],
                        data_def_tarnode_norm=data_def_tarnode_norm[..., :max_timestep],
                        data_vel=data_vel[..., :max_timestep],
                        data_vel_tarnode_norm=data_vel_tarnode_norm[..., :max_timestep],
                        data_acc=data_acc[..., :max_timestep],
                        data_acc_tarnode_norm=data_acc_tarnode_norm[..., :max_timestep],
                        data_vehdis=data_vehdis[..., :max_timestep],
                        data_vehvel=data_vehvel[..., :max_timestep],
                        data_vehacc=data_vehacc[..., :max_timestep],
                        data_pgreff=data_pgreff[..., :max_timestep],
                        data_pgr=data_pgr[..., :max_timestep],
                        data_cgr=data_cgr[..., :max_timestep],
                        data_mgr=data_mgr[..., :max_timestep],
                        loss_log=loss_log,
                        loss_weight=self.loss_weight,
                    )

                    # Update parameters (causality weighting)
                    optimizer.zero_grad(set_to_none=True)
                    weighted_sum = weighted_sum * (iter_num - i_iter)
                    weighted_sum.backward()
                    optimizer.step()
                    
        return loss_log
    
    def get_ip(
        self,
        i_iter: int,
        ip_def: Optional[torch.Tensor],
        pred_def: Optional[torch.Tensor],
        stride: int,
        def_dist: List[torch.Tensor],
        batch_num: int,
        ip_timestep: int,
        op_timestep: int,
        delta_t: float,
        data_def_tarnode_norm: torch.Tensor,
        ts_idx: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input sequence and timestamps for the current iteration.

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
            ini_def = ini_def.to(self.device)

            noise = (1.0 / 1e6) * torch.randn(batch_num, self.node_num, ip_timestep, device=self.device)
            ip_def = noise + ini_def * torch.ones(batch_num, self.node_num, ip_timestep, device=self.device)

            timestamp_1d = torch.arange(0, ip_timestep, 1, dtype=torch.float64, device=self.device) * delta_t
        else:
            assert ts_idx is not None, "ts_idx must not be None for i_iter > 0."
            # Open-loop feeding using previously predicted and ground-truth sequences
            if ip_timestep > stride:
                ip_def = torch.cat(
                    (
                        ip_def[..., stride:],
                        data_def_tarnode_norm[..., ts_idx[0] : ts_idx[0] + stride],
                    ),
                    dim=-1,
                )
            else:
                ip_def = data_def_tarnode_norm[
                    ...,
                    (stride - ip_timestep - 1) + ts_idx[0] : (stride - ip_timestep - 1) + ts_idx[0] + ip_timestep,
                ]

            timestamp_1d = torch.arange(
                ts_idx[0],
                ts_idx[0] + ip_timestep,
                1,
                dtype=torch.float64,
                device=self.device,
            ) * delta_t

        timestamp = repeat(timestamp_1d, "t -> b k t", b=batch_num, k=1)
        return ip_def, timestamp
    
    
# ===================================================================== #
# Loss and helper functions
# ===================================================================== #
def get_loss(
    ts_idx: List[int],
    pred_def: torch.Tensor,
    keep_mask: torch.Tensor,
    def_dist: List[torch.Tensor],
    vel_dist: List[torch.Tensor],
    acc_dist: List[torch.Tensor],
    p_dist: torch.Tensor,
    delta_t: float,
    data_def: torch.Tensor,
    data_def_tarnode_norm: torch.Tensor,
    data_vel: torch.Tensor,
    data_vel_tarnode_norm: torch.Tensor,
    data_acc: torch.Tensor,
    data_acc_tarnode_norm: torch.Tensor,
    data_vehdis: torch.Tensor,
    data_vehvel: torch.Tensor,
    data_vehacc: torch.Tensor,
    data_pgreff: torch.Tensor,
    data_pgr: torch.Tensor,
    data_cgr: torch.Tensor,
    data_mgr: torch.Tensor,
    loss_log: Optional[torch.Tensor],
    loss_weight: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute total weighted loss and update loss log.

    Returns
    -------
    weighted_sum : torch.Tensor
        Scalar total loss.
    loss_log : torch.Tensor
        Updated log of data losses.
    """
    l1_loss = nn.L1Loss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")
    
    def get_geloss(
        ts_idx_local: List[int],
        pred_def_sb: torch.Tensor,
        pred_vel_sb: torch.Tensor,
        pred_acc_sb: torch.Tensor,
        keep_mask_local: torch.Tensor,
        data_def_local: torch.Tensor,
        data_vel_local: torch.Tensor,
        data_acc_local: torch.Tensor,
        data_vehdis_local: torch.Tensor,
        data_vehvel_local: torch.Tensor,
        data_vehacc_local: torch.Tensor,
        data_pgreff_local: torch.Tensor,
        data_pgr_local: torch.Tensor,
        data_cgr_local: torch.Tensor,
        data_mgr_local: torch.Tensor,
        delta_t_local: float,
        p_dist_local: torch.Tensor,
        lossfn,
    ) -> torch.Tensor:
        """
        Compute governing equation (GE) loss (residual).
        """
        batch = data_def.shape[0]
        sys_dof = int(math.sqrt(data_cgr.shape[1]))
                
        data_pgr_ = data_pgr[:,:,ts_idx[0]:ts_idx[1]].clone()
        data_mgr_ = data_mgr.reshape(batch, sys_dof, sys_dof, -1)[:,:,:,ts_idx[0]:ts_idx[1]].clone()
        data_cgr_ = data_cgr.reshape(batch, sys_dof, sys_dof, -1)[:,:,:,ts_idx[0]:ts_idx[1]].clone()
        
        data_def_ = data_def[:,:,ts_idx[0]:ts_idx[1]].clone()  
        data_vel_ = data_vel[:,:,ts_idx[0]:ts_idx[1]].clone()
        data_acc_ = data_acc[:,:,ts_idx[0]:ts_idx[1]].clone()
        
        # Replace bridge DOFs with predicted values
        data_def_[:, ::2, :] = data_def_[:, ::2, :] * 0. + pred_def
        data_vel_[:, ::2, 1:-1] = data_vel_[:, ::2, 1:-1] * 0. + pred_vel
        data_acc_[:, ::2, 1:-1] = data_acc_[:, ::2, 1:-1] * 0. + pred_acc
        
        # Remove boundary DOFs
        data_def_ = torch.cat((data_def_[:,1:-2,:], data_def_[:,-1:,:]), dim=1)
        data_vel_ = torch.cat((data_vel_[:,1:-2,:], data_vel_[:,-1:,:]), dim=1)
        data_acc_ = torch.cat((data_acc_[:,1:-2,:], data_acc_[:,-1:,:]), dim=1)
               
        # Assemble vehicle DOFs
        data_def_ = torch.cat((data_def_, data_vehdis[:,:,ts_idx[0]:ts_idx[1]]), dim=1)
        data_vel_ = torch.cat((data_vel_, data_vehvel[:,:,ts_idx[0]:ts_idx[1]]), dim=1)
        data_acc_ = torch.cat((data_acc_, data_vehacc[:,:,ts_idx[0]:ts_idx[1]]), dim=1)
                
        # Matrix multiplication in a Newmark-beta time-integration fashion
        m_mul = data_def_[:,:,:-1] * 4/delta_t/delta_t + \
                data_vel_[:,:,:-1] * 4/delta_t + \
                data_acc_[:,:,:-1]
        c_mul = data_def_[:,:,:-1] * 2/delta_t + \
                data_vel_[:,:,:-1]

        pred_pgreff = data_pgr_[:,:,1:] + \
                      torch.einsum('bijt,bjt->bit', data_mgr_[:,:,:,1:], m_mul) + \
                      torch.einsum('bijt,bjt->bit', data_cgr_[:,:,:,1:], c_mul) 
        
        # Generate mask (use only valid timesteps)
        keep_mask_local = repeat(
            keep_mask_local[:, 0, :],
            "b t -> b d t",
            d=pred_pgreff.shape[1],
        )[..., :-1]
        
        # Normalize equivalent force
        min_p = repeat(
            p_dist_local[:, 0],
            "d -> b d t",
            b=pred_pgreff.shape[0],
            t=pred_pgreff.shape[-1],
        )
        max_p = repeat(
            p_dist_local[:, 1],
            "d -> b d t",
            b=pred_pgreff.shape[0],
            t=pred_pgreff.shape[-1],
        )
        
        pred_pgreff = norm_data(pred_pgreff, min_p, max_p) * keep_mask_local
        pgreff = norm_data(
            data_pgreff_local[..., ts_idx_local[0] + 1 : ts_idx_local[1]],
            min_p,
            max_p,
        ) * keep_mask_local
        
        geloss_val = lossfn(pred_pgreff.to(pred_def_sb.dtype), pgreff.to(pred_def_sb.dtype))
        geloss_val = geloss_val.sum() / (keep_mask_local.sum() + 1e-8)
        return geloss_val
    
    # ------------------------------------------------------------------ #
    # Response loss (time-domain)
    # ------------------------------------------------------------------ #
    dataloss = mse_loss(
        pred_def * keep_mask,
        data_def_tarnode_norm[..., ts_idx[0] : ts_idx[1]] * keep_mask,
    )
    dataloss = dataloss.sum() / (keep_mask.sum() + 1e-8)
    
    # Denormalize displacement and compute velocity/acc via finite differences
    pred_def_sb = denorm_data(pred_def, def_dist[0], def_dist[1])
    pred_vel_sb = FDM(pred_def_sb, delta_t, order=1)
    pred_acc_sb = FDM(pred_def_sb, delta_t, order=2)
    pred_vel = norm_data(pred_vel_sb, vel_dist[0], vel_dist[1])
    pred_acc = norm_data(pred_acc_sb, acc_dist[0], acc_dist[1])

    pde_mask = keep_mask[..., 1:-1]
    pdeloss_vel = mse_loss(
        pred_vel * pde_mask,
        data_vel_tarnode_norm[..., ts_idx[0] + 1 : ts_idx[1] - 1] * pde_mask,
    )
    pdeloss_acc = mse_loss(
        pred_acc * pde_mask,
        data_acc_tarnode_norm[..., ts_idx[0] + 1 : ts_idx[1] - 1] * pde_mask,
    )
    pdeloss_vel = pdeloss_vel.sum() / (pde_mask.sum() + 1e-8)
    pdeloss_acc = pdeloss_acc.sum() / (pde_mask.sum() + 1e-8)
    
    # ------------------------------------------------------------------ #
    # Frequency-domain loss
    # ------------------------------------------------------------------ #
    tar_fft_def = torch.abs(
        torch.fft.rfft(
            data_def_tarnode_norm[..., ts_idx[0] : ts_idx[1]].clone() * keep_mask,
            dim=-1,
            norm="forward",
        )
    )
    pred_fft_def = torch.abs(
        torch.fft.rfft(pred_def * keep_mask, dim=-1, norm="forward")
    )
    freqloss_def = l1_loss(pred_fft_def, tar_fft_def)
    freqloss_def = freqloss_def.sum() / (
        keep_mask[..., : keep_mask.shape[-1] // 2 + 1].sum() + 1e-8
    )

    tar_fft_vel = torch.abs(
        torch.fft.rfft(
            data_vel_tarnode_norm[..., ts_idx[0] + 1 : ts_idx[1] - 1].clone() * pde_mask,
            dim=-1,
            norm="forward",
        )
    )
    pred_fft_vel = torch.abs(
        torch.fft.rfft(pred_vel * pde_mask, dim=-1, norm="forward")
    )
    freqloss_vel = l1_loss(pred_fft_vel, tar_fft_vel)
    freqloss_vel = freqloss_vel.sum() / (
        pde_mask[..., : pde_mask.shape[-1] // 2 + 1].sum() + 1e-8
    )

    tar_fft_acc = torch.abs(
        torch.fft.rfft(
            data_acc_tarnode_norm[..., ts_idx[0] + 1 : ts_idx[1] - 1].clone() * pde_mask,
            dim=-1,
            norm="forward",
        )
    )
    pred_fft_acc = torch.abs(
        torch.fft.rfft(pred_acc * pde_mask, dim=-1, norm="forward")
    )
    freqloss_acc = l1_loss(pred_fft_acc, tar_fft_acc)
    freqloss_acc = freqloss_acc.sum() / (
        pde_mask[..., : pde_mask.shape[-1] // 2 + 1].sum() + 1e-8
    )
    
    # ------------------------------------------------------------------ #
    # Boundary condition loss (at boundary nodes)
    # ------------------------------------------------------------------ #
    temp_mask_def = torch.cat((keep_mask[:, :1, :], keep_mask[:, -1:, :]), dim=1)
    bcloss_def = mse_loss(
        torch.cat((pred_def[:, :1, :], pred_def[:, -1:, :]), dim=1) * temp_mask_def,
        torch.cat(
            (
                data_def_tarnode_norm[:, :1, ts_idx[0] : ts_idx[1]],
                data_def_tarnode_norm[:, -1:, ts_idx[0] : ts_idx[1]],
            ),
            dim=1,
        )
        * temp_mask_def,
    )
    bcloss_def = bcloss_def.sum() / (temp_mask_def.sum() + 1e-8)

    temp_mask_pde = torch.cat((pde_mask[:, :1, :], pde_mask[:, -1:, :]), dim=1)
    bcloss_vel = mse_loss(
        torch.cat((pred_vel[:, :1, :], pred_vel[:, -1:, :]), dim=1) * temp_mask_pde,
        torch.cat(
            (
                data_vel_tarnode_norm[:, :1, ts_idx[0] + 1 : ts_idx[1] - 1],
                data_vel_tarnode_norm[:, -1:, ts_idx[0] + 1 : ts_idx[1] - 1],
            ),
            dim=1,
        )
        * temp_mask_pde,
    )
    bcloss_vel = bcloss_vel.sum() / (temp_mask_pde.sum() + 1e-8)

    bcloss_acc = mse_loss(
        torch.cat((pred_acc[:, :1, :], pred_acc[:, -1:, :]), dim=1) * temp_mask_pde,
        torch.cat(
            (
                data_acc_tarnode_norm[:, :1, ts_idx[0] + 1 : ts_idx[1] - 1],
                data_acc_tarnode_norm[:, -1:, ts_idx[0] + 1 : ts_idx[1] - 1],
            ),
            dim=1,
        )
        * temp_mask_pde,
    )
    bcloss_acc = bcloss_acc.sum() / (temp_mask_pde.sum() + 1e-8)
    
    # ------------------------------------------------------------------ #
    # Initial/terminal consistency loss between iterations
    # ------------------------------------------------------------------ #
    bound_limit = 16
    if bound_limit > pde_mask.shape[-1]:
        bound_limit = pde_mask.shape[-1] // 2

    icloss_def = mse_loss(
        torch.cat(
            (
                pred_def[..., :bound_limit] * keep_mask[..., :bound_limit],
                pred_def[..., -bound_limit:] * keep_mask[..., -bound_limit:],
            ),
            dim=-1,
        ),
        torch.cat(
            (
                data_def_tarnode_norm[..., ts_idx[0] : ts_idx[0] + bound_limit] * keep_mask[..., :bound_limit],
                data_def_tarnode_norm[..., ts_idx[1] - bound_limit : ts_idx[1]] * keep_mask[..., -bound_limit:],
            ),
            dim=-1,
        ),
    )
    icloss_def = icloss_def.sum() / (
        keep_mask[..., :bound_limit].sum()
        + keep_mask[..., -bound_limit:].sum()
        + 1e-8
    )

    icloss_vel = mse_loss(
        torch.cat(
            (
                pred_vel[..., :bound_limit] * pde_mask[..., :bound_limit],
                pred_vel[..., -bound_limit:] * pde_mask[..., -bound_limit:],
            ),
            dim=-1,
        ),
        torch.cat(
            (
                data_vel_tarnode_norm[..., ts_idx[0] + 1 : ts_idx[0] + 1 + bound_limit]
                * pde_mask[..., :bound_limit],
                data_vel_tarnode_norm[..., ts_idx[1] - 1 - bound_limit : ts_idx[1] - 1]
                * pde_mask[..., -bound_limit:],
            ),
            dim=-1,
        ),
    )
    icloss_vel = icloss_vel.sum() / (
        pde_mask[..., :bound_limit].sum()
        + pde_mask[..., -bound_limit:].sum()
        + 1e-8
    )

    icloss_acc = mse_loss(
        torch.cat(
            (
                pred_acc[..., :bound_limit] * pde_mask[..., :bound_limit],
                pred_acc[..., -bound_limit:] * pde_mask[..., -bound_limit:],
            ),
            dim=-1,
        ),
        torch.cat(
            (
                data_acc_tarnode_norm[..., ts_idx[0] + 1 : ts_idx[0] + 1 + bound_limit]
                * pde_mask[..., :bound_limit],
                data_acc_tarnode_norm[..., ts_idx[1] - 1 - bound_limit : ts_idx[1] - 1]
                * pde_mask[..., -bound_limit:],
            ),
            dim=-1,
        ),
    )
    icloss_acc = icloss_acc.sum() / (
        pde_mask[..., :bound_limit].sum()
        + pde_mask[..., -bound_limit:].sum()
        + 1e-8
    )
    
    # ------------------------------------------------------------------ #
    # Governing equation residual
    # ------------------------------------------------------------------ #
    geloss = get_geloss(
        ts_idx_local=ts_idx,
        pred_def_sb=pred_def_sb,
        pred_vel_sb=pred_vel_sb,
        pred_acc_sb=pred_acc_sb,
        keep_mask_local=keep_mask,
        data_def_local=data_def,
        data_vel_local=data_vel,
        data_acc_local=data_acc,
        data_vehdis_local=data_vehdis,
        data_vehvel_local=data_vehvel,
        data_vehacc_local=data_vehacc,
        data_pgreff_local=data_pgreff,
        data_pgr_local=data_pgr,
        data_cgr_local=data_cgr,
        data_mgr_local=data_mgr,
        delta_t_local=delta_t,
        p_dist_local=p_dist,
        lossfn=mse_loss,
    )
    
    # ------------------------------------------------------------------ #
    # Weighted sum of all loss terms
    # ------------------------------------------------------------------ #
    weighted_terms = [
        (loss_weight[0] * dataloss).reshape(1),
        (loss_weight[1] * freqloss_def).reshape(1),
        (loss_weight[2] * bcloss_def).reshape(1),
        (loss_weight[3] * pdeloss_vel).reshape(1),
        (loss_weight[4] * freqloss_vel).reshape(1),
        (loss_weight[5] * bcloss_vel).reshape(1),
        (loss_weight[6] * pdeloss_acc).reshape(1),
        (loss_weight[7] * freqloss_acc).reshape(1),
        (loss_weight[8] * bcloss_acc).reshape(1),
        (loss_weight[9] * geloss).reshape(1),
    ]
    weighted_sum = torch.cat(weighted_terms).sum()
    
    # Log the main data loss
    temp = torch.tensor([dataloss.item()])
    if loss_log is None:
        loss_log = temp
    else:
        loss_log = torch.cat((loss_log, temp))

    return weighted_sum, loss_log


# ===================================================================== #
# Utilities
# ===================================================================== #
def FDM(x: torch.Tensor, delta_t: float, order: int = 1) -> torch.Tensor:
    """
    Finite difference approximation for temporal derivatives.
    """
    if order == 1:
        derivative = (x[..., 2:] - x[..., :-2]) / (2.0 * delta_t)  # symmetric derivative
    elif order == 2:
        derivative = (x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]) / (delta_t ** 2)
    else:
        raise ValueError("Finite difference only supports order 1 or 2.")
    return derivative
    

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
    
    
    
# ===================================================================== #
# Script entry
# ===================================================================== #
if __name__ == "__main__":
    main()
    print("All done!")
    
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from torchmetrics import AUROC
from .utils import make_cdist_mask, mask_hidden_state, align_conformer_hat_to_conformer, make_mask_for_pyd_batch_graph
from torch_geometric.typing import Adj, OptTensor, SparseTensor
import itertools


def get_soft_label(distance, threshold=5.0, width=0.5):
    return 1 / (1 + torch.exp((distance - threshold) / width))


class ConformerPredictionHead(nn.Module):
    def __init__(self, hidden_X_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_X_dim, hidden_X_dim * 3), nn.GELU(), nn.Linear(hidden_X_dim * 3, 3))
        self.lin = nn.Linear(hidden_X_dim, 1)
        
    def forward(self, conformer: torch.Tensor, hidden_X: torch.Tensor, padding_mask: torch.Tensor = None, 
                compute_loss: bool = True, conformer_base: torch.Tensor = None) -> torch.Tensor:
        
        if conformer_base is not None:
            hidden_X = torch.stack((hidden_X, conformer_base), -1)
            attn_score = self.lin(hidden_X.transpose(3, 2)).squeeze(-1) # (B, N, 2)
            scale = hidden_X.shape[2] ** 0.5
            attn_score = attn_score / torch.tensor(scale)
            attn_score = torch.softmax(attn_score, dim=-1).unsqueeze(2)
            hidden_X = torch.sum(attn_score * hidden_X, dim=-1)
        
        conformer_hat = self.head(hidden_X)
        conformer_hat = mask_hidden_state(conformer_hat, padding_mask)
        
        conformer_hat = align_conformer_hat_to_conformer(conformer_hat, conformer)
        conformer_hat = mask_hidden_state(conformer_hat, padding_mask)
        
        if not compute_loss:
            return conformer_hat
        
        result_dict = self._compute_loss(conformer, conformer_hat, padding_mask)
            
        return result_dict

    def _compute_loss(self, conformer: torch.Tensor, conformer_hat: torch.Tensor, 
                      padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Convenient for design loss function in the future.
        cdist, cdist_hat = torch.cdist(conformer, conformer), torch.cdist(conformer_hat, conformer_hat)
        c_dist_mask = make_cdist_mask(padding_mask)
        cdist, cdist_hat = cdist * c_dist_mask, cdist_hat * c_dist_mask
        cdist_mae, loss = self._compute_cdist_mae(cdist, cdist_hat, c_dist_mask)
        cdist_mse = self._compute_cdist_mse(cdist, cdist_hat, c_dist_mask)
        coord_rmsd = self._compute_conformer_rmsd(conformer, conformer_hat, padding_mask)

        return {
            "loss": loss,
            "cdist_mae": cdist_mae.detach(),
            "cdist_mse": cdist_mse.detach(),
            "coord_rmsd": coord_rmsd.detach(),
            "conformer": conformer.detach(),
            "conformer_hat": conformer_hat.detach(),
        }

    @staticmethod
    def _compute_cdist_mae(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, 
                           cdist_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean absolute error of conformer and conformer_hat.

        Args:
            - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
            - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
            - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.

        Returns:
            torch.Tensor: The mean absolute error of conformer and conformer_hat.
        """
        loss = F.l1_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
        mae = loss.clone()
        return mae, loss

    @staticmethod
    def _compute_cdist_mse(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, 
                           cdist_mask: torch.Tensor) -> torch.Tensor:
        """Compute root mean squared error of conformer and conformer_hat.

        Args:
            - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
            - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
            - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.

        Returns:
            torch.Tensor: The root mean squared error of conformer and conformer_hat.
        """
        mse = F.mse_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
        return mse

    @staticmethod
    def _compute_conformer_rmsd(masked_conformer: torch.Tensor, masked_conformer_hat: torch.Tensor, 
                                padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute root mean squared deviation of conformer and conformer_hat.

        Args:
            - masked_conformer (torch.Tensor): A torch tensor of shape (b, l, 3), which denotes the coordinate of the conformer.
            - masked_conformer_hat (torch.Tensor): A torch tensor of shape (b, l, 3), which denotes the coordinate of the conformer_hat.
            - padding_mask (torch.Tensor): A torch tensor of shape (b, l), which denotes the mask of the conformer.

        Returns:
            torch.Tensor: The root mean squared deviation of conformer and conformer_hat.
        """
        R, R_h, M = masked_conformer, masked_conformer_hat, padding_mask
        delta = (R - R_h).to(torch.float32)
        point_2_norm = torch.norm(delta, p=2, dim=-1)
        MSD = torch.sum(point_2_norm**2, dim=-1) / torch.sum(M, dim=-1)
        RMSD = torch.sqrt(MSD)
        return RMSD.mean()


class GNNConformerPredictionHead(nn.Module):
    def __init__(self, hidden_X_dim: int = 300) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_X_dim, hidden_X_dim * 3), nn.ReLU(), nn.Linear(hidden_X_dim * 3, 3))
        
    def forward(self, conformer: torch.Tensor, hidden_X: torch.Tensor, batch: torch.Tensor):
        conformer_hat = self.head(hidden_X)
        return self._compute_loss(conformer, conformer_hat, batch)

    def _compute_loss(self, conformer: torch.Tensor, conformer_hat: torch.Tensor, batch: torch.Tensor):
        cdist = torch.cdist(conformer, conformer)
        cdist_hat = torch.cdist(conformer_hat, conformer_hat)
        M = make_mask_for_pyd_batch_graph(batch)
        masked_cdist, masked_cdist_hat = cdist * M, cdist_hat * M # only consider pairs in the same molecule
        mae = self._compute_cdist_mae(masked_cdist, masked_cdist_hat, M)
        mse = self._compute_cdist_mse(masked_cdist, masked_cdist_hat, M)
        num_samples = len(torch.unique(batch))
        coord_rmsd = self._compute_conformer_rmsd(conformer, conformer_hat, batch, num_samples)
        loss = mae

        return {
            "loss": loss,
            "cdist_mae": mae.detach(),
            "cdist_mse": mse.detach(),
            "coord_rmsd": coord_rmsd.detach(),
            "conformer": conformer.detach(),
            "conformer_hat": conformer_hat.detach(),
        }

    @staticmethod
    def _compute_cdist_mae(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        D, D_h, M = masked_cdist, masked_cdist_hat, cdist_mask
        mae = F.l1_loss(D, D_h, reduction="sum") / M.sum()
        return mae

    @staticmethod
    def _compute_cdist_mse(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        D, D_h, M = masked_cdist, masked_cdist_hat, cdist_mask
        mse = F.mse_loss(D, D_h, reduction="sum") / M.sum()
        return mse
    
    @staticmethod
    def _compute_conformer_rmsd(masked_conformer: torch.Tensor, masked_conformer_hat: torch.Tensor, batch: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Compute root mean squared deviation of conformer and conformer_hat.

        Args:
            - masked_conformer (torch.Tensor): A torch tensor of shape (num_atoms, 3), which denotes the coordinate of molecules' conformer.
            - masked_conformer_hat (torch.Tensor): A torch tensor of shape (num_atoms, 3), which denotes the coordinate of molecules' conformer_hat.

        Returns:
            torch.Tensor: The root mean squared deviation of conformer and conformer_hat.
        """
        R, R_h = masked_conformer, masked_conformer_hat
        delta = (R - R_h).to(torch.float32)
        point_2_norm = torch.norm(delta, p=2, dim=-1)
        RMSD = [torch.sqrt(torch.sum(point_2_norm[batch==i]**2) / (batch==i).sum()) for i in range(num_samples)]
        RMSD = torch.tensor(RMSD)
        # MSD = torch.sum(point_2_norm**2, dim=-1) / point_2_norm.shape[0]
        # RMSD = torch.sqrt(MSD)
        return RMSD.mean()
    


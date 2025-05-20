"""
    This module contains different attention modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from torch import Tensor
from typing import Optional, Union, Tuple, List
from torch_geometric.nn.inits import glorot, zeros


class SelfAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop_out = nn.Dropout(dropout)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attention_mask: torch.Tensor = None, attention_bias: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute (multi-head) Self-Attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

        Args:
            - Q (torch.Tensor): Query, shape: (b, l, d) or (b, h, l, d)
            - K (torch.Tensor): Key, shape: (b, l, d) or (b, h, l, d)
            - V (torch.Tensor): Value, shape: (b, l, d) or (b, h, l, d)
            - attention_mask (torch.Tensor, optional): Attention mask, shape: (b, l) or (b, l, l), 1 for valid, 0 for invalid. Defaults to None.
            - attention_bias (torch.Tensor, optional): Attention bias, shape: (b, l, l) or (b, h, l, l). Defaults to None.

        Returns:
            torch.Tensor: Weighted sum of value, shape: (b, l, d) or (b, h, l, d)
        """
        scale = Q.shape[-1] ** 0.5
        # Q @ K.mT <==> torch.matmul(Q, K.mT)
        attention_score = (Q @ K.mT) / torch.tensor(scale)  # (b, l, l) | (b, h, l, l)
        if attention_bias is not None:
            attention_score += attention_bias
        if attention_mask is not None:
            attention_score = utils.mask_attention_score(attention_score, attention_mask)
        attention_weight = F.softmax(attention_score, dim=-1)  # (b, l, l) | (b, h, l, l)
        return self.drop_out(attention_weight) @ V  # (b, l, d) | (b, h, l, d)


class MultiRelationalSelfAttention(nn.Module):
    def __init__(self, num_heads: int = 1, dropout: float = 0.0, 
                 use_adjacency: bool = True, 
                 num_edge_type: int = 0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.use_adjacency = use_adjacency
        self.num_edge_type = num_edge_type
        self.drop_out = nn.Dropout(dropout)

        assert num_edge_type > 0
        self.weight_E = nn.ParameterDict()
        for typ in range(num_edge_type):
            weight_E = torch.empty(num_heads).view(1, num_heads, 1, 1)
            self.weight_E[str(typ)] = nn.Parameter(weight_E, requires_grad=True)

        if num_edge_type > 0:
            self.reset_parameters()

    def reset_parameters(self):
        for typ in range(self.num_edge_type):
            glorot(self.weight_E[str(typ)])
                
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor = None,
        adjacency_matrix: Union[Tensor, List[Tensor]] = None,
    ) -> torch.Tensor:
        M, A = attention_mask, adjacency_matrix
        assert type(A) == list
        A = [_A.unsqueeze(1) for _A in A]
        scale = Q.shape[-1] ** 0.5

        attn_score = Q @ K.mT  # (b, l, l) | (b, h, l, l)
        attn_score = utils.mask_attention_score(attn_score, M, 0.0) if M is not None else attn_score
        

        assert type(A) == list
        if self.num_edge_type > 0:
            for typ in range(self.num_edge_type):
                if typ == 0:
                    B_E = attn_score * (A[typ] * self.weight_E[str(typ)])
                else:
                    B_E += attn_score * (A[typ] * self.weight_E[str(typ)])
        else:
            B_E = None

        attn_score = attn_score + B_E if B_E is not None else attn_score
        attn_score = attn_score / torch.tensor(scale)  # (b, h, l, l) scaled by sqrt(d) after adding residual terms
        attention_weight = F.softmax(attn_score, dim=-1)  # (b, l, l) | (b, h, l, l)
        return {
            "out": self.drop_out(attention_weight) @ V,  # (b, l, d) | (b, h, l, d)
            "attn_weight": attention_weight.detach(),  # (b, l, l) | (b, h, l, l)
        }

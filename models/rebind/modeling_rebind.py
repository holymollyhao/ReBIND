"""
    This module contains self-defined GraphBert model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from .configuration_rebind import REBINDConfig

from ..modules import ConformerPredictionHead
from ..modules import ConformerPredictionOutput
from ..modules import MultiHeadAttention, AddNorm, PositionWiseFFN, Residual
from ..modules import AtomEmbedding
from transformers import PretrainedConfig, PreTrainedModel
from ..modules.utils import make_cdist_mask
from torch_geometric.data import Data


def retain_top_k(D_matrix, num_retain_edges, descending=False):
    B, N, _ = D_matrix.shape

    # Sort values in each row (descending order)
    sorted_values, sorted_indices = torch.sort(D_matrix, dim=2, descending=descending)

    # Create a mask for top-k values
    row_indices = torch.arange(N).unsqueeze(0).unsqueeze(2).expand(B, N, N).to(D_matrix.device)
    column_indices = torch.arange(N).unsqueeze(0).unsqueeze(1).expand(B, N, N).to(D_matrix.device)
    k_indices = num_retain_edges.unsqueeze(2).expand(B, N, N)

    mask = column_indices < k_indices

    # Apply the mask to the sorted values
    top_k_values = sorted_values * mask

    # Reorder the values back to their original positions
    reordered_top_k = torch.zeros_like(D_matrix)
    reordered_top_k.scatter_(2, sorted_indices, top_k_values)

    return reordered_top_k


class MultiRelationalBlock(nn.Module):
    def __init__(self, config: PretrainedConfig = None, encoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        
        if encoder:
            num_edge_type = 1 # adjacency only
        else:
            num_edge_type = 3 # force based
            
        self.multi_attention = MultiHeadAttention(
            d_q=getattr(config, "d_q", 256),
            d_k=getattr(config, "d_k", 256),
            d_v=getattr(config, "d_v", 256),
            d_model=getattr(config, "d_model", 256),
            n_head=getattr(config, "n_head", 8),
            qkv_bias=getattr(config, "qkv_bias", True),
            attn_drop=getattr(config, "attn_drop", 0.1),
            num_edge_type=num_edge_type,
        )
        self.add_norm01 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1),
                                  pre_ln=getattr(config, "pre_ln", True))
        self.position_wise_ffn = PositionWiseFFN(
            d_in=getattr(config, "d_model", 256),
            d_hidden=getattr(config, "d_ffn", 1024),
            d_out=getattr(config, "d_model", 256),
            dropout=getattr(config, "ffn_drop", 0.1),
        )
        self.add_norm02 = AddNorm(norm_shape=getattr(config, "d_model", 256), dropout=getattr(config, "norm_drop", 0.1),
                                  pre_ln=getattr(config, "pre_ln", True))

    def forward(self, **inputs):
        X, M = inputs.get("node_embedding"), inputs.get("node_mask")
        if self.encoder:
            A = [
                inputs.get("adjacency").type(torch.float32),
            ]
        else:
            A = [
                inputs.get("adjacency").type(torch.float32),
                inputs.get("attraction_adjacency").type(torch.float32),
                inputs.get("repulsion_adjacency").type(torch.float32),
            ]
        
        attn_out = self.multi_attention(X, X, X, attention_mask=M, adjacency_matrix=A)
        Y, attn_weight = attn_out["out"], attn_out["attn_weight"]
        X = self.add_norm01(X, Y)
        Y = self.position_wise_ffn(X)
        X = self.add_norm02(X, Y)
        return {
            "out": X,
            "attn_weight": attn_weight,
        }


class REBINDPreTrainedModel(PreTrainedModel):
    """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = REBINDConfig
    base_model_prefix = "Conformer"
    is_parallelizable = False
    # NOTE: version 1.0.0
    main_input_name = "node_attr"
    # NOTE: version 2.0.0

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


class Encoder(REBINDPreTrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.embed_style = getattr(config, "embed_style", "atom_type_ids")
        self.node_embedding = AtomEmbedding(atom_embedding_dim=getattr(config, "d_embed", 256), attr_reduction="sum")
        self.encoder_blocks = nn.ModuleList([MultiRelationalBlock(config, encoder=True) for _ in range(getattr(config, "n_encode_layers", 6))])
        self.__init_weights__()

    def forward(self, **inputs):
        node_attr = inputs.get("node_attr")
        node_embedding = self.node_embedding(node_attr)
        lap = inputs.get("lap_eigenvectors")
        node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap
        inputs["node_embedding"] = node_embedding

        attn_weight_dict = {}
        for i, encoder_block in enumerate(self.encoder_blocks):
            block_out = encoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"encoder_block_{i}"] = attn_weight

        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}


class Decoder(REBINDPreTrainedModel):
    def __init__(self, config: PretrainedConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        assert config is not None, f"config must be specified to build {self.__class__.__name__}"
        self.decoder_blocks = nn.ModuleList([MultiRelationalBlock(config, encoder=False) for _ in range(getattr(config, "n_decode_layers", 6))])
        self.__init_weights__()

    def forward(self, **inputs):
        node_embedding, lap = inputs.get("node_embedding"), inputs.get("lap_eigenvectors")
        node_embedding[:, :, : lap.shape[-1]] = node_embedding[:, :, : lap.shape[-1]] + lap
        inputs["node_embedding"] = node_embedding

        attn_weight_dict = {}
        for i, decoder_block in enumerate(self.decoder_blocks):
            block_out = decoder_block(**inputs)
            node_embedding, attn_weight = block_out["out"], block_out["attn_weight"]
            inputs["node_embedding"] = node_embedding
            attn_weight_dict[f"decoder_block_{i}"] = attn_weight
            
        return {"node_embedding": node_embedding, "attn_weight_dict": attn_weight_dict}
    

class REBIND(REBINDPreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.conformer_head = ConformerPredictionHead(hidden_X_dim=getattr(config, "d_model", 256))
        self.residual_head = ConformerPredictionHead(hidden_X_dim=getattr(config, "d_model", 256))
            
        self.__init_weights__()

    def forward(self, **inputs):
        conformer, node_mask = inputs.get("conformer"), inputs.get("node_mask")

        encoder_out = self.encoder(**inputs)
        node_embedding = encoder_out["node_embedding"] #, encoder_out["attn_weight_dict"]

        cache_out = self.conformer_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask,
                                        compute_loss=True)
        loss_cache, conformer_cache = cache_out["loss"], cache_out["conformer_hat"]
            
        D_cache, D_M = torch.cdist(conformer_cache, conformer_cache).detach(), make_cdist_mask(node_mask)
        inputs["pred_conformation"] = node_embedding
        inputs["node_embedding"] = node_embedding

        ##### Calculation of LJ force #####
        sigma, epsilon = inputs.get("sigma"), inputs.get("epsilon")

        ### energy potential
        # LJ_pot = 4 * epsilon * ((sigma / D_cache) ** 12 - (sigma / D_cache) ** 6)
        ## force (derivative of lj potential)
        s = sigma / D_cache
        s6 = s ** 6
        LJ_force_orig = 24 * epsilon * (s6 / D_cache) * (2 * s6 - 1)
        LJ_force = torch.abs(LJ_force_orig)

        adj_mask = inputs["adjacency"].bool()
        self_mask = torch.eye(D_cache.shape[1], device=D_cache.device).bool().unsqueeze(0).expand(D_cache.shape[0], -1, -1)
        LJ_force[adj_mask | self_mask] = 0
        LJ_force[~D_M.bool()] = 0

        num_retain_edges = inputs.get("num_near_edges")  # (B, N)

        ret_val = retain_top_k(LJ_force, num_retain_edges, descending=True)
        nonzeros = ret_val.nonzero(as_tuple=False)
        ret_val[nonzeros[:, 0], nonzeros[:, 1], nonzeros[:, 2]] = 1

        LJ_nonzero_mask = LJ_force > 0
        LJ_above_zero_mask = LJ_force_orig > 0
        LJ_below_zero_mask = LJ_force_orig < 0

        repulsive = LJ_nonzero_mask.float() * LJ_above_zero_mask.float() * ret_val * -1
        attractive = LJ_nonzero_mask.float() * LJ_below_zero_mask.float() * ret_val * 1
        inputs["attraction_adjacency"] = attractive
        inputs["repulsion_adjacency"] = repulsive
        
        del (adj_mask, nonzeros, self_mask, LJ_force, LJ_force_orig, LJ_nonzero_mask, 
                LJ_above_zero_mask, LJ_below_zero_mask, ret_val)
        torch.cuda.empty_cache()
            
        decoder_out = self.decoder(**inputs)
        node_embedding = decoder_out["node_embedding"] #, decoder_out["attn_weight_dict"]

        outputs = self.residual_head(conformer=conformer, hidden_X=node_embedding, padding_mask=node_mask, 
                                            compute_loss=True, conformer_base=inputs["pred_conformation"]) 


        return ConformerPredictionOutput(
            loss=(outputs["loss"] + loss_cache) / 2,
            cdist_mae=outputs["cdist_mae"],
            cdist_mse=outputs["cdist_mse"],
            coord_rmsd=outputs["coord_rmsd"],
            conformer=outputs["conformer"],
            conformer_hat=outputs["conformer_hat"],
        )


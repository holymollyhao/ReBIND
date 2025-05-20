"""
    This module contains self-defined ModelOutput classes based on transformers.modeling_outputs.BaseModelOutput.
"""

import torch

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput
    

@dataclass
class ConformerPredictionOutput(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    cdist_mae: Optional[torch.Tensor] = None
    cdist_mse: Optional[torch.Tensor] = None
    coord_rmsd: Optional[torch.Tensor] = None
    conformer: Optional[torch.Tensor] = None  # original conformer aligned to origin
    conformer_hat: Optional[torch.Tensor] = None  # predicted conformer aligned to origin


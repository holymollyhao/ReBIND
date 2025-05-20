"""
    This module contains useful functions for the modules.
"""
import torch
import torch.nn.functional as F


ALLOWABLE_FEATURES_VOCAB = {
    # This dictionary contains the allowable features for each node and edge.
    "possible_atomic_num": list(range(1, 119)) + ["misc"],
    "possible_chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
    "possible_degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic": [False, True],
    "possible_is_in_ring": [False, True],
    "possible_bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_dirs": ["NONE", "ENDUPRIGHT", "ENDDOWNRIGHT"],
    "possible_bond_stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "possible_is_conjugated": [False, True],
}


def get_atom_vocab_dims():
    """get the dimensions of the atom vocabulary.

    Returns:
        List[int]: A list of integers, which denotes the dimensions of the atom vocabulary.
    """
    dims_ls = list(
        map(
            len,
            [
                ALLOWABLE_FEATURES_VOCAB["possible_atomic_num"],
                ALLOWABLE_FEATURES_VOCAB["possible_chirality"],
                ALLOWABLE_FEATURES_VOCAB["possible_degree"],
                ALLOWABLE_FEATURES_VOCAB["possible_formal_charge"],
                ALLOWABLE_FEATURES_VOCAB["possible_numH"],
                ALLOWABLE_FEATURES_VOCAB["possible_number_radical_e"],
                ALLOWABLE_FEATURES_VOCAB["possible_hybridization"],
                ALLOWABLE_FEATURES_VOCAB["possible_is_aromatic"],
                ALLOWABLE_FEATURES_VOCAB["possible_is_in_ring"],
            ],
        )
    )
    return dims_ls


def get_bond_vocab_dims():
    """get the dimensions of the bond vocabulary.

    Returns:
        List[int]: A list of integers, which denotes the dimensions of the bond vocabulary.
    """
    dims_ls = list(
        map(
            len,
            [
                ALLOWABLE_FEATURES_VOCAB["possible_bond_type"],
                ALLOWABLE_FEATURES_VOCAB["possible_bond_dirs"],
                ALLOWABLE_FEATURES_VOCAB["possible_bond_stereo"],
                ALLOWABLE_FEATURES_VOCAB["possible_is_conjugated"],
            ],
        )
    )
    return dims_ls


def make_mask_for_pyd_batch_graph(batch: torch.Tensor):
    """
    Make a mask for the distance matrix of Pyg's big (disconnected) graph that consists of a batch of graphs.

    Args:
        batch: A torch tensor of shape (n,) like [0, 0, 0, 1, 1, 1, 1], denoting which graph each node belongs to.

    Returns:
        torch.Tensor: A torch tensor of shape (n, n), where n denotes the total number of nodes in this batch.
    """
    n = batch.shape[0]
    mask = torch.eq(batch.unsqueeze(1), batch.unsqueeze(0))
    mask = (torch.ones((n, n)) - torch.eye(n)).to(batch.device) * mask
    return mask


def valid_length_to_mask(valid_length: torch.TensorType, max_len: int = None):
    """Convert the valid length to mask

    Args:
        valid_length (torch.TensorType): The valid length of each sequence with shape (b,).

    Returns:
        torch.TensorType: The mask with shape (b, l).
    """
    max_len = valid_length.max() if max_len is None else max_len
    mask = (torch.arange(max_len)[None, :]) < valid_length[:, None]
    return mask.to(torch.float32)


def mask_attention_score(attention_score: torch.Tensor, attention_mask: torch.Tensor, mask_pos_value: float = 0.0):
    """Mask the attention score with the attention_mask || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

    Args:
        attention_score (torch.Tensor): The attention score with shape (b, h, l, l) or (b, l, l).
        attention_mask (torch.Tensor): The attention mask with shape (b, l) or (b, l, l).
        mask_pos_value (float, optional): The value of the position to be masked. Defaults to 0.0.
    Returns:
        torch.Tensor: The masked attention score with shape (b, h, l, l) or (b, l, l).
    """
    shape = attention_score.shape
    b, h, l, _ = shape if len(shape) == 4 else (shape[0], 1, shape[1], shape[2])
    # (b, l, l) -> (b, 1, l, l) if (b, l, l) else (b, h, l, l)
    attention_score = attention_score.view(b, h, l, l) if len(shape) == 3 else attention_score
    # (b, l) -> (b*h, l)
    attention_mask = attention_mask.repeat_interleave(h, dim=0)  # (b*h, l) or (b*h, l, l)
    attention_mask = attention_mask.view(b, h, -1, l)  # (b, h, 1, l) or (b, h, l, l)
    # attention_mask = (1 - attention_mask) * (-100000.0)
    # attention_score += attention_mask  # (b, h, l, l)
    attention_score = attention_score.masked_fill(attention_mask == mask_pos_value, -10000.0)
    return attention_score.view(shape) if len(shape) == 3 else attention_score


def mask_hidden_state(hidden_X: torch.Tensor, padding_mask: torch.Tensor = None):
    """Mask the hidden state with the padding mask || b: batch_size, l: seq_len, d: hidden_dims

    Args:
        hidden_X (torch.Tensor): The hidden state with shape (b, l, d)
        padding_mask (torch.Tensor, optional): The mask of each sequence with shape (b, l). Defaults to None.

    Returns:
        torch.Tensor: The masked hidden state with shape (b, l, d)
    """
    if padding_mask is None:
        return hidden_X
    # (b, l) -> (b, l, 1)
    padding_mask = padding_mask.unsqueeze(-1).to(torch.bool)
    return padding_mask * hidden_X


def make_cdist_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """Make mask for coordinate pairwise distance from padding mask.

    Args:
        padding_mask (torch.Tensor): Padding mask for the batched input sequences with shape (b, l).

    Returns:
        torch.Tensor: Mask for coordinate pairwise distance with shape (b, l, l).
    """
    padding_mask = padding_mask.unsqueeze(-1)
    mask = padding_mask * padding_mask.transpose(-1, -2)
    return mask


def align_conformer_to_origin(conformer: torch.Tensor):
    """Align the first atom of each molecule to the origin (x, y, z) -> (0, 0, 0).

    Args:
        - conformer (torch.Tensor): The conformer with shape (b, l, 3).

    Returns:
        torch.Tensor: The aligned conformer with shape (b, l, 3).
    """
    delta = conformer[:, 0, :]
    conformer_aligned = conformer - delta.unsqueeze(1)
    return conformer_aligned


def align_conformer_hat_to_conformer(conformer_hat, conformer):
    """Align conformer_hat to conformer using Kabsch algorithm.

    Args:
        - conformer_hat (torch.Tensor): The conformer to be aligned, with shape (b, l, 3).
        - conformer (torch.Tensor): The reference conformer, with shape (b, l, 3).

    Returns:
        - conformer_hat_aligned (torch.Tensor): The aligned conformer, with shape (b, l, 3).
    """
    # compute the mean of conformer_hat and conformer, and then center them
    p_mean = conformer_hat.mean(dim=1, keepdim=True)
    q_mean = conformer.mean(dim=1, keepdim=True)
    p_centered = conformer_hat - p_mean
    q_centered = conformer - q_mean
    # compute the rotation matrix using Kabsch algorithm
    H = torch.matmul(q_centered.transpose(1, 2), p_centered).float()  # shape: (b, 3, 3)
    U, S, V = torch.svd(H)  # shape: (b, 3, 3)
    R = torch.matmul(V, U.transpose(1, 2))  # shape: (b, 3, 3)
    # rotate p_centered using R
    p_rotated = torch.matmul(R, p_centered.transpose(1, 2))  # shape: (b, 3, l)
    p_rotated = p_rotated.transpose(1, 2)  # shape: (b, l, 3)
    # align p_rotated to the spatial position of q
    conformer_hat_aligned = p_rotated + q_mean  # shape: (b, l, 3)
    return conformer_hat_aligned


def get_mask_with_ratio(node_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Mask the original mask with ratio on the 1's.

    Args:
        - node_mask (torch.Tensor): the original node mask, 1 means the valid position and 0 means the padding position.
        - ratio (float): the ratio of the 1's to be masked.

    Returns:
        torch.Tensor: the re-masked node mask with ratio, 1 means the masked position and 0 means the valid position.
    """
    mask_hat = node_mask == 1
    mask_hat = mask_hat * (1 - ratio)
    mask_hat = torch.bernoulli(mask_hat)
    return mask_hat.to(torch.long)


def get_masked_atom_mask(mask_with_ratio: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Get the masked atom mask. 1 means the masked position and 0 means the valid position.

    Args:
        - mask_with_ratio (torch.Tensor): the re-masked node mask with ratio, 1 means the masked position and 0 means the valid position.
        - node_mask (torch.Tensor): the original node mask, 1 means the valid position and 0 means the padding position.

    Returns:
        torch.Tensor: the masked atom mask, 1 means the masked position and 0 means the valid position.
    """
    masked_atom_mask = node_mask * (1 - mask_with_ratio)
    return masked_atom_mask.to(torch.long)


def compute_distance_residual_bias(cdist: torch.Tensor, cdist_mask: torch.Tensor, raw_max: bool = True) -> torch.Tensor:
    b, l, _ = cdist.shape
    D = cdist * cdist_mask
    if not raw_max:
        D_max, _ = torch.max(D.view(b, -1), dim=-1)  # max value of each sample
        D = D_max.view(b, 1, 1) - D  # sample-max value subtract every value
    else:
        D_max, _ = torch.max(D, dim=-1)  # max value of every raw in each sample
        D = D_max.view(b, l, 1) - D  # raw-max value subtract every raw
    D.diagonal(dim1=-2, dim2=-1)[:] = 0  # set diagonal to 0
    D = D * cdist_mask
    return D


def get_sigma_and_epsilon(mol_data, drugs=False):
    lj_parameters = {
        0: {'sigma': 2.886, 'epsilon': 0.0440},  # H
        1: {'sigma': 2.362, 'epsilon': 0.0560},  # He
        2: {'sigma': 2.451, 'epsilon': 0.0250},  # Li
        3: {'sigma': 2.745, 'epsilon': 0.0850},  # Be
        4: {'sigma': 3.637, 'epsilon': 0.1800},  # B
        5: {'sigma': 3.431, 'epsilon': 0.1050},  # C
        6: {'sigma': 3.260, 'epsilon': 0.0690},  # N
        7: {'sigma': 3.118, 'epsilon': 0.0600},  # O
        8: {'sigma': 2.996, 'epsilon': 0.0500},  # F
        9: {'sigma': 2.889, 'epsilon': 0.0420},  # Ne
        10: {'sigma': 2.983, 'epsilon': 0.0300},  # Na
        11: {'sigma': 2.905, 'epsilon': 0.1110},  # Mg
        12: {'sigma': 4.008, 'epsilon': 0.5050},  # Al
        13: {'sigma': 3.826, 'epsilon': 0.4020},  # Si
        14: {'sigma': 3.694, 'epsilon': 0.3050},  # P
        15: {'sigma': 3.594, 'epsilon': 0.2740},  # S
        16: {'sigma': 3.516, 'epsilon': 0.2270},  # Cl
        17: {'sigma': 3.404, 'epsilon': 0.1850},  # Ar
        18: {'sigma': 3.812, 'epsilon': 0.0350},  # K
        19: {'sigma': 3.487, 'epsilon': 0.2380},  # Ca
        20: {'sigma': 3.316, 'epsilon': 0.0190},  # Sc
        21: {'sigma': 3.294, 'epsilon': 0.0170},  # Ti
        22: {'sigma': 3.273, 'epsilon': 0.0160},  # V
        23: {'sigma': 3.249, 'epsilon': 0.0150},  # Cr
        24: {'sigma': 3.210, 'epsilon': 0.0130},  # Mn
        25: {'sigma': 3.174, 'epsilon': 0.0130},  # Fe
        26: {'sigma': 3.144, 'epsilon': 0.0130},  # Co
        27: {'sigma': 3.116, 'epsilon': 0.0130},  # Ni
        28: {'sigma': 3.083, 'epsilon': 0.0050},  # Cu
        29: {'sigma': 3.002, 'epsilon': 0.1240},  # Zn
        30: {'sigma': 4.383, 'epsilon': 0.4150},  # Ga
        31: {'sigma': 4.310, 'epsilon': 0.3790},  # Ge
        32: {'sigma': 4.280, 'epsilon': 0.3090},  # As
        33: {'sigma': 4.336, 'epsilon': 0.2910},  # Se
        34: {'sigma': 4.403, 'epsilon': 0.2510},  # Br
        35: {'sigma': 4.463, 'epsilon': 0.2200},  # Kr
    }
    eps_list = []
    sig_list = []
    for (i, atom_id) in enumerate(mol_data.node_type):
        if drugs:
            id_to_type = drugs_types[atomic_number_to_symbol[atom_id.item()]]
            mol_data.node_attr[i, 0] = id_to_type
        else:
            id_to_type = atom_id.item()
        eps_list.append(lj_parameters[id_to_type]["epsilon"])
        sig_list.append(lj_parameters[id_to_type]["sigma"])
    return torch.Tensor(eps_list), torch.Tensor(sig_list)


drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

atomic_number_to_symbol = {1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg',
                           13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 23: 'V',
                           24: 'Cr', 25: 'Mn', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
                           35: 'Br', 47: 'Ag', 49: 'In', 51: 'Sb', 53: 'I', 64: 'Gd', 78: 'Pt', 79: 'Au',
                           80: 'Hg', 83: 'Bi'}
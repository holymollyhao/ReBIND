import torch
import numpy as np
from typing import Dict, List, Sequence

BOND_LENGTH_DICT = {
    (5, 5, "single"): 1.54,
    (5, 5, "double"): 1.34,
    (5, 5, "triple"): 1.20,
    (5, 5, "aromatic"): 1.39,
    (5, 8, "single"): 1.35,

    (0, 5, "single"): 1.09,
    (6, 7, "single"): 1.43,
    (6, 7, "double"): 1.23,
    (5, 7, "single"): 1.43,
    (5, 7, "double"): 1.23,
    (5, 6, "single"): 1.47,
    (5, 6, "double"): 1.28,
    (5, 6, "triple"): 1.16,
    (5, 8, "single"): 1.42,

    (0, 6, "single"): 1.01,
    (0, 7, "single"): 0.96,
    (6, 6, "single"): 1.45,
    (6, 6, "double"): 1.25,
    (6, 6, "triple"): 1.10,

    (7, 7, "single"): 1.48,

    (5, 6, "aromatic"): 1.34,  # e.g., C–N bond in pyridine
    (5, 5, "aromatic"): 1.39,  # e.g., C–C bond in benzene
    (6, 6, "aromatic"): 1.30,  # e.g., C–C bond in benzene
    (5, 7, "aromatic"): 1.37,
    (6, 7, "aromatic"): 1.23,
}

BOND_TYPES = ["single", "double", "triple", "aromatic"]


# Function to print bond distances based on atom types and bond types
def print_bond_distances(atom_types, bond_types, edge_list):
    bond_types = torch.argsort(bond_types, dim=-1, descending=True)[:, 0]
    # import pdb; pdb.set_trace()
    bond_len_list = []
    for i, (atom1_idx, atom2_idx) in enumerate(edge_list):
        atom1 = atom_types[atom1_idx].item()
        atom2 = atom_types[atom2_idx].item()
        sorted_atoms = tuple(sorted([atom1, atom2]))
        bond_type = bond_types[i]
        bond_len = BOND_LENGTH_DICT[(sorted_atoms[0], sorted_atoms[1], BOND_TYPES[bond_type])]
        bond_len_list.append(bond_len)
    return torch.Tensor(bond_len_list)


def stability_ratio(eigenvalues):
    n = len(eigenvalues)
    ratios = np.zeros(n - 1)  # There will be n-1 ratios since we calculate for each p

    for p in range(n - 1):
        # Numerator: |λ_p − λ_{p+1}|
        numerator = np.abs(eigenvalues[p] - eigenvalues[p + 1])

        # Denominator: min(1≤k≤p) |λ_k − λ_{k+1}|
        min_diff = np.min([np.abs(eigenvalues[k] - eigenvalues[k + 1]) for k in range(p + 1)])

        # Calculate stability ratio ρ_p
        ratios[p] = numerator / min_diff if min_diff != 0 else np.inf  # To handle division by zero

    return ratios


def zero_out_p_percent(tensor, p):
    # Find indices of non-zero elements
    non_zero_indices = torch.nonzero(tensor)

    # Determine the number of elements to zero out
    num_to_zero_out = int(len(non_zero_indices) * p)

    # Randomly select indices to zero out
    selected_indices = non_zero_indices[torch.randperm(len(non_zero_indices))[:num_to_zero_out]]

    # Zero out the selected elements
    for index in selected_indices:
        tensor[tuple(index)] = 0

    return tensor


def edge_indices_to_connectivity_matrix(edge_indices, num_nodes, is_directed=False):
    connectivity_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    for edge in edge_indices:
        if len(edge) == 3:
            i, j, weight = edge
        else:
            i, j = edge
            weight = 1  # Default weight if not provided

        connectivity_matrix[i, j] = weight
        if not is_directed:
            connectivity_matrix[j, i] = weight

    return connectivity_matrix


def matrix_accuracy(bond_matrix, matrix1, matrix2):
    # Create a mask where the bond_matrix is non-zero (elements to exclude)
    mask = bond_matrix == 0

    # Apply the mask to both matrices
    masked_matrix1 = np.where(mask, matrix1, np.nan)
    masked_matrix2 = np.where(mask, matrix2, np.nan)

    # Compare the elements of the two masked matrices
    equal_elements = np.equal(masked_matrix1, masked_matrix2)

    # Count the number of valid comparisons (excluding NaNs)
    valid_comparisons = np.sum(~np.isnan(masked_matrix1) & ~np.isnan(masked_matrix2))

    # Count the number of equal elements
    correct_matches = np.sum(equal_elements & ~np.isnan(masked_matrix1))

    # Calculate accuracy
    accuracy = correct_matches / valid_comparisons if valid_comparisons > 0 else np.nan

    return accuracy


def merge_edges_and_attributes(edge_list):
    # Concatenate edges and attributes
    all_edges = torch.cat(edge_list, dim=1).long()

    return all_edges


def valid_length_to_mask(valid_length: List[int], max_length: int = None) -> torch.Tensor:
    max_length = max(valid_length) if max_length is None else max_length
    mask = torch.zeros((len(valid_length), max_length), dtype=torch.long)
    for i, length in enumerate(valid_length):
        mask[i, :length] = 1
    return mask


def get_distance_weighted_adjacency(num_atoms: int, edge_indexes: List[List[int]], conformation) -> np.ndarray:
    assert len(edge_indexes) == 2 and len(edge_indexes[0]) == len(edge_indexes[1])

    # Calculate the distance between each pair of atoms
    coords = conformation
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    # Example adjacency matrix (same shape as dist_matrix)
    adjacency = np.zeros_like(dist_matrix)

    # Condition to select values from dist_matrix
    condition = dist_matrix < 5.0
    nonzero_mask = (dist_matrix != 0) & condition
    adjacency[nonzero_mask] = 5 / dist_matrix[nonzero_mask]

    # remove the bonds from the adjacency matrix
    adjacency[edge_indexes[0], edge_indexes[1]] = 5 / dist_matrix[edge_indexes[0], edge_indexes[1]]
    adjacency[edge_indexes[1], edge_indexes[0]] = 5 / dist_matrix[edge_indexes[1], edge_indexes[0]]
    return adjacency


def get_weighted_adjacency(num_atoms: int, edge_indexes: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
    assert len(edge_indexes) == 2 and len(edge_indexes[0]) == len(edge_indexes[1])
    adjacency = torch.zeros((num_atoms, num_atoms))
    adjacency[edge_indexes[0], edge_indexes[1]] = edge_weights
    return adjacency


def get_adjacency(num_atoms: int, edge_indexes: torch.Tensor) -> torch.Tensor:
    assert len(edge_indexes) == 2 and len(edge_indexes[0]) == len(edge_indexes[1])
    adjacency = torch.zeros((num_atoms, num_atoms))
    adjacency[edge_indexes[0], edge_indexes[1]] = 1
    return adjacency



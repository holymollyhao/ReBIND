"""
    Collator for Conformer.
"""

import torch
from typing import Dict, Sequence
from ..modules.utils import get_sigma_and_epsilon
from torch_geometric.data import Data
from rdkit import Chem
from ..modules.collating_utils import get_adjacency, valid_length_to_mask
from torch_geometric.utils import degree
import warnings
import numpy as np

warnings.simplefilter("ignore", UserWarning)


atomic_symbol_to_number = {
    'H': 0, 'C': 5, 'N': 6, 'O': 7, 'F': 8
}

atomic_number_to_symbol = {
    0: 'H', 5: 'C', 6: 'N', 7: 'O', 8: 'F'
}

bond_type_map = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3
}

def get_laplacian_eigenvectors(adjacency: np.ndarray) -> np.ndarray:
    A = adjacency
    l, _ = A.shape
    epsilon = 1e-8
    D = np.diag(1 / np.sqrt(A.sum(axis=1) + epsilon))
    L = np.eye(l) - D @ A @ D
    w, v = np.linalg.eigh(L)
    return v

class Collator:
    def __init__(self, max_nodes: int = None, max_edges: int = None) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.keys = None
        self.max_degree = None

    @torch.no_grad()
    def __call__(self, mol_sq: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        keys = mol_sq[0].keys()
        self.keys = keys

        mol_sq = [self._transform(self._get_pyg_data(mol)) for mol in mol_sq]

        num_mol = len(mol_sq)
        num_nodes = [len(mol.node_type) for mol in mol_sq]
        max_nodes = max(num_nodes) if self.max_nodes is None else self.max_nodes
        node_type = torch.zeros((num_mol, max_nodes), dtype=torch.long)
        
        lap_eigenvectors = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)
        adjacency = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)
        
        conformer = torch.zeros((num_mol, max_nodes, 3), dtype=torch.float)  # if "conformer" in keys else None
        labels = torch.tensor([mol.labels for mol in mol_sq], dtype=torch.float) if "labels" in keys else None
        node_attr = torch.zeros((num_mol, max_nodes, 9), dtype=torch.long) if "node_attr" in keys else None
        
        edge_attr = torch.zeros((num_mol, max_nodes, max_nodes, 4), dtype=torch.long)
        num_near_edges = torch.zeros((num_mol, max_nodes), dtype=torch.long)
        
        sigma = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)
        epsilon = torch.zeros((num_mol, max_nodes, max_nodes), dtype=torch.float32)


        for i, mol in enumerate(mol_sq):

            adj = get_adjacency(num_nodes[i], mol.edge_index)
            adjacency[i, :num_nodes[i], :num_nodes[i]] = adj
            lap_eigenvectors[i, : num_nodes[i], : num_nodes[i]] = torch.from_numpy(get_laplacian_eigenvectors(adj.numpy()))
            
            if hasattr(mol, "sigma") and hasattr(mol, "epsilon"):
                sigma[i, :num_nodes[i], :num_nodes[i]] = mol.sigma
                epsilon[i, :num_nodes[i], :num_nodes[i]] = mol.epsilon
                    
            node_type[i, : num_nodes[i]] = mol.node_type + 1  # 0 for padding
            
            if "conformer" in keys:
                conformer[i, : num_nodes[i]] = mol.conformer
            if "node_attr" in keys:
                node_attr[i, : num_nodes[i]] = mol.node_attr + 1  # 0 for padding
            if "num_near_edges" in keys:
                num_near_edges[i, : num_nodes[i]] = mol.num_near_edges

        res_dic = {
            "node_type": node_type,
            "node_mask": valid_length_to_mask(num_nodes, max_nodes),
            "adjacency": adjacency,
            "lap_eigenvectors": lap_eigenvectors,
            "edge_attr": edge_attr,
            "num_near_edges": num_near_edges,
            "sigma": sigma,
            "epsilon": epsilon,
        }

        if labels is not None:
            res_dic["labels"] = labels
        if conformer is not None:
            res_dic["conformer"] = conformer
        if node_attr is not None:
            res_dic["node_attr"] = node_attr
        
        return res_dic

    def _get_pyg_data(self, mol: Dict) -> Data:
        return Data(
            node_type=torch.tensor(mol["node_type"], dtype=torch.long),
            edge_index=torch.tensor(mol["edge_index"], dtype=torch.long),
            node_attr=torch.tensor(mol["node_attr"], dtype=torch.long) if "node_attr" in self.keys else None,
            edge_attr=torch.tensor(mol["edge_attr"], dtype=torch.long),
            conformer=torch.tensor(mol["conformer"], dtype=torch.float32),
            labels=torch.tensor(mol["labels"], dtype=torch.float32) if "labels" in self.keys else None,
        )

    def _transform(self, data: Data) -> Data:
        if not data.edge_attr.numel():
            data.edge_type = torch.empty((0), dtype=torch.long)
        elif data.edge_attr.dim() == 1:
            data.edge_type = data.edge_attr.view(-1)[0]
        else:
            data.edge_type = data.edge_attr[:, 0]

        d = degree(data.edge_index[1], num_nodes=len(data.node_type), dtype=torch.long)
        max_degree = d.max().item()
        num_near_edges = max_degree - d
        data.num_near_edges = num_near_edges

        epsilon, sigma = get_sigma_and_epsilon(data)
        data.epsilon = torch.sqrt((epsilon.unsqueeze(1) * epsilon.unsqueeze(0)))
        data.sigma = ((sigma.unsqueeze(1) + sigma.unsqueeze(0))/2)
        return data
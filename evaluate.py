import os
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn.functional as F

from tqdm import tqdm
from rdkit import Chem
from copy import deepcopy
from rdkit import RDLogger
from typing import Literal
from rdkit.Chem import AllChem
from pprint import pprint, pformat
from rdkit.Chem import rdchem, rdMolAlign
from rdkit.Chem.rdmolfiles import SDMolSupplier

from data.utils import mol_to_graph_dict
from models.rebind import Collator, REBIND

import random
import math
from collections import defaultdict


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def get_energy_weighted_rmsd(ref_mol, pred_mol, R, R_h):
    try:

        pred_mmff_props = AllChem.MMFFGetMoleculeProperties(pred_mol)
        ref_mmff_props = AllChem.MMFFGetMoleculeProperties(ref_mol)

        if pred_mmff_props is None or ref_mmff_props is None:
            raise ValueError
        else:

            pred_mmff_forcefield = AllChem.MMFFGetMoleculeForceField(pred_mol, pred_mmff_props)
            ref_mmff_forcefield = AllChem.MMFFGetMoleculeForceField(ref_mol, ref_mmff_props)

            # Calculate energy differences
            pred_energy = pred_mmff_forcefield.CalcEnergy()
            ref_energy = ref_mmff_forcefield.CalcEnergy()
            energy_diff = pred_energy - ref_energy

            if math.isnan(energy_diff):
                raise ValueError

            num_atoms = pred_mol.GetNumAtoms()
            pred_grad = pred_mmff_forcefield.CalcGrad()
            ref_grad = ref_mmff_forcefield.CalcGrad()

            pred_atom_grad = [pred_grad[i*3:(i+1)*3] for i in range(num_atoms)]
            ref_atom_grad = [ref_grad[i*3:(i+1)*3] for i in range(num_atoms)]

            pos_diff_list = []
            force_diff_list = []
            for i in range(num_atoms):
                grad = np.linalg.norm(np.array(pred_atom_grad[i]) - np.array(ref_atom_grad[i]))

                if np.isnan(grad):
                    force_diff_list.append(0)
                else:
                    force_diff_list.append(grad)

                pos_diff = np.linalg.norm(np.array(R[i]) - np.array(R_h[i]))
                pos_diff_list.append(pos_diff)

            k = 0.001987  # Boltzmann constant in kcal/mol/K
            T = 298.15  # Room temperature in Kelvin

            boltzmann = min(np.exp(energy_diff/(k * T)), 2.0)

            weights = num_atoms * np.array(force_diff_list) / np.sum(force_diff_list)
            if np.isnan(weights).any():
                weights = np.array(pos_diff_list)
            else:
                weights = weights * boltzmann
                weights = weights * np.array(pos_diff_list)
    except:
        num_atoms = pred_mol.GetNumAtoms()
        pos_diff_list = []
        for i in range(num_atoms):
            pos_diff = np.linalg.norm(np.array(R[i]) - np.array(R_h[i]))
            pos_diff_list.append(pos_diff)
        pos_diff_list = np.array(pos_diff_list)
        weights = pos_diff_list
        energy_diff = 0

    return energy_diff, weights



def get_molecule3d_supplier(root_dir: str, mode: Literal["random", "scaffold"] = "random", split: Literal["valid", "test"] = "test"):
    sdf_files = osp.join(root_dir, mode, f"{split}.sdf")
    sdf_files = os.path.expanduser(sdf_files)
    if not osp.exists(sdf_files):
        raise FileNotFoundError(f"{sdf_files} does not exist.")
    supplier = Chem.SDMolSupplier(sdf_files, removeHs=False, sanitize=True)
    return supplier


def get_qm9_supplier(root_dir: str, split: Literal["valid", "test"] = "test"):
    sdf_files = osp.join(root_dir, "gdb9.sdf")
    sdf_files = os.path.expanduser(sdf_files)
    if not osp.exists(sdf_files):
        raise FileNotFoundError(f"{sdf_files} does not exist.")
    supplier = Chem.SDMolSupplier(sdf_files, removeHs=False, sanitize=True)
    indices_df = pd.read_csv(osp.join(root_dir, f"{split}_indices.csv"))
    indices = indices_df["index"].tolist()
    supplier = [supplier[i] for i in indices]
    return supplier


def get_metrics(mol: rdchem.Mol, mol_h: rdchem.Mol = None, R: np.ndarray = None, R_h: np.ndarray = None, removeHs: bool = False):
    if mol_h is None and R_h is None:
        raise ValueError("mol_h and R_h cannot both be None.")
    if mol_h is not None and R_h is not None:
        raise ValueError("mol_h and R_h cannot both be not None.")
    if mol is None and R is not None:
        pass
    if mol_h is None and R_h is not None:
        mol_h = deepcopy(mol)
        R_h = R_h.tolist()
        conf_h = rdchem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf_h.SetAtomPosition(i, R_h[i])
        mol_h.RemoveConformer(0)
        mol_h.AddConformer(conf_h)
    
    R, R_h = mol.GetConformer().GetPositions(), mol_h.GetConformer().GetPositions()
    R, R_h = torch.from_numpy(R), torch.from_numpy(R_h) 
    D, D_h = torch.cdist(R, R), torch.cdist(R_h, R_h)
    
    mae = F.l1_loss(D, D_h, reduction="sum").item()
    mse = F.mse_loss(D, D_h, reduction="sum").item()
    num_dist = D.numel()
    if removeHs:
        try:
            mol, mol_h = Chem.RemoveHs(mol), Chem.RemoveHs(mol_h)
        except Exception as e:
            pass
    rmsd = rdMolAlign.GetBestRMS(mol, mol_h)
    
    rdMolAlign.AlignMol(mol_h, mol)
    R = mol.GetConformer().GetPositions()
    R_h = mol_h.GetConformer().GetPositions()
    e_diff, e_weights  = get_energy_weighted_rmsd(mol, mol_h, R, R_h)
    
    return {
        "mae": mae,
        "mse": mse,
        "rmsd": rmsd,
        "num_dist": num_dist,
        "weighted_pos_diff_per_atom": e_weights,
    }


def evaluate(
    model: REBIND,
    collator: Collator,
    supplier: SDMolSupplier,
    batch_size: int = 300,
    removeHs: bool = False,
):
    num_mol = len(supplier)
    num_batch = num_mol // batch_size + 1
    model.eval()
    total_mae, total_mse, total_dist, total_rmsd = 0.0, 0.0, 0.0, 0.0
    total_ewrmsd = 0.0
    inf_pro_bar = tqdm(total=num_batch, desc="Inference on REBIND", ncols=100, leave=False)
    eval_pro_bar = tqdm(total=num_mol, desc="Evaluation", ncols=100, leave=False)
    num_mol = 0
    for i in range(num_batch):
        start, end = (i * batch_size, (i + 1) * batch_size) if i < num_batch - 1 else (i * batch_size, num_mol)
        mol_ls = [supplier[j] for j in range(start, end) if supplier[j] is not None]
        if not mol_ls:
            continue
        num_mol += len(mol_ls)
        mol_dict_ls = []
        for mol in mol_ls:
            mol_dict = mol_to_graph_dict(mol)
            mol_dict_ls.append(mol_dict)
        batch = collator(mol_dict_ls)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        padding_mask = batch["node_mask"]
        R_h = out["conformer_hat"]
        batch_rmsd = 0.0
        # compute coord rmsd
        for j, mol in enumerate(mol_ls):
            R_h_i = R_h[j]
            mask = padding_mask[j]
            R_h_i = R_h_i[mask == 1].detach().cpu().numpy()
            mol_gt = deepcopy(mol)
            metrics = get_metrics(mol=mol_gt, R_h=R_h_i, removeHs=removeHs)
            mae, mse, rmsd, num_dist = metrics["mae"], metrics["mse"], metrics["rmsd"], metrics["num_dist"]
            ewrmsd = np.mean(metrics["weighted_pos_diff_per_atom"])
            total_mae += mae
            total_mse += mse
            total_dist += num_dist
            total_rmsd += rmsd
            batch_rmsd += rmsd
            total_ewrmsd += ewrmsd

            eval_pro_bar.set_postfix({"mae": f"{mae/num_dist:.3f}", "rmse": f"{np.sqrt(mse/num_dist):.3f}", "rmsd": f"{rmsd:.3f}"})
            eval_pro_bar.update()

        inf_pro_bar.set_postfix({"batch": i, "rmsd": f"{batch_rmsd/len(mol_ls):.3f}"})
        inf_pro_bar.update()
        
    mae = total_mae / total_dist
    rmse = np.sqrt(total_mse / total_dist)
    rmsd = total_rmsd / num_mol
    ewrmsd = total_ewrmsd / num_mol

    return {"mae": mae, "rmse": rmse, "rmsd": rmsd, "ewrmsd": ewrmsd}



if __name__ == "__main__":
    parse = argparse.ArgumentParser("Conformation Prediction Evaluation")
    parse.add_argument("--data_dir", type=str, default="~/DataSets/", help="The root directory of the dataset.")
    parse.add_argument("--dataset", type=str, default="Molecule3D", choices=["Molecule3D", "QM9", "DRUGS"], help="The dataset to be evaluated.")
    parse.add_argument("--mode", type=str, default="random", choices=["random", "scaffold"], help="The mode of Molecule3D.")
    parse.add_argument("--split", type=str, default="test", choices=["valid", "test"], help="The split of Molecule3D.")
    parse.add_argument("--seed", type=int, default=42, help="The random seed.")
    parse.add_argument("--log_file", type=str, default="./evaluate.txt", help="The log file to save the evaluation results.")
    parse.add_argument("--removeHs", action="store_true", help="Whether to remove Hs.")
    parse.add_argument(
        "--REBIND_checkpoint", type=str, default="./checkpoints/CP/REBIND_Molecule3D_Random", help="The checkpoint of REBIND."
    )
    parse.add_argument("--device", type=str, default="cuda:1", help="The device to run evaluation on.")
    parse.add_argument("--batch_size", type=int, default=100, help="The batch size to run evaluation.")
    parse.add_argument("--pe_type", type=str, default="laplacian")
    parse.add_argument("--log_results", type=int, default=1, help="log performance in .txt file")
    parse.add_argument("--pe", action="store_true", help="laplacian positional encoding for multi-relational gnn")
    
    args = vars(parse.parse_args())
    pprint(args)

    # make log file
    os.makedirs(osp.dirname(args["log_file"]), exist_ok=True)
    init_seed(args["seed"])
    

    prox_collator, prox_predictor = None, None

    # get supplier
    supplier = None
        
    if args["dataset"] == "QM9":
        root_dir = osp.join(args["data_dir"], "QM9")
        supplier = get_qm9_supplier(root_dir=root_dir, split=args["split"])
    else:
        raise ValueError(f"Unknown dataset: {args['dataset']}")

    collator = Collator()
    model = REBIND.from_pretrained(args["REBIND_checkpoint"]).to(device)
    print(model.config)
    
    # evaluate
    metrics = evaluate(
        model=model,
        collator=collator,
        supplier=supplier,
        batch_size=args["batch_size"],
        removeHs=args["removeHs"],
    )
    
    info = f"\nmae: {metrics['mae']:.4f}\n rmse: {metrics['rmse']:.4f}\n rmsd: {metrics['rmsd']:.4f}\n , ewrmsd: {metrics['ewrmsd']:.4f}\n"
    info += pformat(args)
    print(info)

    if args["log_results"]:
        with open(args["log_file"], "a") as f:
            f.write(info)
            f.write("\n")

# https://github.com/atomicarchitects/symphony/blob/eacce5a905081c5f0b47641a4d012c1c489f8391/analyses/metrics.py#L216C1-L230C16

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger

def check_validity_smiles(M):
    """Checks whether a molecule is valid using xyz2mol."""
    mol = Chem.MolFromXYZBlock(M.xyzfile)

    # We should only have one conformer.
    if mol.GetNumConformers() != 1:
        return False, None

    try:
        RDLogger.DisableLog('rdApp.*')
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        smi = Chem.MolToSmiles(mol)
        RDLogger.EnableLog('rdApp.*')
    except:
        return False, None

    if mol.GetNumBonds() == 0:
        return False, None

    return True, smi

### IMPORTANT: validity depends on rdkit version
### when calculating validity on the 100k training set of QM9:
### rdkit==2023.03.3 gives 99.99%
### later versions rdkit>2023.03.3 give 94.78%, tested up to rdkit=2024.09.4
def compute_valid_unique(Ms, return_invalid=False):
    assert type(Ms) is list

    num_samples = len(Ms)
    num_valid = 0
    all_smiles = []
    invalid_idxs = []
    for i, M in enumerate(Ms):
        valid, smiles = check_validity_smiles(M)
        if valid:
            num_valid += 1
            all_smiles.append(smiles)
        elif return_invalid:
            invalid_idxs.append(i)

    validity = num_valid / num_samples
    uniqueness = len(set(all_smiles)) / num_valid if num_valid > 0 else 0

    if return_invalid:
        return validity, uniqueness, invalid_idxs
    return validity, uniqueness

import torch
import scipy.optimize
import numpy as np

def compute_rmsd(atoms_pred, atoms_true, coords_pred, coords_true):
    # assumes numpy arrays

    if atoms_pred.shape[0] != atoms_true.shape[0]:
        return False, np.inf

    costs = np.square(coords_pred[:, np.newaxis, :] - coords_true[np.newaxis, :, :]).sum(axis=-1)
    costs = np.where(atoms_pred[:, np.newaxis] == atoms_true[np.newaxis, :], costs, np.inf)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    rmsd = np.sqrt(np.mean(costs[row_ind, col_ind]))

    return True, rmsd

from edm_metrics import edm_metrics
import os
import json
def evaluate(samples_dir, dataset):
    samples_path = os.path.join(samples_dir, "gen.pt")
    all_out = torch.load(samples_path, weights_only=False)
    
    samples = sum([s[0].unbatch() for s in all_out], [])

    valid, unique, invalid_idxs = compute_valid_unique(samples, return_invalid=True)
    print(f"Validity: {valid:.3f}")
    print(f"Uniqueness: {unique:.3f}")
    print(f"Validity * Uniqueness: {valid * unique:.3f}")

    invalid_path = os.path.join(samples_dir, "invalid_idxs.json")
    with open(invalid_path, "w") as f:
        json.dump(invalid_idxs, f, indent=4)

    atom_stability, mol_stability, edm_validity, edm_valid_uniq, edm_invalid_idxs = edm_metrics(samples, dataset)

    print(f"Atom stability: {atom_stability:.3f}")
    print(f"Mol stability: {mol_stability:.3f}")
    print(f"EDM validity: {edm_validity:.3f}")
    print(f"EDM validity * uniqueness: {edm_valid_uniq:.3f}")

    edm_invalid_path = os.path.join(samples_dir, "edm_invalid_idxs.json")
    with open(edm_invalid_path, "w") as f:
        json.dump(edm_invalid_idxs, f, indent=4)
    
    sample_metrics = {
        "validity": valid,
        "uniqueness": unique,
        "valid_uniq": valid * unique,
        "atom_stability": atom_stability,
        "mol_stability": mol_stability,
        "edm_validity": edm_validity,
        "edm_valid_uniq": edm_valid_uniq
    }

    metrics_path = os.path.join(samples_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(sample_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.samples_dir, args.dataset)

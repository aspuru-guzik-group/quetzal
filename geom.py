# https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/build_geom_dataset.py
import os
import msgpack
from qm9 import download_url
import numpy as np
import tqdm
from torch.utils.data import Dataset
import torch
from chem import GEN, STOP

def extract_conformers():

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    for i, drugs_1k in enumerate(unpacker):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            all_smiles.append(smiles)
            conformers = all_info['conformers']
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:conformations]
            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer['xyz']).astype(float)        # n x 4
                n = coords.shape[0]
                all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                dataset_conformers.append(id_coords)
                mol_id += 1

    print("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    print("Total number of atoms in the dataset", dataset.shape[0])
    print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    np.save(save_file, dataset)
    # Save SMILES
    with open(smiles_list_file, 'w') as f:
        for s in all_smiles:
            f.write(s)
            f.write('\n')

    # Save number of atoms per conformation
    np.save(number_atoms_file, all_number_atoms)
    print("Dataset processed.")


def load_split_data():
    val_proportion = 0.1
    test_proportion = 0.1

    all_data = np.load(save_file)  # 2d array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)

    # https://github.com/atomicarchitects/symphony/blob/eacce5a905081c5f0b47641a4d012c1c489f8391/symphony/data/datasets/geom_drugs.py
    permutation = np.load(geom_permutation)

    num_mol = len(permutation)
    val_proportion = 0.1
    val_split = int(num_mol * val_proportion)
    test_proportion = 0.1
    test_split = val_split + int(num_mol * test_proportion)
    val_indices, test_indices, train_indices = np.split(
        permutation, [val_split, test_split]
    )

    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    return train_data, val_data, test_data

########

DRUGS_URL = "https://dataverse.harvard.edu/api/access/datafile/4360331"
GEOM_PERMUTATION_URL = "https://github.com/ehoogeboom/e3_diffusion_for_molecules/raw/fce07d701a2d2340f3522df588832c2c0f7e044a/data/geom/geom_permutation.npy"

root_dir = "data"
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

conformations = 30

tar_save_path = os.path.join(root_dir, "4360331")
drugs_file = os.path.join(root_dir, "drugs_crude.msgpack")
save_file = os.path.join(root_dir, f"geom_drugs_{conformations}.npy")
smiles_list_file = os.path.join(root_dir, 'geom_drugs_smiles.txt')
number_atoms_file = os.path.join(root_dir, f"geom_drugs_n_{conformations}.npy")

geom_permutation = os.path.join(root_dir, 'geom_permutation.npy')

if not os.path.exists(drugs_file):
    import subprocess
    print("Downloading drugs dataset...")
    subprocess.run(["wget", "-P", root_dir, DRUGS_URL])

    print("Extracting drugs dataset...")
    subprocess.run(["tar", "-xzf", tar_save_path, "-C", root_dir])

    print("Drugs dataset extracted.")

if not os.path.exists(geom_permutation):
    download_url(GEOM_PERMUTATION_URL, root_dir)

if not os.path.exists(save_file):
    extract_conformers()

if not os.path.exists(os.path.join(root_dir, "geom_train_atoms.npy")):

    for split, data in zip(["train", "val", "test"], load_split_data()):
        all_atoms = []
        all_coords = []
        all_sizes = []

        for mol in tqdm.tqdm(data):
            atoms = mol[:, 0].astype(int)
            coords = mol[:, 1:4].astype(np.float32)

            # zero-center coords and do PCA
            coords -= coords.mean(0)
            U, _, _ = np.linalg.svd(coords.T)
            if np.linalg.det(U) < 0:
                U[:, -1] *= -1
            coords = coords @ U

            size = atoms.shape[0]

            all_atoms.append(atoms)
            all_coords.append(coords)
            all_sizes.append(size)

        print("Saving...")

        atoms = np.concatenate(all_atoms)
        coords = np.concatenate(all_coords)
        sizes = np.array(all_sizes)

        np.save(os.path.join(root_dir, f"geom_{split}_atoms.npy"), atoms)
        np.save(os.path.join(root_dir, f"geom_{split}_coords.npy"), coords)
        np.save(os.path.join(root_dir, f"geom_{split}_sizes.npy"), sizes)

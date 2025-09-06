import random
import os
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from chem import Molecule, GEN, STOP, PAD
from pack import pack_using_lpfhp
from scipy.spatial.transform import Rotation
import tqdm
import torch

def random_rotate_translate(coords):
    # Rotate
    Q = Rotation.random().as_matrix()
    translation = np.random.randn(3).astype(coords.dtype)
    
    return coords @ Q.astype(coords.dtype) + translation

def get_perm(coords, start=None, beta=10, alpha=0.2, gamma=0.9):
    """
    Performs a probabilistic nearest neighbor traversal using softmax.
    
    - `beta`: Controls sharpness of softmax (higher = more greedy).
    - `alpha`: Balances between min distance to visited set and weighted history distance.
      - `alpha = 0.0`: Purely based on min distance to visited nodes.
      - `alpha = 1.0`: Purely based on weighted history distance.
    - `gamma`: Exponential decay factor for past nodes' influence.
      - `gamma = 1.0`: All visited nodes contribute equally.
      - `gamma < 1.0`: Older nodes contribute less.
    """

    distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
    np.fill_diagonal(distance_matrix, np.inf)  # No self-loops

    if start is None:
        start = np.random.randint(0, distance_matrix.shape[0])
    
    num_nodes = distance_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=bool)
    traversal = [start]
    visited[start] = True

    # Initialize the min_distance array with distances from the start node
    min_distances = distance_matrix[start].copy()
    history_nodes = [start]  # Track visited nodes in order

    for _ in range(num_nodes - 1):
        # Mask visited nodes
        min_distances[visited] = np.inf

        # Compute distance contribution from history with exponential decay
        history_distances = distance_matrix[history_nodes]  # (history_len, num_nodes)
        decay_weights = gamma ** np.arange(len(history_nodes) - 1, -1, -1).reshape(-1, 1)  # (history_len, 1)
        weighted_history_distance = np.sum(decay_weights * history_distances, axis=0) / np.sum(decay_weights)

        # Mask visited nodes in history distance
        weighted_history_distance[visited] = np.inf

        # Compute combined distance metric
        combined_distances = alpha * weighted_history_distance + (1 - alpha) * min_distances

        # Compute softmax probabilities
        logits = -beta * combined_distances  # Negative because lower distance is better
        probs = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probs = probs / np.sum(probs)

        # Sample the next node
        next_node = np.random.choice(num_nodes, p=probs)

        # Mark node as visited and update traversal
        visited[next_node] = True
        traversal.append(next_node)
        history_nodes.append(next_node)

        # Update min_distances
        min_distances = np.minimum(min_distances, distance_matrix[next_node])

    return traversal


def cache_permutations(coords_list, num_perms=20):
    import tqdm
    import joblib
    NUM_PROCESS_JOBS = 64

    def handle(coords):
        perms = np.array([get_perm(coords) for _ in range(num_perms)]).T
        return perms

    all_perms = joblib.Parallel(n_jobs=NUM_PROCESS_JOBS, backend="loky")(
        joblib.delayed(handle)(coords) for coords in tqdm.tqdm(coords_list, desc="Caching permutations")
    )

    perms = np.concatenate(all_perms)
    return perms

class MoleculeDataset(Dataset):
    # requires atoms, coords, sizes
    # supports rotate, perm

    def __init__(self, name, rotate=False, perm=None, num_perms=20, prop=False):
        # num_perms only matters the first time you run this with traverse_cache

        self.name = name
        self.rotate = rotate
        self.perm = perm
        self.prop = prop

        if name == "geomconf_train":
            atoms = np.load(f"data/{name}_atoms.npy")
            sizes = np.load(f"data/{name}_sizes.npy")
            cumsum = np.cumsum(sizes)[:-1]
            atoms = np.split(atoms, cumsum)
            num_confs = np.load(f"data/{name}_num_confs.npy")
            self.atoms = [arr for arr, rep in zip(atoms, num_confs) for _ in range(rep)]

            coords = np.load(f"data/{name}_crest_coords.npy").astype(np.float32)
            sizes = np.repeat(sizes, num_confs)
            cumsum = np.cumsum(sizes)[:-1]
            self.coords = np.split(coords, cumsum)

            self.sizes = sizes + 2
        else:
            atoms = np.load(f"data/{name}_atoms.npy")
            coords = np.load(f"data/{name}_coords.npy").astype(np.float32)
            sizes = np.load(f"data/{name}_sizes.npy")

            cumsum = np.cumsum(sizes)[:-1]
            self.atoms = np.split(atoms, cumsum)
            self.coords = np.split(coords, cumsum)
            self.sizes = sizes + 2

        if self.perm == "traverse_cache":
            if not os.path.exists(f"data/{name}_perms.npy"):
                # create cache if it doesn't exist
                self.perms = cache_permutations(self.coords, num_perms=num_perms)
                np.save(f"data/{name}_perms.npy", self.perms)
            else:
                self.perms = np.load(f"data/{name}_perms.npy")

            self.perms = np.split(self.perms, cumsum)
            self.num_perms = self.perms[0].shape[1]
        
        if self.prop:
            self.props = np.load(f"data/{name}_props.npy")
            self.prop_tokens = np.load(f"data/{name}_prop_tokens.npy")
            self.num_props = self.prop_tokens.shape[0]

    def __getitem__(self, i):
        atoms = self.atoms[i]
        coords = self.coords[i]
        
        # mask = (atoms >= 1) & (atoms <= 118)
        if self.rotate:
            coords = random_rotate_translate(coords)
        
        if random.random() < 0.667:
            if self.perm == "traverse_cache":
                j = random.randint(0, self.num_perms-1)
                p = self.perms[i][:, j]
            elif self.perm == "traverse":
                p = get_perm(coords)
            elif self.perm == "random":
                p = np.random.permutation(atoms.shape[0])
            else:
                p = np.arange(atoms.shape[0])

            atoms = atoms[p]
            coords = coords[p]

        atoms = np.concatenate([np.array([GEN]), atoms, np.array([STOP])], axis=0)
        coords = np.concatenate([np.zeros((1, 3), dtype=np.float32), coords, np.zeros((1, 3), np.float32)], axis=0)

        loss_mask = np.ones(atoms.shape[0], dtype=np.float32)
        loss_mask[-1] = 0.0

        idx = np.arange(atoms.shape[0])

        return idx, atoms, coords, loss_mask

    def __len__(self):
        return len(self.atoms)
    
    def mol(self, i):
        _, atoms, coords, _ = self[i]
        return Molecule(atoms[1:-1], coords[1:-1])

class PackedDataset(Dataset):
    def __init__(self, dataset, packlen=512, packdepth=8):
        self.dataset = dataset
        self.packlen = packlen
        self.packdepth = packdepth

        if isinstance(dataset, ConcatDataset):
            self.sizes = np.concatenate([ds.sizes for ds in dataset.datasets])
        else:
            self.sizes = dataset.sizes
        
        # calculate exact histogram of example sizes-1
        self.histogram = np.zeros(packlen+1, dtype=int)
        np.add.at(self.histogram, self.sizes-1, 1)
        
        strategy_set, strategy_repeat_count = pack_using_lpfhp(self.histogram, packlen+1, packdepth) # we pack to N+1 because the stop token is removed from input

        self.bucket_ptrs = [0 for _ in range(packlen+1)]
        self.buckets = [[] for _ in range(packlen+1)]
        for i, size in enumerate(tqdm.tqdm(self.sizes, desc="Bucketing")):
            # size-1 is the number of tokens, not counting stop token
            assert 0 < size-1 < packlen+1, f"Size {size-1} out of range"
            self.buckets[size-1].append(i)

        self.strats = []
        for strat, count in zip(strategy_set, strategy_repeat_count):
            for _ in range(count):
                self.strats.append(strat)


        order = list(range(len(self.strats)))
        random.seed(0)
        random.shuffle(order)
        for bucket in self.buckets:
            random.shuffle(bucket)
        
        # create packs
        self.packs = []
        for i in tqdm.tqdm(order, desc="Packing"):
            strat = self.strats[i]

            pack = []
            for length in strat:
                idx = self.buckets[length-1][self.bucket_ptrs[length-1]]
                pack.append(idx)
                self.bucket_ptrs[length-1] += 1
            
            self.packs.append(pack)

    def __getitem__(self, i):
        pack = self.packs[i]

        pack_idx = []
        pack_atoms = []
        pack_coords = []
        pack_loss_mask = []
        pack_length = 0

        for idx in pack:
            idx, atoms, coords, loss_mask = self.dataset[idx]
            pack_idx.append(idx)
            pack_atoms.append(atoms)
            pack_coords.append(coords)
            pack_loss_mask.append(loss_mask)
            pack_length += idx.shape[0]
        
        num_pad = self.packlen - pack_length + 1 # +1 for stop token
        pack_idx.append(np.zeros(num_pad, dtype=np.int64))
        pack_atoms.append(np.full((num_pad,), PAD, dtype=np.int64))
        pack_coords.append(np.zeros((num_pad, 3), dtype=np.float32))
        pack_loss_mask.append(np.zeros(num_pad, dtype=np.float32))

        idx = torch.from_numpy(np.concatenate(pack_idx, axis=0))
        atoms = torch.from_numpy(np.concatenate(pack_atoms, axis=0))
        coords = torch.from_numpy(np.concatenate(pack_coords, axis=0))
        loss_mask = torch.from_numpy(np.concatenate(pack_loss_mask, axis=0))

        return idx[:-1], atoms[:-1], coords[:-1], atoms[1:], coords[1:], loss_mask[:-1]

    def __len__(self):
        return len(self.packs)

import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

class SimpleDataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers):
        super().__init__()
        self.stateful_loader = StatefulDataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pack_collate_fn,
            persistent_workers=True,
            shuffle=True,
            drop_last=True
            )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pack_collate_fn,
            persistent_workers=True,
            shuffle=False,
            drop_last=False
            )

    def train_dataloader(self):
        return self.stateful_loader
    
    def val_dataloader(self):
        return self.val_loader

    def state_dict(self):
        print("[DataModule] Saving state_dict")
        # Save the state of the DataLoader
        return {"dataloader_state": self.stateful_loader.state_dict()}

    def load_state_dict(self, state_dict):
        # Restore the state of the DataLoader
        if "dataloader_state" in state_dict:
            self.stateful_loader.load_state_dict(state_dict["dataloader_state"])

def pack_collate_fn(batch):
    idx, atoms, coords, target_atoms, target_coords, loss_mask = zip(*batch)

    idx = torch.cat(idx)
    atoms = torch.cat(atoms)
    coords = torch.cat(coords)
    target_atoms = torch.cat(target_atoms)
    target_coords = torch.cat(target_coords)
    loss_mask = torch.cat(loss_mask)

    return idx, atoms, coords, target_atoms, target_coords, loss_mask

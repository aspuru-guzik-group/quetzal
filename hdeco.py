import torch
import numpy as np
from train import Config, LitQuetzal
from metrics import compute_valid_unique, compute_rmsd
import tqdm

ckpt = "checkpoints/original.ckpt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIFF_STEPS = 60
dataset = "qm9_test"


lit = LitQuetzal.load_from_checkpoint(ckpt, map_location=DEVICE)
model = lit.ema.module
model.eval();

model.denoise = torch.compile(model.denoise)
model.encode1 = torch.compile(model.encode1)
model.encode2 = torch.compile(model.encode2)

kwargs = {
    "device": DEVICE,
    "num_steps": DIFF_STEPS,
    "pbar": False,
    "max_len": 32,
    "mask_atoms": "H",
}

atoms = np.load(f"data/{dataset}_atoms.npy")
coords = np.load(f"data/{dataset}_coords.npy")
sizes = np.load(f"data/{dataset}_sizes.npy")

cumsum = np.cumsum(sizes)[:-1]
atoms = np.split(atoms, cumsum)
coords = np.split(coords, cumsum)

all_samples = []
all_correct = []
all_rmsds = []
for idx in tqdm.trange(10):
# for idx in tqdm.trange(len(sizes)):
    a_true = atoms[idx]
    c_true = coords[idx]

    # remove H
    mask = (a_true == 1)
    a = a_true[~mask]
    c = c_true[~mask]

    prefix = (torch.from_numpy(a.astype(int)), torch.from_numpy(c.astype(np.float32)))

    samples, traj = model.generate(1, prefix=prefix, **kwargs)

    # calculate RMSD
    a_pred = samples[0].atoms.numpy()
    c_pred = samples[0].coords.numpy()
    
    correct_number_of_atoms, rmsd = compute_rmsd(a_pred, a_true, c_pred, c_true)
    all_correct.append(correct_number_of_atoms)
    all_rmsds.append(rmsd)
    all_samples.append(samples)

rmsds = np.array(all_rmsds)
corrects = np.array(all_correct)

finite_rmsds = rmsds[np.isfinite(rmsds)]
average_rmsd = np.mean(finite_rmsds)
print(f"Average RMSD (when finite): {average_rmsd}")

percentage_less_than_50 = np.mean(finite_rmsds < 0.5) * 100
percentage_less_than_10 = np.mean(finite_rmsds < 0.1) * 100
percentage_less_than_5 = np.mean(finite_rmsds < 0.05) * 100
percentage_correct = np.mean(corrects) * 100

print(f"Percentage of examples with RMSD < 0.5: {percentage_less_than_50}%")
print(f"Percentage of examples with RMSD < 0.1: {percentage_less_than_10}%")
print(f"Percentage of examples with RMSD < 0.05: {percentage_less_than_5}%")
print(f"Percentage of examples with correct number of atoms: {percentage_correct}%")

import os

save_dir = "samples/hdeco/quetzal"
os.makedirs(save_dir, exist_ok=False)

np.save(os.path.join(save_dir, "rmsds.npy"), rmsds)
np.save(os.path.join(save_dir, "corrects.npy"), corrects)
torch.save(all_samples, os.path.join(save_dir, "samples.pt"))

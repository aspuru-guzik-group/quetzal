from dataclasses import dataclass, asdict, MISSING
from train import Config, LitQuetzal
from datasets import MoleculeDataset
from chem import Molecule
import torch
import os
import time
import json
import numpy as np
import argparse

SAVE_LOCATION = "samples/density"

@dataclass
class DensityConfig:
    ckpt: str
    name: str
    dataset: str = "qm9_test"
    device: str = "cuda"
    num_samples: int = 0 # 0 means all
    chunk_size: int = 1000
    diff_steps: int = 60

def compute_density(config):
    # create the output directory
    output_dir = os.path.join(SAVE_LOCATION, config.name)
    os.makedirs(output_dir, exist_ok=False)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=4)

    lit = LitQuetzal.load_from_checkpoint(config.ckpt, map_location=config.device)
    model = lit.ema.module
    model.eval()

    dset = MoleculeDataset(config.dataset)
    num_samples = len(dset) if config.num_samples == 0 else config.num_samples
    assert num_samples <= len(dset), "num_samples exceeds dataset size"

    total_start_time = time.time()  # Start timing the entire process
    timings = []  # List to store timing information

    atom_log_density_chunks = []
    coord_log_density_chunks = []

    total_chunks = (num_samples + config.chunk_size - 1) // config.chunk_size

    for chunk_idx in range(total_chunks):
        print(f"Processing chunk {chunk_idx + 1}/{total_chunks}...")
        start_idx = chunk_idx * config.chunk_size
        end_idx = min(start_idx + config.chunk_size, num_samples)
        mols = []

        for i in range(start_idx, end_idx):
            _, atoms, coords, _ = dset[i]
            atoms = torch.from_numpy(atoms).to(config.device)
            coords = torch.from_numpy(coords.astype(np.float32)).to(config.device)
            mols.append(Molecule(atoms, coords))

        M = Molecule.batch(mols, True)

        start_time = time.time()  # Start timing the chunk
        atom_log_density, coord_log_density = model.log_density(
            (M.atoms, M.coords), device=config.device, num_steps=config.diff_steps
        )
        end_time = time.time()  # End timing the chunk

        duration = end_time - start_time
        print(f"Chunk {chunk_idx + 1} processed in {duration:.2f} seconds.")
        timings.append(duration)

        atom_log_density_chunks.append(atom_log_density.cpu())
        coord_log_density_chunks.append(coord_log_density.cpu())

    total_end_time = time.time()  # End timing the entire process
    total_duration = total_end_time - total_start_time
    print(f"Total log density computation time: {total_duration:.2f} seconds.")

    # Determine the maximum size across all chunks
    max_atoms = max(chunk.shape[1] for chunk in atom_log_density_chunks)

    # Pad all chunks to the maximum size
    atom_log_density_chunks = [
        torch.nn.functional.pad(chunk, (0, max_atoms - chunk.shape[1], 0, 0))
        for chunk in atom_log_density_chunks
    ]
    coord_log_density_chunks = [
        torch.nn.functional.pad(chunk, (0, max_atoms - chunk.shape[1], 0, 0))
        for chunk in coord_log_density_chunks
    ]

    atom_log_density = torch.cat(atom_log_density_chunks, dim=0)
    coord_log_density = torch.cat(coord_log_density_chunks, dim=0)

    # Save tensors
    torch.save(atom_log_density, os.path.join(output_dir, "atom_log_density.pt"))
    torch.save(coord_log_density, os.path.join(output_dir, "coord_log_density.pt"))
    print("Log density tensors saved.")

    # Compute averages
    avg_log_density = (atom_log_density + coord_log_density).sum(dim=-1).mean().item()
    avg_log_atom_density = atom_log_density.sum(dim=-1).mean().item()
    avg_log_coord_density = coord_log_density.sum(dim=-1).mean().item()

    print(f"Average log density: {avg_log_density:.4f}")
    print(f"Average atom log density: {avg_log_atom_density:.4f}")
    print(f"Average coord log density: {avg_log_coord_density:.4f}")

    # Save results
    results = {
        "log_density": avg_log_density,
        "atom_log_density": avg_log_atom_density,
        "coord_log_density": avg_log_coord_density,
        "chunk_timings": timings,
        "total_time": total_duration
    }

    results_path = os.path.join(output_dir, "density_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Density results saved to {results_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    for field in DensityConfig.__dataclass_fields__.values():
        if isinstance(field.default, bool):
            if field.default is False:
                parser.add_argument(f"--{field.name}", dest=field.name, action="store_true", help=f"Set {field.name} to True (default: False)")
            else:
                parser.add_argument(f"--no_{field.name}", dest=field.name, action="store_false", help=f"Set {field.name} to False (default: True)")
            parser.set_defaults(**{field.name: field.default})
        elif field.default is MISSING:  # Check if the field has no default value
            parser.add_argument(f"--{field.name}", type=field.type, required=True)
        else:
            parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    return DensityConfig(**vars(parser.parse_args()))

if __name__ == "__main__":
    config = parse_args()
    print(config)

    compute_density(config)

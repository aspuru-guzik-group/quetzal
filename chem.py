import torch
import numpy as np
from rdkit import Chem
PTABLE = Chem.GetPeriodicTable()

from dataclasses import dataclass


GEN = 127
PAD = 126
STOP = 0

QM9_ATOMS = [0, 1, 6, 7, 8, 9]
QM9_MASK = torch.zeros(128, dtype=torch.bool)
QM9_MASK[QM9_ATOMS] = True


@dataclass
class Molecule:
    atoms: torch.Tensor
    coords: torch.Tensor

    def __repr__(self):
        if self.batched:
            return f"BatchedMolecule(batch_size={self.coords.shape[0]}, atoms={self.atoms.shape}, coords={self.coords.shape})"
        else:
            num_atoms = 0
            for i in range(self.atoms.shape[0]):
                if self.atoms[i] == 0:
                    break
                num_atoms += 1
            return f"Molecule(num_atoms={num_atoms}, atoms={self.atoms.shape}, coords={self.coords.shape})"

    @property
    def xyzfile(self):
        assert not self.batched

        num_atoms = 0
        
        xyz = ""
        for i in range(self.atoms.shape[0]):
            atom = self.atoms[i]
            if atom == 0:
                break
            if atom > 118:
                print(f"Invalid atom number: {atom}")
                atom = 118
            element = PTABLE.GetElementSymbol(int(atom))
            coord = self.coords[i]
            xyz += f"{element} {coord[0]:f} {coord[1]:f} {coord[2]:f}\n"
            num_atoms += 1
        
        return f"{num_atoms}\n\n{xyz}"

    def show(self, view=None, viewer=None, zoom=True, color=None, opacity=None, scale=None):
        import py3Dmol
        if view is None:
            view = py3Dmol.view(width=400, height=400)
        view.addModel(self.xyzfile, "xyz", viewer=viewer)
        style = {
            "stick": {"radius": .2},
            "sphere": {"scale": .2},
        }
        if scale is not None:
            style["stick"]["radius"] *= scale
            style["sphere"]["scale"] *= scale
        if color is not None:
            style["stick"]["color"] = color
            style["sphere"]["color"] = color
        if opacity is not None:
            style["stick"]["opacity"] = opacity
            style["sphere"]["opacity"] = opacity
        view.setStyle({"model": -1}, style, viewer=viewer)
        if zoom:
            view.zoomTo()
        return view
    
    def to_html(self):
        view = self.show()
        view.render()
        t = view.js()
        js = t.startjs + t.endjs
        return js

    @property
    def batched(self):
        return len(self.coords.shape) == 3

    def __getitem__(self, idx):
        assert self.batched, "Can only index batched molecules."

        return Molecule(atoms=self.atoms[idx], coords=self.coords[idx])
    
    @classmethod
    def batch(cls, molecules, pad=False):
        if pad: # pads with the PAD token
            batch_size = len(molecules)
            max_atoms = max([m.atoms.shape[0] for m in molecules])
            atoms = torch.zeros(batch_size, max_atoms, dtype=torch.int64) + PAD
            coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32) + PAD

            for i, m in enumerate(molecules):
                num_atoms = m.atoms.shape[0]
                atoms[i, :num_atoms] = m.atoms
                coords[i, :num_atoms] = m.coords
            
        else:
            atoms = torch.stack([m.atoms for m in molecules])
            coords = torch.stack([m.coords for m in molecules])

        return cls(coords=coords, atoms=atoms)
    
    def unbatch(self):
        assert self.batched, "Can only unbatch batched molecules."
        return [Molecule(atoms=self.atoms[i], coords=self.coords[i]) for i in range(self.coords.shape[0])]
    
    def clone(self):
        return Molecule(atoms=self.atoms.clone(), coords=self.coords.clone())
    
    def to(self, device):
        return Molecule(atoms=self.atoms.to(device), coords=self.coords.to(device))
    
    @classmethod
    def from_xyz(cls, file, read_file=True):
        a, c = parse_xyz(file, read_file)
        return cls(atoms=a, coords=c)


import torch
from rdkit import Chem

# Function to parse the XYZ file and extract atomic numbers and coordinates
def parse_xyz(file, read_file=True):
    if read_file:
        with open(file, 'r') as f:
            lines = f.readlines()
    else:
        lines = file.split('\n')

    # Extract the number of atoms from the first line
    num_atoms = int(lines[0].strip())

    # The atomic data starts at line 3 (after the number of atoms and a comment line)
    atomic_numbers = []
    coordinates = []

    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])

        # Get atomic number from the element symbol using RDKit
        atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(element)

        atomic_numbers.append(atomic_number)
        coordinates.append([x, y, z])

    # Convert lists to PyTorch tensors
    a = torch.tensor(atomic_numbers, dtype=torch.int64)
    c = torch.tensor(coordinates, dtype=torch.float32)

    return a, c

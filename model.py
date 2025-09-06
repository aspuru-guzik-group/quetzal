# https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm

from simple_mlp import SimpleMLPAdaLN, FourierCoords, MPFourier
from attention import Block, LayerNorm

from chem import Molecule, GEN, STOP, PAD, QM9_MASK

from torch.nn.attention.flex_attention import create_block_mask

P_mean = -1.2
P_std = 1.2

class Quetzal(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.n_embd = config.n_embd

        if config.pe == "learned":
            self.wpe = nn.Parameter(torch.randn(config.block_size, config.n_embd))
        elif config.pe == "none":
            self.register_buffer("wpe", torch.zeros(config.block_size, config.n_embd))
        else:
            raise ValueError(f"Unknown positional encoding {config.pe}")

        self.sigma_data = config.sigma_data
        self.diff_mult = config.diff_mult
        
        assert config.n_layer % 2 == 0
        nlayer = config.n_layer // 2

        self.blocks1 = nn.ModuleList([Block(config) for _ in range(nlayer)])
        self.blocks2 = nn.ModuleList([Block(config) for _ in range(nlayer)])

        # self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        # self.ln2 = LayerNorm(config.n_embd, bias=config.bias)

        self.embed_atoms = nn.Embedding(128, config.n_embd)

        self.embed_fourier = None
        if config.coord_fourier > 0:
            self.embed_fourier = FourierCoords(
                in_features=3, num_channels=config.coord_fourier, out_features=config.n_embd, bandwidth=config.coord_bandwidth)
        else:
            self.embed_fourier = lambda x: torch.zeros(*x.shape[:-1], config.n_embd, device=x.device, dtype=x.dtype)
        
        self.embed_scalars = MPFourier(config.n_embd)
        
        self.embed_coords = nn.Linear(3, config.n_embd, bias=config.bias)
        self.proj_logits = nn.Linear(config.n_embd, 128, bias=config.bias)

        self.simple_mlp = SimpleMLPAdaLN(config)
    
    def encode1(self, idx, atoms, coords, block_mask=None):
        seq = self.embed_atoms(atoms)
        seq = seq + self.embed_coords(coords)
        seq = seq + self.embed_fourier(coords)

        # the next two lines are not necessary, but were used in training
        scalars = torch.zeros_like(atoms, dtype=seq.dtype, device=seq.device)
        seq = seq + self.embed_scalars(scalars.unsqueeze(-1))

        seq = seq + self.wpe[idx]

        for block in self.blocks1:
            seq = block(seq, block_mask)
        
        # seq = self.ln1(seq)

        return seq
    
    def encode2(self, atoms, seq, block_mask=None):
        seq = seq + self.embed_atoms(atoms)

        for block in self.blocks2:
            seq = block(seq, block_mask)
        
        # seq = self.ln2(seq)
        
        return seq

    def denoise(self, x, t, z_prefix):
        sigma = t
        # preconditioning
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.simple_mlp(
            x=(c_in * x),
            t=c_noise,
            c=z_prefix
        )
        D_x = c_skip * x + c_out * F_x
        return D_x

    def forward(self, idx, atoms, coords, target_atoms, target_coords, loss_mask):

        document_id = (atoms == STOP).cumsum(dim=0).roll(shifts=1, dims=0)
        document_id[0] = 0  # Reset the first document_id to 0
        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = document_id[q_idx] == document_id[kv_idx]
            pad_mask = (atoms[q_idx] == PAD) | (atoms[kv_idx] == PAD)
            return causal_mask & document_mask & ~pad_mask

        S = len(atoms)
        block_mask = create_block_mask(document_causal_mask, 1, 1, S, S, device=coords.device, _compile=True)

        return self.actual_forward(block_mask, idx, atoms, coords, target_atoms, target_coords, loss_mask)
    
    @torch.compile
    def actual_forward(self, block_mask, idx, atoms, coords, target_atoms, target_coords, loss_mask):

        idx = idx.unsqueeze(0)
        atoms = atoms.unsqueeze(0)
        coords = coords.unsqueeze(0)
        target_atoms = target_atoms.unsqueeze(0)
        target_coords = target_coords.unsqueeze(0)

        seq = self.encode1(idx, atoms, coords, block_mask)
        # seq = self.encode1(idx, atoms, coords, block_mask)
        logits = self.proj_logits(seq)
        cross_entropy = F.cross_entropy(logits.view(-1, 128), target_atoms.view(-1), reduction="none")
        cross_entropy = (cross_entropy * loss_mask.view(-1)).mean()

        z_prefix = self.encode2(target_atoms, seq, block_mask)

        # timestep sampling
        bsz, n, _ = target_coords.shape
        diff_bsz = bsz * self.diff_mult

        target_coords = target_coords.repeat(self.diff_mult, 1, 1)
        z_prefix = z_prefix.repeat(self.diff_mult, 1, 1)
        rnd_normal = torch.randn([diff_bsz, n, 1]).to(z_prefix)
        sigma = (rnd_normal * P_std + P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noisy_coords = target_coords + torch.randn_like(target_coords) * sigma

        pred = self.denoise(noisy_coords, sigma, z_prefix)

        loss = weight * ((pred - target_coords) ** 2) # you could log loss by timestep and see which times are reducible
        # you can filter out loss computation for start tokens here
        loss = (loss * loss_mask.unsqueeze(-1)).mean()
        return loss, cross_entropy

    @torch.no_grad()
    def sample_coord(self, z_prefix, device="cpu", num_steps=18, sigma_min=1e-4, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, truncate_w=1.0, truncate_t=0.0):
        # z_prefix: (bsz, n_embd)
        bsz = z_prefix.shape[0]

        # Time step discretization.
        step_indices = torch.arange(num_steps, device=z_prefix.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        latents = torch.randn(bsz, 3, device=device)
        x_next = latents * t_steps[0]
        traj = [x_next.cpu().clone()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.denoise(x_hat, t_hat.expand(bsz, 1), z_prefix)
            d_cur = (x_hat - denoised) / t_hat
            if t_hat < truncate_t:
                d_cur = d_cur * truncate_w
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x_next, t_next.expand(bsz, 1), z_prefix)
                d_prime = (x_next - denoised) / t_next
                if t_next < truncate_t:
                    d_prime = d_prime * truncate_w
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            traj.append(x_next.cpu().clone())

        return x_next, torch.stack(traj, dim=1)
        # traj: (bsz, num_steps, 3)
    
    @torch.no_grad()
    def log_density(self, prefix, device="cpu", num_steps=120, sigma_min=1e-4, sigma_max=80, rho=7, truncate_w=1.0, truncate_t=0.0):
        # the prefix should contain GEN and STOP
        atoms, coords = prefix
        atoms = atoms.to(device)
        coords = coords.to(device)
        # assert atoms.ndim == 1, "prefix should be a single molecule"
        # assert coords.ndim == 2, "prefix should be a single molecule"
        assert atoms.ndim == 2, "prefix should be batched"
        assert coords.ndim == 3, "prefix should be batched"
        bsz = atoms.shape[0]
        n = atoms.shape[1]

        idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)
        seq = self.encode1(idx, atoms, coords)
        logits = self.proj_logits(seq)
        atom_log_density = F.log_softmax(logits, dim=-1)
        target_atoms = atoms[:, 1:]
        atom_log_density = atom_log_density.gather(-1, target_atoms.unsqueeze(-1)).squeeze(-1)
        atom_log_density[target_atoms == PAD] = 0
        
        z_prefix = self.encode2(target_atoms, seq[:, :-1])

        # return logits, atom_log_density, z_prefix

        def drift_aux(x, t, z):
            denoised = self.denoise(x, t, z)
            out = (x - denoised) / t
            return out, out

        def drift_with_div(x, t, z):
            jac, aux = torch.func.jacrev(drift_aux, argnums=0, has_aux=True)(x, t, z)
            return aux, torch.trace(jac)
        
        batch_drift_div = torch.vmap(drift_with_div, in_dims=(0, 0, 0))

        # flatten and compute log-density for all atom coords
        not_pad = atoms[:, :-1] != PAD
        x = coords[:, :-1][not_pad]
        z_prefix = z_prefix[not_pad]

        bsz = z_prefix.shape[0]

        # Time step discretization.
        step_indices = torch.arange(num_steps, device=z_prefix.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = t_steps.flip(0)

        # Main sampling loop.
        logdensity = torch.zeros(bsz, device=device)
        x_next = x

        # traj = [x_next.cpu().clone()]
        # logp_traj = [logdensity.cpu().clone()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            
            t_hat = t_cur
            x_hat = x_next

            # Euler step.
            d_cur, div_cur = batch_drift_div(x_hat, t_hat.expand(bsz, 1), z_prefix)
            # if t_hat < truncate_t:
            #     d_cur = d_cur * truncate_w

            x_next = x_hat + (t_next - t_hat) * d_cur
            logdensity = logdensity + (t_next - t_hat) * div_cur

            # Apply 2nd order correction.
            d_prime, div_prime = batch_drift_div(x_next, t_next.expand(bsz, 1), z_prefix)
            # if t_next < truncate_t:
            #     d_prime = d_prime * truncate_w

            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            logdensity = logdensity + (t_next - t_hat) * (0.5 * div_cur + 0.5 * div_prime)
            
            # traj.append(x_next.cpu().clone())
            # logp_traj.append(logdensity.cpu().clone())

        logdensity = logdensity + randn_log_density(x, sigma_max)
        # logdensity = logdensity.cpu()
        # return x_next, torch.stack(traj, dim=1), logdensity, torch.stack(logp_traj, dim=1)

        coord_log_density = torch.zeros(atoms[:, :-1].shape, device=device)
        coord_log_density[not_pad] = logdensity

        return atom_log_density, coord_log_density

    @torch.no_grad()
    def generate(self, bsz=1, max_len=None, device="cpu", pbar=True, mask_atoms=None, prefix=None, **kwargs):
        if mask_atoms is None: # TODO: should just accept a list of allowed atoms
            # mask is True if allowed
            mask = torch.ones(128, dtype=torch.bool, device=device)
            mask[GEN] = False
            mask[PAD] = False
        elif mask_atoms == "qm9":
            mask = QM9_MASK.to(device)
        elif mask_atoms == "H":
            mask = torch.zeros(128, dtype=torch.bool, device=device)
            mask[0] = True
            mask[1] = True
        elif type(mask_atoms) == list:
            mask = torch.zeros(128, dtype=torch.bool, device=device)
            mask[mask_atoms] = True
        
        max_len = max_len or self.block_size

        if prefix is None:
            atoms = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
            coords = torch.zeros(bsz, 1, 3, device=device)
        else:
            atoms, coords = prefix
            atoms = atoms.to(device)
            coords = coords.to(device)
            assert atoms.ndim == 1, "prefix should be a single molecule"
            assert coords.ndim == 2, "prefix should be a single molecule"
            # prepend GEN
            atoms = torch.cat([torch.tensor([GEN], device=device), atoms])
            coords = torch.cat([torch.zeros(1, 3, device=device), coords])

            # expand to bsz
            atoms = atoms.expand(bsz, -1)
            coords = coords.expand(bsz, -1, 3)
            max_len = max_len - atoms.shape[1]

        all_traj = []
        stop_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
        
        pbar = tqdm.trange(max_len) if pbar else range(max_len)
        for i in pbar:

            idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)
            
            seq = self.encode1(idx, atoms, coords)

            logits = self.proj_logits(seq[:, -1, :])
            logits[:, ~mask] = -float('inf')
            next_atom = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            stop_mask |= next_atom.squeeze(-1) == 0
            if stop_mask.all():
                break

            atoms = torch.cat([atoms, next_atom], dim=1)

            x = self.encode2(atoms[:, 1:], seq)
            x = x[:, -1, :]
            
            next_coord, traj = self.sample_coord(x, device=device, **kwargs)
            all_traj.append(traj)

            coords = torch.cat([coords, next_coord.view(bsz, 1, 3)], dim=1)
        
        # # find the first 0 in atoms
        # num_atoms = (atoms == 0).argmax(dim=1)
        
        if len(all_traj) == 0:
            all_traj = torch.zeros(bsz, 0, 0, 3, device=device)
        else:
            all_traj = torch.stack(all_traj, dim=1)
            # (bsz, block_size, num_steps, 3)
        
        return Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu"), all_traj

def randn_log_density(x, sigma=1.0):
    d = 3
    return -0.5 * (d * np.log(2 * np.pi * sigma**2) + torch.sum(x**2, dim=-1) / sigma**2)

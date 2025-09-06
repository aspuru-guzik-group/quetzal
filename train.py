# Standard library imports
import os
import datetime
import argparse
import glob
from dataclasses import dataclass, asdict, fields

# Third-party imports
import torch
torch.set_float32_matmul_precision('medium')
import numpy as np
import wandb
import py3Dmol
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy
import lightning as L

# Local imports
from chem import Molecule
from model import Quetzal
from draw import show_traj
from metrics import compute_valid_unique
from datasets import (
    MoleculeDataset,
    PackedDataset,
    SimpleDataModule,
)

entity = os.getenv("WANDB_ENTITY")

# -------------------- Callbacks --------------------

class LogGradientNorm(Callback):
    """
    Logs the gradient norm.
    This should log the gradient norm unscaled and before clipping.
    """

    def on_before_optimizer_step(self, trainer, *args, **kwargs):
        total_norm = 0.0
        for param in trainer.lightning_module.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        trainer.lightning_module.log_dict({"train/grad_norm": total_norm})

# -------------------- Configuration --------------------

@dataclass
class Config:
    # General settings
    devices: int = 1
    num_nodes: int = 1
    dataset: str = "qm9"
    name: str = "try"
    debug: bool = False
    resume_path: str = None

    # Data settings
    packlen: int = 128
    packdepth: int = 6
    bsz: int = 180
    num_workers: int = 15

    # Training settings
    max_epochs: int = 2000
    vis_every_n_epochs: int = 50
    diff_steps: int = 18
    lr: float = 4e-4
    beta1: float = 0.9
    beta2: float = 0.95
    wd: float = 1e-5
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    val_check_interval: float = 1.0

    # Augmentation settings
    rotate: bool = True
    perm: str = "none" # "none", "random", "traverse", "traverse_cache"
    num_perms: int = 20 # only used if perm is "traverse_cache" for the first time

    # Model settings
    pe: str = "learned"
    trunk_mlp: str = "mlp" # "mlp" or "swiglu"
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    coord_fourier: int = 256
    coord_bandwidth: float = 20
    qk_norm: bool = True
    dropout: float = 0.0
    bias: bool = False
    diff_mult: int = 4
    diff_w: int = 1536
    diff_d: int = 6
    diff_mlp: str = "mlp"
    diff_mlp_expand: int = 1
    diff_fourier: int = 512
    diff_bandwidth: float = 20
    sigma_data: float = 1.4 # 2.47 for GEOM

    # Checkpoint settings
    save_interval_minutes: int = 30

# -------------------- Model --------------------

class LitQuetzal(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        config = {k: v for k, v in config.items() if k in {f.name for f in fields(Config)}}
        config = Config(**config)

        self.config = config
        self.model = Quetzal(config)
        self.simple_mlp = self.model.simple_mlp
        self.warmup_steps = 5000

        self.ema = AveragedModel(
            self.model,
            device="cpu",
            multi_avg_fn=get_ema_multi_avg_fn(self.config.ema_decay),
            use_buffers=False,
        )
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad = False

    def forward(self, batch):
        return self.model.forward(*batch)

    def training_step(self, batch, batch_idx):
        diffusion_loss, cross_entropy_loss = self(batch)
        total_loss = diffusion_loss + cross_entropy_loss

        self.log("train/diffusion_loss", diffusion_loss, prog_bar=True)
        self.log("train/cross_entropy_loss", cross_entropy_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        self.ema.update_parameters(self.model)
        return total_loss

    def validation_step(self, batch, batch_idx):
        diffusion_loss, cross_entropy_loss = self(batch)
        total_loss = diffusion_loss + cross_entropy_loss

        self.log(f"val/diffusion_loss", diffusion_loss, prog_bar=True, sync_dist=True)
        self.log(f"val/cross_entropy_loss", cross_entropy_loss, prog_bar=True, sync_dist=True)
        self.log(f"val/total_loss", total_loss, prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self):
        if self.current_epoch % self.config.vis_every_n_epochs != 0 or self.global_rank != 0:
            return

        if self.config.dataset == "qm9":
            gens = []
            for i in range(1):
                out = self.ema.module.generate(10000, device=self.device, max_len=32, num_steps=self.config.diff_steps, pbar=self.config.debug, mask_atoms="qm9")
                if i == 0:
                    samples, all_traj = out
                else:
                    samples, _ = out
                gens.append(samples)
            samples = sum([s.unbatch() for s in gens], [])
            valid, unique = compute_valid_unique(samples)
            self.log("validity", valid, rank_zero_only=True)
            self.log("uniqueness", unique, rank_zero_only=True)
            self.log("valid_unique", valid * unique, rank_zero_only=True)
        elif self.config.dataset == "geom":
            samples, all_traj = self.ema.module.generate(1000, device=self.device, max_len=192, num_steps=self.config.diff_steps, pbar=self.config.debug)
            samples = samples.unbatch()
            valid, unique = compute_valid_unique(samples)
            self.log("validity", valid, rank_zero_only=True)
            self.log("uniqueness", unique, rank_zero_only=True)
            self.log("valid_unique", valid * unique, rank_zero_only=True)
        else:
            return

        grid = (2, 3)
        view = py3Dmol.view(width=1200, height=800, viewergrid=grid)

        for i in range(3):
            M = samples[i]
            view = M.show(view=view, viewer=(0, i))
            view = show_traj((samples, all_traj), b_idx=i, view=view, viewer=(1, i))
        
        view.render()
        t = view.js()
        js = t.startjs + t.endjs
        html = wandb.Html(js)

        wandb.log({"samples": html})

    def configure_optimizers(self):
        wd, no_wd = [], []
        for name, param in self.model.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or "simple_mlp" in name:
                no_wd.append(param)
            else:
                wd.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": wd, "weight_decay": self.config.wd},
                {"params": no_wd, "weight_decay": 0.0},
            ],
            betas=(self.config.beta1, self.config.beta2),
            lr=self.config.lr,
            eps=1e-15,
            fused=self.config.grad_clip == 0.0,
        )

        scheduler = LambdaLR(optimizer, lambda step: step / self.warmup_steps if step < self.warmup_steps else 1.0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# -------------------- Main --------------------

def parse_args():
    parser = argparse.ArgumentParser()
    for field in Config.__dataclass_fields__.values():
        if isinstance(field.default, bool):
            if field.default is False:
                parser.add_argument(f"--{field.name}", dest=field.name, action="store_true", help=f"Set {field.name} to True (default: False)")
            else:
                parser.add_argument(f"--no_{field.name}", dest=field.name, action="store_false", help=f"Set {field.name} to False (default: True)")
            parser.set_defaults(**{field.name: field.default})
        else:
            parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    return Config(**vars(parser.parse_args()))

if __name__ == "__main__":
    config = parse_args()
    lit = LitQuetzal(asdict(config))

    @rank_zero_only
    def print_once():
        print(config)
        mlp_params = sum(p.numel() for p in lit.model.simple_mlp.parameters())
        total_params = sum(p.numel() for p in lit.model.parameters())
        trunk_params = total_params - mlp_params
        print(f"Total parameters: {total_params/1e6:.1f}M")
        print(f"Trunk parameters: {trunk_params/1e6:.1f}M")
        print(f"MLP parameters:   {mlp_params/1e6:.1f}M")

    print_once()

    checkpoint_dir = f"logs/quetzal/{config.name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        save_dir="logs",
        project="quetzal",
        entity=entity,
        name=config.name,
        config=asdict(config),
        offline=True,
    )

    def get_most_recent_checkpoint(checkpoint_dir):
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        print(ckpts)
        if not ckpts:
            return None
        return max(ckpts, key=os.path.getmtime)  # latest by modification time

    # Use explicitly provided path if set, otherwise get latest
    resume_path = config.resume_path or get_most_recent_checkpoint(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        train_time_interval=datetime.timedelta(minutes=config.save_interval_minutes),
        save_last="link",
        save_on_train_epoch_end=True,
    )

    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_epochs=1,
    )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices,
        num_nodes=config.num_nodes,
        strategy=DDPStrategy(timeout=datetime.timedelta(minutes=10)) if config.num_nodes > 1 else "auto",
        logger=wandb_logger,
        log_every_n_steps=10,
        gradient_clip_val=config.grad_clip if config.grad_clip > 0 else None,
        precision="bf16-mixed",
        enable_progress_bar=config.debug,
        callbacks=[LogGradientNorm(), checkpoint_callback, epoch_checkpoint_callback],
        num_sanity_val_steps=0 if config.debug else 1,
        val_check_interval=config.val_check_interval,
    )

    train_dataset = MoleculeDataset(
        f"{config.dataset}_train",
        rotate=config.rotate,
        perm=config.perm,
        num_perms=config.num_perms,
    )
    val_dataset = MoleculeDataset(f"{config.dataset}_val")

    dset = PackedDataset(train_dataset, config.packlen, config.packdepth)
    val_dset = PackedDataset(val_dataset, config.packlen, "max")

    dm = SimpleDataModule(
        dset,
        val_dset,
        config.bsz,
        config.num_workers,
    )

    # Pass the resume_path to the trainer
    trainer.fit(lit, datamodule=dm, ckpt_path=resume_path)
    wandb_logger.finalize(status="success")

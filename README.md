# Quetzal

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2505.13791-b31b1b.svg)](https://arxiv.org/abs/2505.13791)&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aspuru-guzik-group/quetzal/blob/main/colab.ipynb)

Code for [Scalable Autoregressive 3D Molecule Generation](https://arxiv.org/abs/2505.13791)

![Animated Molecule Generation](figures/anim.gif)

Setup:
```
mamba create -f environment.yml
```

This environment was prepared via:
```
mamba create -n quetzal python=3.10
mamba activate quetzal
mamba install c-compiler cxx-compiler # needed for torch.compile
pip install torch==2.6 lightning==2.5.0.post0 rdkit==2023.03.3 jupyter notebook ipywidgets scipy "numpy<2" matplotlib tqdm pandas wandb==0.18.7 seaborn msgpack py3Dmol torchdata
```

`rdkit==2023.03.3` is important to have consistent validity metrics

Optional W&B setup:
```
export WANDB_ENTITY=<your_entity>
```

Create a folder for SLURM logs (required):
```
mkdir -p slurm
```

Download checkpoint(s):
```
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/auhcheng/quetzal/resolve/main/original.ckpt # best qm9 model
wget https://huggingface.co/auhcheng/quetzal/resolve/main/geom.ckpt # best geom model
```
You can find the rest of the checkpoints for ablation studies [here](https://huggingface.co/auhcheng/quetzal/tree/main).

Start playing around with the model in [`play.ipynb`](./play.ipynb)!

## Training and evaluation

Directly download the preprocessed data:
```
wget https://huggingface.co/auhcheng/quetzal/resolve/main/data.tar.gz
tar -xf data.tar.gz
```

Or download and preprocess data from raw:
```
python qm9.py # less than a minute
python geom.py # 30-60 minutes, ~100G space
```

The command for training on QM9:
```
python train.py --name=qm9_run
```

Add `--debug` for a progress bar. See `train.py` for more options.

The command for evaluating on QM9: (change `--ckpt` if needed)
```
python generate.py --ckpt=logs/quetzal/qm9_run/checkpoints/epoch=1999-step=188000.ckpt --name=qm9_samples --device=cuda --num_samples=10000 --num_chunks=1 --diff_steps=60 --max_len=32
python metrics.py --samples_dir=samples/gen/qm9_samples --dataset=qm9
```

The command for training on GEOM:
```
sbatch 4run.sh
```

To continue the run for longer than 24 hours, simply run the same training command and make sure the run has the same `--name`, or pass `--resume_path=<path>.ckpt`.

The command for evaluating on GEOM:
```
python generate.py --ckpt=logs/quetzal/geom_run/checkpoints/epoch=201-step=734272.ckpt --name=geom_samples --device=cuda --num_samples=10000 --num_chunks=10 --diff_steps=120 --max_len=192
python metrics.py --samples_dir=samples/gen/geom_samples --dataset=geom
```

To submit multiple jobs, specify commands in the `jobs` file, and run `./submit.sh` to submit each line in the `jobs` file.


You can find almost all figures and how they were generated in `figures/`.
First, download the generated samples:
```
wget https://huggingface.co/auhcheng/quetzal/resolve/main/samples.tar.gz
tar -xf samples.tar.gz
```
The samples in `samples/` may be in .xyz format, or batched together as `Molecule` objects stored with their diffusion traces as `gen.pt`. You can see how these are loaded in `figures/uncurated/show.ipynb` or `figures/anim/anim.ipynb`.
For automating conversion of html to png, `figures/render.py` may be useful.

## Hydrogen decoration

Evaluate Quetzal on hydrogen decoration:
```
python hdeco.py
```
The progress bar may appear to hang due to `torch.compile`.

For OpenBabel+Hydride:
```
mamba install openbabel
pip install hydride
```
Use `addH.sh`. You will need to prepare some `.xyz` files of the test set without hydrogens. You also need to rewrite `hdeco.py` to calculate RMSD for these generated `.xyz` files.

For Olex2, you may find `run_olex2.scpt` useful.

Some of the samples are flipped along the x/y/z axes, because the QM9 test data were reprocessed using PCA at some point.


## Exact log-likelihood computation

```
python density.py --ckpt=checkpoints/original.ckpt --name=qm9_density
python density.py --ckpt=checkpoints/geom.ckpt --name=geom_density
```

---

This work was made possible by several previous works, including but not limited to:
- [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Elucidating Diffusion Models](https://arxiv.org/abs/2206.00364)
- [Equivariant Diffusion Models](https://arxiv.org/abs/2203.17003)
- [Symphony](https://openreview.net/forum?id=MIEnYtlGyv)

If you find any of the code in this repo useful, please cite!

```
@article{cheng2025scalable,
  title={Scalable Autoregressive {3D} Molecule Generation},
  author={Cheng, Austin H and Sun, Chong and Aspuru-Guzik, Al{\'a}n},
  journal={arXiv preprint arXiv:2505.13791},
  year={2025}
}
```

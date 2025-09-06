from chem import Molecule
import py3Dmol
import numpy as np
import torch

def show_traj(out, b_idx=0, view=None, viewer=None, interval=50, step=1, PCA=False, tailpad=0):
    M_batched, traj = out
    M_full = M_batched[b_idx]
    traj = traj[b_idx]

    atoms = M_full.atoms
    coords = M_full.coords

    if PCA:
        mean = coords.mean(dim=0, keepdim=True)
        centered = coords - mean

        # PCA for visualization
        U, _, _ = np.linalg.svd(centered.T.numpy())
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        rotate = torch.tensor(U)
        # coords = coords @ rotate
    
    N = traj.shape[0]
    T = traj.shape[1]

    offset = atoms.shape[0] - N

    Ms = []
    for i in range(N):
        i += offset
        if atoms[i] == 0:
            break

        a = atoms[:i+1]
        x = coords[:i+1]

        for t in range(T):
            x[i] = traj[i-offset, t]

            if PCA:
                M = Molecule(a, (x - mean) @ rotate).clone()
            else:
                M = Molecule(a, x).clone()
            Ms.append(M)
    
    xyzfiles = [m.xyzfile for m in Ms]
    if len(xyzfiles) > 0:
        xyzfiles = [xyzfiles[-1]] + xyzfiles
        xyzfiles = xyzfiles + [xyzfiles[-1]] * tailpad

    if view is None:
        view = py3Dmol.view(width=400, height=400)
    view.addModelsAsFrames("".join(xyzfiles), "xyz", viewer=viewer)
    view.setStyle({"model": -1}, {"stick": {"radius": .2}, "sphere": {"scale": .2}}, viewer=viewer)
    view.animate({'interval': interval, "step": step})
    # view.animate({'interval': interval, "loop": 'forward', 'reps': 1})
    view.zoomTo()
    return view

def show_order(M, view=None, viewer=None, interval=100):
    atoms = M.atoms
    coords = M.coords
    N = atoms.shape[0]

    Ms = [Molecule(atoms[:i+1], coords[:i+1]) for i in range(N)]
    
    xyzfiles = [m.xyzfile for m in Ms]
    # repeat each frame 5 times
    xyzfiles = [f for f in xyzfiles for _ in range(10)]

    if len(xyzfiles) > 0:
        xyzfiles = [xyzfiles[-1]] + xyzfiles
    if view is None:
        view = py3Dmol.view(width=400, height=400)
    view.addModelsAsFrames("".join(xyzfiles), "xyz", viewer=viewer)
    view.setStyle({"model": -1}, {"stick": {"radius": .2}, "sphere": {"scale": .2}}, viewer=viewer)
    view.animate({'interval': interval})
    view.zoomTo()
    return view

def show_grid(mol_list, nrows, ncols, view=None, res=200, offset=0):
    # mol_list is indexable
    res = 200
    view = py3Dmol.view(width=ncols * res, height=nrows * res, viewergrid=(nrows, ncols))

    for i in range(nrows*ncols):
        row = i // ncols
        col = i % ncols

        M = mol_list[i+offset]
        view = M.show(view=view, viewer=(row, col))
    return view

def make_html(view, fname="input.html"):
    net = f'<img id="img_A"><script src="https://3Dmol.org/build/3Dmol-min.js"></script><script src="https://3Dmol.org/build/3Dmol.ui-min.js"></script>' + view._make_html()
    net = net[:-14] + f'\nvar png = viewer_{view.uniqueid}.pngURI();\ndocument.getElementById("img_A").src = png;' + net[-14:]

    with open(fname, "w") as f:
        f.write(net)

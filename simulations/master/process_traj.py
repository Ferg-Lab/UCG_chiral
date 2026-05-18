import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Optional

def _read_lammps_dump_frames(path: str) -> List[Dict[str, np.ndarray]]:
    """
    Minimal parser for LAMMPS 'atom' style dump.
    Expects sections like:
      ITEM: TIMESTEP
      <int>
      ITEM: NUMBER OF ATOMS
      <int>
      ITEM: BOX BOUNDS [flags...]
      xlo xhi
      ylo yhi
      zlo zhi
      ITEM: ATOMS <col1> <col2> ... <colK>
      <rows of K columns>

    - Last atom column is assumed to be p (probability).
    - We look for columns named at least: x, y, z; 'p' can be last or explicitly named 'p'.
    - Returns list of frames: {"pos": (N,3), "p": (N,), "box": (3,2) [lo, hi]}
    """
    frames = []
    with open(path, "r") as fh:
        lines = fh.readlines()

    i = 0
    nlines = len(lines)
    while i < nlines:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        # TIMESTEP
        i += 1
        timestep = int(lines[i].strip()); i += 1

        # NUMBER OF ATOMS
        assert lines[i].startswith("ITEM: NUMBER OF ATOMS"); i += 1
        natoms = int(lines[i].strip()); i += 1

        # BOX BOUNDS
        assert lines[i].startswith("ITEM: BOX BOUNDS"); i += 1
        xlo, xhi = map(float, lines[i].split()); i += 1
        ylo, yhi = map(float, lines[i].split()); i += 1
        zlo, zhi = map(float, lines[i].split()); i += 1
        box = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]], float)

        # ATOMS header
        assert lines[i].startswith("ITEM: ATOMS")
        header = lines[i].strip().split()[2:]  # after "ITEM: ATOMS"
        i += 1

        # find columns
        # We require x,y,z; for p prefer named column; otherwise take last col as p.
        col_map = {name: idx for idx, name in enumerate(header)}
        #print(col_map)
        req_xyz = all(k in col_map for k in ["x","y","z"])
        if not req_xyz:
            raise ValueError(f"Dump {path} frame @t={timestep} is missing x/y/z in ATOMS header: {header}")

        if "c_compute" in col_map:
            col_map["p"] = col_map.pop("c_compute")
        elif "c_compute_chirality" in col_map:
            col_map["p"] = col_map.pop("c_compute_chirality")

        has_p_named = "p" in col_map

        pos = np.empty((natoms, 3), float)
        pvec = np.empty((natoms,), float)

        # read atoms rows
        for a in range(natoms):
            parts = lines[i+a].split()
            x = float(parts[col_map["x"]])
            y = float(parts[col_map["y"]])
            z = float(parts[col_map["z"]])
            pos[a] = (x, y, z)
            if has_p_named:
                
                pvec[a] = float(parts[col_map["p"]])
            else:
                pvec[a] = float(parts[-1])  # assume last column is p
        i += natoms

        frames.append({"pos": pos, "p": pvec, "box": box, "t": timestep})

    if not frames:
        raise ValueError(f"No frames parsed from {path}")
    return frames


def _pairwise_distances_pbc(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Pair distances with orthorhombic PBC via minimum image.
    positions: (N,3)
    box: (3,2) [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    """
    L = box[:,1] - box[:,0]            # box lengths
    invL = 1.0 / L
    # fractional coords in [0,1) then min image in fractional space
    frac = (positions - box[:,0]) * invL
    # pairwise fractional differences
    dfrac = frac[:,None,:] - frac[None,:,:]  # (N,N,3)
    dfrac -= np.round(dfrac)                 # minimum image
    dxyz = dfrac * L                         # back to Cartesian
    return np.linalg.norm(dxyz, axis=-1)     # (N,N)


def dump_to_batch(
    dump_path: str,
    REF: bool
) -> List[Dict[str, np.ndarray]]:
    """
    Convert a dump into a batch (list of frames) expected by REM:
    each item is {"R": (N,N) distances, "p": (N,)}.
    """

    if REF == False:
        sel = _read_lammps_dump_frames(dump_path)
        batch = []
        for fr in sel:
            R = _pairwise_distances_pbc(fr["pos"], fr["box"])
            batch.append({"R": R, "p": fr["p"]})
  
            # ADD another batch of opp chirality.
            batch.append({"R": R, "p": 1-fr["p"]})

    else:
        # Load AA data
        if not os.path.exists("/project2/andrewferguson/sivadasetty/doe/ucg/notebooks/chiral/manuscript/simulations/rem/high-t-repeat-nneigh12-NR-Hess-7/master/TTTT_AAData/aabatch_stride10.pkl"):
            sel = _read_lammps_dump_frames(dump_path)
            batch = []
            for fr in sel[::10]:
                R = _pairwise_distances_pbc(fr["pos"], fr["box"])
                batch.append({"R": R, "p": fr["p"]})
           
                # ADD another batch of opp chirality.
                batch.append({"R": R, "p": 1-fr["p"]})                


            with open("/project2/andrewferguson/sivadasetty/doe/ucg/notebooks/chiral/manuscript/simulations/rem/high-t-repeat-nneigh12-NR-Hess-7/master/TTTT_AAData/aabatch_stride10.pkl", "wb") as f:
                pickle.dump(batch, f)
        else:
            with open("/project2/andrewferguson/sivadasetty/doe/ucg/notebooks/chiral/manuscript/simulations/rem/high-t-repeat-nneigh12-NR-Hess-7/master/TTTT_AAData/aabatch_stride10.pkl", "rb") as f:
                batch = pickle.load(f)
        
    return batch



#print(dump_to_batch('../trash/UCG_chiral.lammpstrj', REF=False))

#print(_read_lammps_dump_frames('/project2/andrewferguson/sivadasetty/doe/ucg/chiral_tetramer/SystemSize1000tetramers/T-4.6_freqdouble_racemic/dump_rerun_files/ucg_traj_reformat_joinfrag_vmd_fixed_forces_binary2_chirality.lammpstrj'))
#print(_read_lammps_dump_frames('iter_0/UCG_chiral.lammpstrj')[0]['pos'].shape, _read_lammps_dump_frames('iter_0/UCG_chiral.lammpstrj')[0]['p'].shape)











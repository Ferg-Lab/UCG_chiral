
UCG model of a chiral molecular system
---

This repository contains template simulation files for performing UCG simulations of a chiral molecular system using LAMMPS ([https://git.rcc.uchicago.edu/ndtrung/lammps/-/tree/ucg](https://git.rcc.uchicago.edu/ndtrung/lammps/-/tree/ucg)). Additionally, it contains the relative entropy minimization (REM) codes for parameterizing the UCG model with the all-atom (AA) chiral model as the reference system.

---

Overview of the contents

- simulations
  - master: contains REM codes for UCG parameterization.
    -  init_potentials_4.6: construction of initial guess potentials.
  - create_grid.sh: launches the REM iterations for a given temperature and chiral lag.
- notebooks
  - analysis notebooks.
- lammps_version19Nov2024
  - UCG: LAMMPS codes for simulating chiral molecules. 
    - These codes should be copied into the src directory of LAMMPS version 19 Nov 2024 and then compiled.
    - Required codes:
	  - atom_vec_chiral.cpp, atom_vec_chiral.h: Defines new atom class to track chirality.
      - fix_update_chirality.cpp, fix_update_chirality.h: Evolves chirality of UCG beads.
      - pair_table_ucg_chirality.cpp, pair_table_ucg_chirality.h: Computes UCG substate potential forces using tabulated potentials.

---

Cite

If you use the codes or notebooks from this repo in your work, please cite:

S. Dasetty, T. D. Nguyen, P. Sahrmann, H.B.Runesha,  G. A. Voth, and A. L. Ferguson. "Ultra coarse-grained molecular models of chiral molecular fluids" XXXX (submitted). DOI: XXXX

```
@article{ferglab2026UCGChiral,
  title={Ultra coarse-grained molecular models of chiral molecular fluids,
  author={Dasetty, S. and Nguyen, T.D. and Sahrmann, P. and Runesha, H.B. and Voth, G.A. and Ferguson, A.L.},
  journal={XXXX},
  volume={XXXX},
  number={XXXX},
  pages={XXXX-XXXX},
  year={2026},
  publisher={XXXX}
}
```




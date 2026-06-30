# UCG package

The `UCG` package implements the ultra-coarse-graining models and features as described in the paper:

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

In particular, the package provides the UCG chiral model in the pair style `table/ucg/chiral`.

## Build

```
mkdir build & cd build
cmake ../cmake -C ../cmake/presets/basic.cmake -DPKG_UCG=on
make -j4
```

To build with the GPU package:
```
mkdir build & cd build
cmake ../cmake -C ../cmake/presets/basic.cmake -DPKG_UCG=on -DPKG_GPU=on -DGPU_API=cuda
make -j4
```

## Test

The example input script, lookup tables and configuration files are under `examples/PACKAGES/ucg/chiral`. To test the pair style `table/ucg/chiral` and fix `update/chirality` style, run

```
cd examples/PACKAGES/ucg/chiral
mpirun -np 4 /path/to/build/lmp -in ucg-lmp_chiral.lmp
```

and for the LAMMPS version built with the GPU package, to run on 1 GPU:

```
mpirun -np 4 /path/to/build/lmp -in ucg-lmp_chiral.lmp -sf gpu -pk gpu 1 neigh no
```

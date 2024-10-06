build_nlist[![release](https://github.com/molmod/tinyff/actions/workflows/release.yaml/badge.svg)](https://github.com/molmod/tinyff/actions/workflows/release.yaml)
[![pytest](https://github.com/molmod/tinyff/actions/workflows/pytest.yaml/badge.svg)](https://github.com/molmod/tinyff/actions/workflows/pytest.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/tinyff)](https://pypi.org/project/tinyff/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tinyff)
![GPL-3 License](https://img.shields.io/github/license/molmod/tinyff)

# TinyFF

This is a minimalistic force-field engine written in pure Python,
using vectorized NumPy code.
It has minimal dependencies (NumPy, SciPy, attrs and npy-append-array),
so all the force-field specific code is self-contained.

This little library is geared towards teaching and favors simplicity and conciseness over fancy features and top-notch performance.
TinyFF implements the nitty-gritty of linear-scaling pairwise potentials,
so students can build their own molecular dynamics implementation,
skipping the technicalities of implementing the correct potential energy, pressure and forces acting on atoms.

TinyFF is written by Toon Verstraelen for students of the
[Computtional Physics course (C004504)](https://studiekiezer.ugent.be/2024/studiefiche/en/C004504) in the
[Physics and Astronomy program](https://studiekiezer.ugent.be/2024/master-of-science-in-physics-and-astronomy-CMPHYS-en/programma) at
[Ghent University](https://www.ugent.be/).
TinyFF is distributed under the conditions of the GPL-v3 license.


## Installation

TinyFF is available on PyPI.
In a properly configured Python virtual environment,
you can install TinyFF with:

```bash
pip install tinyff
```

## Features

TinyFF is a Python package with the following modules:

- `tinyff.atomsmithy`: functions for creating initial atomic positions.
- `tinyff.forcefield`: implements a pairwise force field: energy, atomic forces and pressure
- `tinyff.neighborlist`: used by the `forcefield` module to compute pairwise interactions
   with a real-space cut-off.
- `tinyff.trajectory`: tools for writing molecular dynamics trajectories to file(s).
- `tinyff.utils`: utility functions used in `tinyff`.

None of these modules implement any molecular dynamics integrators,
nor any post-processing of molecular dynamics trajectories.
This is the part that you are expected to write.


## Basic usage

### Computing energy, forces and pressure

The evaluation of the force field energy and its derivatives requires the following:

```python
from tinyff.forcefield import CutoffWrapper, LennardJones, PairwiseForceField

# Define a pairwise potential, with energy and force shift
rcut = 8.0
lj = CutOffWrapper(LennardJones(2.5, 0.5), rcut)
# Define a force field
pwff = PairwiseForceFcell_lengthield(lj, rcut))
# You need atomic positions and the length of a periodic cell edge.
# The following line defines just two atomic positions.
atpos = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
# Note that the cell must be large enough to contain the cutoff sphere.
cell_length = 20.0
# Compute stuff:
# - The potential energy.
# - An array with Cartesian forces, same shape as `atpos`.
# - The (virial contribution to) the pressure, only the mechanical part.
potential_energy, forces, mech_pressure = pwff(atpos, cell_length)
```

This basic recipe can be extended by passing additional options into the `PairwiseForceField` constructor:

- Linear-scaling neighbor lists with the linked-cell method, a.k.a. [cell lists](https://en.wikipedia.org/wiki/Cell_lists):

    ```python
    from tinyff.neighborlist import build_nlist_linked_cell

    ...
    pwff = PairwiseForceField(lj, rcut, build_nlist=build_nlist_linked_cell))
    ```

- [Verlet lists](https://en.wikipedia.org/wiki/Verlet_list):

    ```python
    pwff = PairwiseForceField(lj, rcut, rmax=rcut + 3, nlist_reuse=10))
    ```


### Forging initial positions

The `atomsmithy` module can generate a cubic box with standard lattices or randomized atomic positions:

```python
from tinyff.atomsmithy import (
    build_bcc_lattice,
    build_cubic_lattice,
    build_fcc_lattice,
    build_random_cell,
)

# Atoms on a regular lattice. args:
# - primitive cell edge length
# - number of repetitions in X, Y and Z directions.
atpos = build_cubic_lattice(2.5, 2)
atpos = build_bcc_lattice(2.5, 3)
atpos = build_fcc_lattice(2.5, 4)

# Randomize positions. args:
# - cell edge length
# - number of atoms
atpos = build_random_cell(10.0, 32)
```

### Writing trajectories to disk

For visualization with [nglview](https://github.com/nglviewer/nglview),
TinyFF provides a `PDBWriter`, to be used as follows:

```python
from tinyff.trajectory import PDBWriter

# Initialization of the writer: specify a file and a conversion factor to angstrom.
# If the PDB file exists, it is overwritten!
# This example shows the conversion factor when your program works in nanometer.
# Through `atnums` you can specify the chemical elements, here 50 argon atoms (Z=18).
pdb_writer = PDBWriter("trajectory.pdb", to_angstrom=10.0, atnums=[18] * 50)

# Somewhere in your code, typically inside some loop.
# cell_length(s) can be a float or an array of 3 floats.
pdb_writer.dump(atpos, cell_length)

# If you are using Jupyter, you can visualize the trajectory with nglview as follows:
import nglview
nglview.show_file("trajectory.pdb")
```

For numerical post-processing, TinyFF provides a more flexible trajectory writer,
which writes NPY files.
It is implemented with the [`npy-append-array`](https://pypi.org/project/npy-append-array/) library, which makes it possible to extend an NPY file without having to rewrite from scratch.
(That would not be possible with NumPy alone.)
Teh `NPYWriter` can be used as follows:

```python
from tinyff.trajectory import NPYWriter

# Initialization, will create (and possibly clean up an existing) a `traj` directory.
npy_writer = NPYWriter("traj")

# Somewhere in your production code, normally inside some loop.
# You can specify any array or float you like,
# as long as the shape and type is the same upon every call.
# This will result in files `traj/atpos.npy`, `traj/pressure.npy`, etc.
# These files will contain arrays with data passed into all `dump` calls.
npy_writer.dump(atpos=atpos, pressure=pressure, temperature=temperature, ...)

# In your post-processing code
pressure = np.load("traj/pressure.npz")
print(np.mean(pressure))
```

Irrespective of the file format,
it is recommended to write trajectory data only every `nstride` steps,
where `nstride` is a parameter you configure, using something along the following lines:

```python
nstride = 100
for istep in range(10000):
    if istep % nstride == 0:
        # Here comes your code to write a frame to one or more trajectory files.
```
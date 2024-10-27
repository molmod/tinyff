[![release](https://github.com/molmod/tinyff/actions/workflows/release.yaml/badge.svg)](https://github.com/molmod/tinyff/actions/workflows/release.yaml)
[![pytest](https://github.com/molmod/tinyff/actions/workflows/pytest.yaml/badge.svg)](https://github.com/molmod/tinyff/actions/workflows/pytest.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/tinyff)](https://pypi.org/project/tinyff/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tinyff)
![GPL-3 License](https://img.shields.io/github/license/molmod/tinyff)

# TinyFF

This is a minimalistic force-field engine written in pure Python,
using vectorized NumPy code.
It has minimal dependencies (NumPy, SciPy, attrs and npy-append-array),
so all the force-field specific code is self-contained.

This little library is geared towards teaching and favors simplicity and conciseness
over fancy features and top-notch performance.
TinyFF implements the nitty-gritty of linear-scaling pairwise potentials,
so students can build their own molecular dynamics implementation,
skipping the technicalities of implementing the correct
potential energy, pressure and forces acting on atoms.

TinyFF is written by Toon Verstraelen for students of the
[Computational Physics course (C004504)](https://studiekiezer.ugent.be/2024/studiefiche/en/C004504) in the
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
- `tinyff.forcefield`: implements a general force field interface: energy, atomic forces and pressure
- `tinyff.pairwise`: pairwise potentials to be used in force fields.
- `tinyff.neighborlist`: used by the `forcefield` module to compute pairwise interactions
   with a real-space cut-off.
- `tinyff.trajectory`: tools for writing molecular dynamics trajectories to file(s).
- `tinyff.utils`: utility functions used in `tinyff`.

Most relevant functions and classes can be imported directly from the top-level `tinyff` package.

None of these modules implement any molecular dynamics integrators,
nor any post-processing of molecular dynamics trajectories.
This is the part that you are expected to write.


## Basic usage

### Computing energy, forces and pressure

The evaluation of the force field energy and its derivatives requires the following:

```python
import numpy as np
from tinyff import CutOffWrapper, LennardJones, ForceField, NBuildSimple

# Define a pairwise potential, with energy and force shift
rcut = 5.0
lj = CutOffWrapper(LennardJones(2.5, 2.0), rcut)

# Define a force field
ff = ForceField([lj], nbuild=NBuildSimple(rcut))

# You need atomic positions and the length of a periodic cell edge.
# The following line defines just two atomic positions.
atpos = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]])

# Note that the cell must be large enough to contain the cutoff sphere.
cell_length = 20.0

# Compute a selection of results with ff.compute.
#   The ff.compute method has `do_*` arguments to compute only some results:
#   - `do_energy` (default True)
#   - `do_forces` (default False)
#   - `do_press` (default False).
#   Requested results are put in a list, even in only one result is requested.
potential_energy, forces = ff.compute(atpos, cell_length, do_forces=True)
```

This basic recipe can be extended by passing additional options
into the `ForceField` constructor:

- Linear-scaling neighbor lists with the
  [cell lists](https://en.wikipedia.org/wiki/Cell_lists) method:

    ```python
    from tinyff import NBuildCellLists

    # Construct your force field object as follows:
    ff = ForceField([lj], nbuild=NBuildCellLists(rcut, nbin_approx=natom / 100))
    ```

    For about 430 atoms, this becomes more efficient than the simple neighborlist build.

- [Verlet lists](https://en.wikipedia.org/wiki/Verlet_list) (cut-off radius + buffer):

    ```python
    rmax = 6.0  # > rcut, so buffer of 1.0
    ff = ForceField([lj], nbuild=NBuildSimple(rmax, nlist_reuse=16))
    ```


### Forging initial positions

The `atomsmithy` defines functions to generate a cubic box
with standard lattices or randomized atomic positions:

```python
from tinyff import (
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
atpos = build_random_cell(10.0, 32, 2.5)
```

### Writing trajectories to disk

For visualization with [nglview](https://github.com/nglviewer/nglview),
TinyFF provides a `PDBWriter`, to be used as follows:

```python
from tinyff import PDBWriter

# Initialization of the writer: specify a file and a conversion factor to angstrom.
# If the PDB file exists, it is overwritten!
# This example shows the conversion factor when your program works in nanometer.
# Through `atnums` you can specify the chemical elements, here 50 argon atoms (Z=18).
pdb_writer = PDBWriter("trajectory.pdb", to_angstrom=10.0, atnums=[18] * 50)

# Somewhere in your code, typically inside some loop.
# cell_length(s) can be a float or an array of 3 floats.
pdb_writer.dump(atpos, cell_length)

# If you are using Jupyter, you can visualize the trajectory with nglview as follows:
import mdtraj
import nglview
traj = mdtraj.load(f"trajectory.pdb")
view = nglview.show_mdtraj(traj)
view.clear()
view.add_hyperball()
view.add_unitcell()
view
```

For numerical post-processing, TinyFF provides a more flexible trajectory writer,
which writes NPY files.
It is implemented with the [`npy-append-array`](https://pypi.org/project/npy-append-array/) library,
which makes it possible to extend an NPY file without having to rewrite from scratch.
(That would not be possible with NumPy alone.)
The `NPYWriter` can be used as follows:

```python
from tinyff import NPYWriter

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
it is recommended to write trajectory data only every so many steps.
The constructors of both trajectory writers have an optional `stride` argument
to control the frequency of the output.
For example:

```python
npy_writer = NPYWriter("traj", stride=100)
```


### Monte Carlo: partial updates

TinyFF supports partial updates in which the change in energy is computed due to the
displacement of a single atom with the methods `ForceField.try_move` and `ForceField.accept_move`.
These methods assume that a neighborlist was previously built.
They will only recompute results for known pairs of atoms, without adding or removing pairs.
This implies that your Monte Carlo implementation needs to take care of the following:

- Build an initial neighborlist before starting the Monte Carlo loop.
- Construct the neighborlist up to a radius `rmax`, which must extend beyond the cutoff radius.
- Rebuild the neighborlist at regular intervals, e.g. by calling `ForceField.compute`,
  which will also compute the energy by default.
  At this stage it is useful to check whether the new energy is still consistent
  with the sum of all energy changes due to accepted Monte Carlo moves.

The following example just shows how to call the functions.
You can use these function calls in your Monte Carlo loop:

```python
import numpy as np
from tinyff import build_fcc_lattice, CutOffWrapper, LennardJones, ForceField, NBuildSimple

# System configuration, a simple (inflated) FCC lattice of Argon atoms.
atpos = build_fcc_lattice(2.5, 4)
cell_lengths = np.array([10.0, 10.0, 10.0])
rmax = 3.0
rcut = 2.5

# Define the force field and compute the initial energy.
lj = CutOffWrapper(LennardJones(2.5, 2.0), rcut)
ff = ForceField([lj], nbuild=NBuildSimple(rmax))
energy0, = ff.compute(atpos, cell_length)

# Try and accept a move of atom 3.
iatom = 3
delta = np.array([0.1, 0.2, -0.1])
energy_change, move = ff.try_move(iatom, delta, cell_lengths)
ff.accept_move(move)
atpos[iatom] += delta

# Verify the change in energy.
energy1, = ff.compute(atpos, cell_length)
assert abs(energy_change - (energy1 - energy0)) < 1e-10
```


## Change log

The history of changes can be found in [CHANGELOG.md](CHANGELOG.md).

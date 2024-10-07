# TinyFF is a minimalistic Force Field evaluator.
# Copyright (C) 2024 Toon Verstraelen
#
# This file is part of TinyFF.
#
# TinyFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# TinyFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Tools to build initial atomic positions."""

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .forcefield import PairPotential, PairwiseForceField

__all__ = (
    "build_general_cubic_lattice",
    "build_cubic_lattice",
    "build_bcc_lattice",
    "build_fcc_lattice",
    "build_random_cell",
)


def build_general_cubic_lattice(prim_frpos: ArrayLike, prim_length: float, nrep: int) -> NDArray:
    """Build a cubic simulation cell by repeating the primitive cell.

    Parameters
    ----------
    prim_frpos
        Positions of atoms in the primitive cell, in fractional coordinates.
    prim_length
        The length of a primitive cubic cell edge.
    nrep
        The number of times to repeat the primitive cell
        (in three directions).

    Returns
    -------
    atpos
        Atomic positions, array with shape (natom, 3).
    """
    prim_frpos = np.asarray(prim_frpos)
    if prim_frpos.ndim != 2:
        raise TypeError("prim_frpos must be a 2D array")
    if prim_frpos.shape[1] != 3:
        raise TypeError("prim_frpos must have three columns")
    onedim_grid = np.arange(nrep) * prim_length
    xv, yv, zv = np.meshgrid(onedim_grid, onedim_grid, onedim_grid, indexing="ij")
    prim_pos = np.stack((xv.ravel(), yv.ravel(), zv.ravel()), axis=1)
    return np.concatenate([prim_pos + frpos * prim_length for frpos in prim_frpos])


def build_cubic_lattice(prim_length: float, nrep: int):
    """Build a simple cubic lattice with given cell length and number of repetitions."""
    return build_general_cubic_lattice([[0.0, 0.0, 0.0]], prim_length, nrep)


def build_bcc_lattice(prim_length: float, nrep: int):
    """Build a simple cubic lattice with given cell length and number of repetitions."""
    return build_general_cubic_lattice([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], prim_length, nrep)


def build_fcc_lattice(prim_length: float, nrep: int):
    """Build a simple cubic lattice with given cell length and number of repetitions."""
    prim_frpos = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    return build_general_cubic_lattice(prim_frpos, prim_length, nrep)


@attrs.define
class PushPotential(PairPotential):
    """Simple and cheap repulsive potential that smoothly goes to zero at cutoff."""

    rcut: float = attrs.field(converter=float, validator=attrs.validators.gt(0))

    def __call__(self, dist: ArrayLike) -> tuple[NDArray, NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        dist = np.asarray(dist, dtype=float)
        x = dist / self.rcut
        energy = (x - 1) ** 2
        gdist = 2 * (x - 1) / self.rcut
        return energy, gdist


def build_random_cell(
    cell_length: float,
    natom: int,
    rcut: float,
    *,
    maxiter: int = 100,
    rng: np.random.Generator | None = None,
):
    """Fill a cell with randomly placed atoms, avoiding close contacts."""
    # Start with completely random
    if rng is None:
        rng = np.random.default_rng()
    atpos0 = rng.uniform(0, cell_length, (natom, 3))

    # Define cost function to push the atoms appart.
    pwff = PairwiseForceField(PushPotential(rcut), rcut)

    def costgrad(atpos_raveled):
        atpos = atpos_raveled.reshape(-1, 3)
        energy, force, _ = pwff(atpos, cell_length)
        return energy, -force.ravel()

    # Optimize and return structure
    sol = minimize(costgrad, atpos0.ravel(), jac=True, options={"maxiter": maxiter})
    return sol.x.reshape(-1, 3)

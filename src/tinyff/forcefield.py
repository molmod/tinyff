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
"""Basic Force Field models."""

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .neighborlist import NBuild

__all__ = ("PairwiseForceField",)


@attrs.define
class PairPotential:
    def __call__(self, dist: ArrayLike) -> tuple[NDArray, NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        raise NotImplementedError  # pragma: nocover


@attrs.define
class LennardJones(PairPotential):
    epsilon: float = attrs.field(converter=float)
    sigma: float = attrs.field(converter=float)

    def __call__(self, dist: ArrayLike) -> tuple[NDArray, NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        dist = np.asarray(dist, dtype=float)
        x = self.sigma / dist
        energy = (4 * self.epsilon) * (x**12 - x**6)
        gdist = (-4 * self.epsilon * self.sigma) * (12 * x**11 - 6 * x**5) / dist**2
        return energy, gdist


@attrs.define
class CutOffWrapper(PairPotential):
    original: PairPotential = attrs.field()
    rcut: float = attrs.field(converter=float)
    ecut: float = attrs.field(init=False, default=0.0, converter=float)
    gcut: float = attrs.field(init=False, default=0.0, converter=float)

    def __attrs_post_init__(self):
        """Post initialization changes."""
        self.ecut, self.gcut = self.original(self.rcut)

    def __call__(self, dist: ArrayLike) -> tuple[NDArray, NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        dist = np.asarray(dist, dtype=float)
        mask = dist < self.rcut
        if mask.ndim == 0:
            # Deal with non-array case
            if mask:
                energy, gdist = self.original(dist)
                energy -= self.ecut + self.gcut * (dist - self.rcut)
                gdist -= self.gcut
            else:
                energy = 0.0
                gdist = 0.0
        else:
            energy, gdist = self.original(dist)
            energy[mask] -= self.ecut + self.gcut * (dist[mask] - self.rcut)
            energy[~mask] = 0.0
            gdist[mask] -= self.gcut
            gdist[~mask] = 0.0
        return energy, gdist


@attrs.define
class PairwiseForceField:
    pair_potential: PairPotential = attrs.field(
        validator=attrs.validators.instance_of(PairPotential)
    )
    """A definition of the pair potential."""

    nbuild: NBuild = attrs.field(validator=attrs.validators.instance_of(NBuild))
    """Algorithm to build the neigborlist."""

    def __call__(self, atpos: NDArray, cell_length: float):
        """Compute microscopic properties related to the potential energy.

        Parameters
        ----------
        atpos
            Atomic positions, one atom per row.
            Array shape = (natom, 3).
        cell_length
            The length of the edge of the cubic simulation cell.

        Returns
        -------
        energy
            The potential energy.
        forces
            The forces acting on the atoms, same shape as atpos.
        frc_pressure
            The force-contribution to the pressure,
            i.e. usually the second term of the virial stress in most text books.
        """
        # Bring neighborlist up to date.
        self.nbuild.update(atpos, cell_length)
        nlist = self.nbuild.nlist
        # Compute all pairwise quantities
        nlist["energy"], nlist["gdist"] = self.pair_potential(nlist["dist"])
        nlist["gdelta"] = (nlist["gdist"] / nlist["dist"]).reshape(-1, 1) * nlist["delta"]
        # Compute the totals
        energy = nlist["energy"].sum()
        forces = np.zeros(atpos.shape, dtype=float)
        np.add.at(forces, nlist["iatom0"], nlist["gdelta"])
        np.add.at(forces, nlist["iatom1"], -nlist["gdelta"])
        frc_pressure = -np.dot(nlist["gdist"], nlist["dist"]) / (3 * cell_length**3)
        return energy, forces, frc_pressure

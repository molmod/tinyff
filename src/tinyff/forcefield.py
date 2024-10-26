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

from .neighborlist import NLIST_DTYPE, NBuild

__all__ = ("ForceTerm", "PairPotential", "LennardJones", "CutOffWrapper", "ForceField")


@attrs.define
class ForceTerm:
    def compute_nlist(
        self, nlist: NDArray[NLIST_DTYPE], do_energy: bool = True, do_gdist: bool = False
    ):
        """Compute energies and derivatives and add them to the neighborlist."""
        raise NotImplementedError  # pragma: nocover


@attrs.define
class PairPotential:
    def compute_nlist(
        self, nlist: NDArray[NLIST_DTYPE], do_energy: bool = True, do_gdist: bool = False
    ):
        """Compute energies and derivatives and add them to the neighborlist."""
        results = self.compute(nlist["dist"], do_energy, do_gdist)
        if do_energy:
            nlist["energy"] += results.pop(0)
        if do_gdist:
            nlist["gdist"] += results.pop(0)

    def compute(
        self, dist: ArrayLike, do_energy: bool = True, do_gdist: bool = False
    ) -> list[NDArray | None]:
        """Compute pair potential energy and its derivative towards distance."""
        raise NotImplementedError  # pragma: nocover


@attrs.define
class LennardJones(PairPotential):
    epsilon: float = attrs.field(converter=float)
    sigma: float = attrs.field(converter=float)

    def compute(
        self, dist: ArrayLike, do_energy: bool = True, do_gdist: bool = False
    ) -> list[NDArray | None]:
        """Compute pair potential energy and its derivative towards distance."""
        dist = np.asarray(dist, dtype=float)
        x = self.sigma / dist
        result = []
        if do_energy:
            energy = (4 * self.epsilon) * (x**12 - x**6)
            result.append(energy)
        if do_gdist:
            gdist = (-4 * self.epsilon * self.sigma) * (12 * x**11 - 6 * x**5) / dist**2
            result.append(gdist)
        return result


@attrs.define
class CutOffWrapper(PairPotential):
    original: PairPotential = attrs.field()
    rcut: float = attrs.field(converter=float)
    ecut: float = attrs.field(init=False, default=0.0, converter=float)
    gcut: float = attrs.field(init=False, default=0.0, converter=float)

    def __attrs_post_init__(self):
        """Post initialization changes."""
        self.ecut, self.gcut = self.original.compute(self.rcut, do_gdist=True)

    def compute(
        self, dist: ArrayLike, do_energy: bool = True, do_gdist: bool = False
    ) -> list[NDArray | None]:
        """Compute pair potential energy and its derivative towards distance."""
        dist = np.asarray(dist, dtype=float)
        mask = dist < self.rcut
        results = []
        if mask.ndim == 0:
            # Deal with non-array case
            if mask:
                orig_results = self.original.compute(dist, do_energy, do_gdist)
                if do_energy:
                    energy = orig_results.pop(0)
                    energy -= self.ecut + self.gcut * (dist - self.rcut)
                    results.append(energy)
                if do_gdist:
                    gdist = orig_results.pop(0)
                    gdist -= self.gcut
                    results.append(gdist)
            else:
                if do_energy:
                    results.append(0.0)
                if do_gdist:
                    results.append(0.0)
        else:
            orig_results = self.original.compute(dist, do_energy, do_gdist)
            if do_energy:
                energy = orig_results.pop(0)
                energy[mask] -= self.ecut + self.gcut * (dist[mask] - self.rcut)
                energy[~mask] = 0.0
                results.append(energy)
            if do_gdist:
                gdist = orig_results.pop(0)
                gdist[mask] -= self.gcut
                gdist[~mask] = 0.0
                results.append(gdist)
        return results


@attrs.define
class ForceField:
    force_terms: list[ForceTerm] = attrs.field()
    """A list of contributions to the potential energy."""

    nbuild: NBuild = attrs.field(validator=attrs.validators.instance_of(NBuild))
    """Algorithm to build the neigborlist."""

    def __call__(self, atpos: NDArray, cell_length: float):
        """Compute the potential energy, atomic forces and the force contribution to the pressure.

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
        return self.compute(atpos, cell_length, do_forces=True, do_press=True)

    def compute(
        self,
        atpos: NDArray,
        cell_length: float,
        do_energy: bool = True,
        do_forces: bool = False,
        do_press: bool = False,
    ):
        """Compute microscopic properties related to the potential energy.

        Parameters
        ----------
        atpos
            Atomic positions, one atom per row.
            Array shape = (natom, 3).
        cell_length
            The length of the edge of the cubic simulation cell.
        do_energy
            if True, the energy is returned.
        do_forces
            if True, the atomic forces are returned.
        do_press
            if True, the force contribution to the pressure is returned.

        Returns
        -------
        results
            A list containing the requested values.
        """
        # Bring neighborlist up to date.
        self.nbuild.update(atpos, cell_length)
        nlist = self.nbuild.nlist

        # Compute all pairwise quantities, if needed with derivatives.
        for force_term in self.force_terms:
            force_term.compute_nlist(nlist, do_energy, do_forces | do_press)

        # Compute the totals
        results = []
        if do_energy:
            energy = nlist["energy"].sum()
            results.append(energy)
        if do_forces or do_press:
            nlist["gdelta"] = (nlist["gdist"] / nlist["dist"]).reshape(-1, 1) * nlist["delta"]
            if do_forces:
                atfrc = np.zeros(atpos.shape, dtype=float)
                np.add.at(atfrc, nlist["iatom0"], nlist["gdelta"])
                np.add.at(atfrc, nlist["iatom1"], -nlist["gdelta"])
                results.append(atfrc)
            if do_press:
                frc_press = -np.dot(nlist["gdist"], nlist["dist"]) / (3 * cell_length**3)
                results.append(frc_press)

        return results

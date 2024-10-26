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
    ) -> list[NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        raise NotImplementedError  # pragma: nocover


@attrs.define
class LennardJones(PairPotential):
    epsilon: float = attrs.field(converter=float)
    sigma: float = attrs.field(converter=float)

    def compute(
        self, dist: ArrayLike, do_energy: bool = True, do_gdist: bool = False
    ) -> list[NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        results = []
        dist = np.asarray(dist, dtype=float)
        x = self.sigma / dist
        x2 = x * x
        x3 = x2 * x
        x5 = x2 * x3
        x6 = x3 * x3
        if do_energy:
            energy = (4 * self.epsilon) * ((x6 - 1) * x6)
            results.append(energy)
        if do_gdist:
            gdist = (-48 * self.epsilon * self.sigma) * ((x6 - 0.5) * x5 / dist / dist)
            results.append(gdist)
        return results


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
    ) -> list[NDArray]:
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
                energy -= self.ecut + self.gcut * (dist - self.rcut)
                energy *= mask
                results.append(energy)
            if do_gdist:
                gdist = orig_results.pop(0)
                gdist -= self.gcut
                gdist *= mask
                results.append(gdist)
        return results


@attrs.define
class Move:
    """All information needed to update the force field internals after accepting a MC move."""

    select: NDArray[int] = attrs.field()
    """Indexes of the rows in the neighborlist involving the moved atom."""

    nlist: NDArray[NLIST_DTYPE] = attrs.field()
    """Part of the neighborlist corresponding to select, with changes due to the trial step."""


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
        cell_lengths: ArrayLike | float,
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
            The length of the edge of the cubic simulation cell,
            or an array of lengths of three cell vectors.
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
        cell_lengths = self.nbuild.update(atpos, cell_lengths)
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
                np.subtract.at(atfrc, nlist["iatom1"], nlist["gdelta"])
                np.add.at(atfrc, nlist["iatom0"], nlist["gdelta"])
                results.append(atfrc)
            if do_press:
                frc_press = -np.dot(nlist["gdist"], nlist["dist"]) / (3 * cell_lengths.prod())
                results.append(frc_press)

        return results

    def try_move(self, iatom: int, delta: NDArray[float], cell_lengths: NDArray[float]):
        """Try moving one atom and compute the change in energy.

        Parameters
        ----------
        iatom
            The atom to move.
        delta
            The displacement vector.
        cell_lengths
            An array (with 3 elements) defining the size of the simulation cell.

        Returns
        -------
        energy_change
            The change in energy due to the displacement of the atom.
        move
            Information to passed on the method `accept_move` to upate the internal
            state of the force field after the move was accepted.
            When not calling `accept_move`, it is assumed that the move was rejected.
        """
        select, nlist = self.nbuild.try_move(iatom, delta, cell_lengths)

        # Copy the old energy still present in the neighborlist
        energy_old = nlist["energy"].sum()

        # Clear results from the neighborlists and compute energy.
        nlist["energy"] = 0.0
        nlist["gdist"] = 0.0
        for force_term in self.force_terms:
            force_term.compute_nlist(nlist)

        # Prepare return values
        energy_new = nlist["energy"].sum()
        return energy_new - energy_old, Move(select, nlist)

    def accept_move(self, move: Move):
        """Update the internal state of the force field object after accepting a move.

        If a move is rejected, simply do not call this method.

        Parameters
        ----------
        move
            The second return value the `try_move` method.
        """
        self.nbuild.nlist[move.select] = move.nlist

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
from .pairwise import PairwiseTerm
from .utils import parse_atpos

__all__ = ("ForceField", "Move")


@attrs.define
class Move:
    """All information needed to update the force field internals after accepting a MC move."""

    select: NDArray[int] = attrs.field()
    """Indexes of the rows in the neighborlist involving the moved atom."""

    nlist: NDArray[NLIST_DTYPE] = attrs.field()
    """Part of the neighborlist corresponding to select, with changes due to the trial step."""


@attrs.define
class ForceField:
    """A force field defined as a sum of pairwise terms, evaluated with a neighborlist."""

    pairwise_terms: list[PairwiseTerm] = attrs.field()
    """A list of contributions to the potential energy."""

    nbuild: NBuild = attrs.field(validator=attrs.validators.instance_of(NBuild), kw_only=True)
    """Algorithm to build the neighborlist."""

    def compute(self, atpos: ArrayLike, cell_lengths: ArrayLike | float, nderiv: int = 0):
        """Compute microscopic properties related to the potential energy.

        Parameters
        ----------
        atpos
            Atomic positions, one atom per row.
            Array shape = (natom, 3).
        cell_lengths
            The length of the edge of the cubic simulation cell,
            or an array of lengths of three cell vectors.
        nderiv
            The order of derivatives to compute, either 0 (energy)
            or 1 (energy, forces and pressure).

        Returns
        -------
        results
            A list containing the requested values.
        """
        # Bring neighborlist up to date.
        atpos = parse_atpos(atpos)
        cell_lengths = self.nbuild.update(atpos, cell_lengths)
        nlist = self.nbuild.nlist

        # Compute all pairwise quantities, if needed with derivatives.
        for pairwise_term in self.pairwise_terms:
            pairwise_term.compute_nlist(nlist, nderiv)

        # Compute the totals
        results = []
        energy = nlist["energy"].sum()
        results.append(energy)
        if nderiv >= 1:
            nlist["gdelta"] = (nlist["gdist"] / nlist["dist"]).reshape(-1, 1) * nlist["delta"]
            # Accumulate pairwise forces on atoms, one Cartesian component at a time.
            # (np.bincount is faster than the equivalent np.add.at and np.subtract.at.)
            natom = len(atpos)
            atfrc = np.stack(
                [
                    np.bincount(nlist["iatom0"], nlist["gdelta"][:, k], natom)
                    - np.bincount(nlist["iatom1"], nlist["gdelta"][:, k], natom)
                    for k in range(3)
                ],
                axis=1,
            )
            results.append(atfrc)
            frc_press = -np.dot(nlist["gdist"], nlist["dist"]) / (3 * cell_lengths.prod())
            results.append(frc_press)

        return results

    def try_move(self, iatom: int, delta: ArrayLike, cell_lengths: ArrayLike | float):
        """Try moving one atom and compute the change in energy.

        The energy change is computed with the pairwise energies stored in the neighborlist.
        These are only up to date if no atoms were moved since the last call to `compute`,
        except for moves accepted with `accept_move`.
        Only pairs already present in the neighborlist are considered.
        Pairs that come within range due to the move are missed.

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
            Information to be passed on to the method `accept_move` to update the internal
            state of the force field after the move was accepted.
            When not calling `accept_move`, it is assumed that the move was rejected.
        """
        select, nlist = self.nbuild.try_move(iatom, delta, cell_lengths)

        # Copy the old energy still present in the neighborlist
        energy_old = nlist["energy"].sum()

        # Clear results from the neighborlists and compute energy.
        nlist["energy"] = 0.0
        nlist["gdist"] = 0.0
        for pairwise_term in self.pairwise_terms:
            pairwise_term.compute_nlist(nlist)

        # Prepare return values
        energy_new = nlist["energy"].sum()
        return energy_new - energy_old, Move(select, nlist)

    def accept_move(self, move: Move):
        """Update the internal state of the force field object after accepting a move.

        If a move is rejected, simply do not call this method.
        This method only updates the neighborlist.
        The caller must also apply the displacement to its own array of atomic positions.

        Parameters
        ----------
        move
            The second return value of the `try_move` method.
        """
        self.nbuild.nlist[move.select] = move.nlist

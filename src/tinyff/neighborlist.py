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
"""Basic Neighborlists."""

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import parse_atpos, parse_cell_lengths

__all__ = ("NLIST_DTYPE", "NBuildSimple", "NBuildCellLists")


NLIST_DTYPE = [
    # First atoms.
    ("iatom0", int),
    # Second atom.
    ("iatom1", int),
    # Relative vector from 0 to 1.
    ("delta", float, 3),
    # Derivative of the energy with respect to the relative vector.
    ("gdelta", float, 3),
    # Distance between the atoms.
    ("dist", float),
    # Derivative of the energy with respect to the distance.
    ("gdist", float),
    # Energy of the pairwise interaction.
    ("energy", float),
]


@attrs.define
class NBuild:
    """Base class for neighborlist building algorithms."""

    rmax: float = attrs.field(
        converter=float, on_setattr=attrs.setters.frozen, validator=attrs.validators.gt(0)
    )
    """Maximum distances retained in the neighborlist.

    Note that the corresponding sphere must fit in the simulation cell.
    """

    nlist: NDArray[NLIST_DTYPE] | None = attrs.field(default=None, init=False)
    """The current neighborlist."""

    nlist_reuse: int = attrs.field(converter=int, default=0, kw_only=True)
    """Number of times the neighbor list is recomputed without rebuilding."""

    _nlist_use_count: int = attrs.field(converter=int, default=0, init=False)
    """Internal counter to decide when to rebuild neigborlist."""

    _atom_cache: dict[int] = attrs.field(init=False, factory=dict)

    @property
    def nlist_use_count(self):
        """The number of times the current neighborlist will be reused in future calculations."""
        return self._nlist_use_count

    def update(self, atpos: ArrayLike, cell_lengths: ArrayLike) -> NDArray[float]:
        """Rebuild or recompute the neighbor list.

        Parameters
        ----------
        atpos
            Atomic positions, one atom per row.
            Array shape = (natom, 3).
        cell_lengths
            The lengths of a periodic orthorombic box.

        Returns
        -------
        cell_lengths
            The interpreted version of cell_lengths,
            guaranteed to be an array with three elements.
        """
        # Rebuild or recompute the neighborlist
        if self._nlist_use_count <= 1:
            self.nlist = None
        else:
            self._nlist_use_count -= 1
        if self.nlist is None:
            cell_lengths = self._rebuild(atpos, cell_lengths)
            self._nlist_use_count = self.nlist_reuse
            self._atom_cache = {}
        else:
            cell_lengths = self._recompute(atpos, cell_lengths)
        return cell_lengths

    def _rebuild(self, atpos: ArrayLike, cell_lengths: ArrayLike) -> NDArray[float]:
        """Build the neighborlist array from scratch, possibly identifying new pairs."""
        raise NotImplementedError

    def _recompute(self, atpos: ArrayLike, cell_lengths: ArrayLike) -> NDArray[float]:
        """Recompute deltas and distances and reset other parts of the neighborlist in-place."""
        # Process parameters.
        atpos = parse_atpos(atpos)
        cell_lengths = parse_cell_lengths(cell_lengths, self.rmax)

        # Do some work.
        self.nlist["delta"] = atpos[self.nlist["iatom1"]] - atpos[self.nlist["iatom0"]]
        self.nlist["dist"] = _apply_mic(self.nlist["delta"], cell_lengths)

        # Reset outdated fields in the neigborlist.
        self.nlist["gdelta"] = 0.0
        self.nlist["gdist"] = 0.0
        self.nlist["energy"] = 0.0

        return cell_lengths

    def try_move(self, iatom: int, delta: NDArray[float], cell_lengths: ArrayLike | float):
        """Compute a subset of the neighborlist after displacing one atom.

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
        select
            The indexes of the global neighborlist that were updated.
        nlist
            The modified subset of the neigborlist.
        """
        if not isinstance(iatom, int):
            raise TypeError("The argument iatom must be an integer.")
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (3,):
            raise TypeError("The displacement vector delta must have shape (3,).")
        cell_lengths = parse_cell_lengths(cell_lengths)

        # Find the related rows in the neighborlist
        info = self._atom_cache.get(iatom)
        if info is None:
            select0 = np.where(self.nlist["iatom0"] == iatom)[0]
            select1 = np.where(self.nlist["iatom1"] == iatom)[0]
            select = np.concatenate([select0, select1])
            signs = np.ones(len(select))
            signs[: len(select0)] = -1
            signs.shape = (-1, 1)
            self._atom_cache[iatom] = (select, signs)
        else:
            select, signs = info

        # Update the copied fragments of the neighborlist with the displacement.
        nlist = self.nlist.take(select)
        nlist["delta"] += delta * signs
        nlist["dist"] = _apply_mic(nlist["delta"], cell_lengths)
        return select, nlist


def _apply_mic(deltas: NDArray[float], cell_lengths: NDArray[float]):
    """Apply the minimum image convention to the deltas and compute the distances.

    Parameters
    ----------
    deltas
        Relative vectors to which the minimum image convention must be applied,
        an array with shape (natom, 3) in which each row is one relative vector.
        The vectors are modified in place.
    cell_lengths
        The lengths of a periodic orthorombic box.

    Returns
    -------
    dists
        An array with lengths of the (updated) relative vectors.
    """
    deltas /= cell_lengths
    deltas -= np.round(deltas)
    deltas *= cell_lengths
    dist = np.einsum("ij,ij->i", deltas, deltas)
    np.sqrt(dist, out=dist)
    return dist


@attrs.define
class NBuildSimple(NBuild):
    def _rebuild(self, atpos: ArrayLike, cell_lengths: ArrayLike):
        """Build the neighborlist array from scratch, possibly identifying new pairs."""
        # Parse parameters
        atpos = parse_atpos(atpos)
        cell_lengths = parse_cell_lengths(cell_lengths, self.rmax)

        # Generate arrays with all pairs below the cutoff.
        iatoms0, iatoms1, deltas, dists = _create_parts_self(atpos, None, cell_lengths, self.rmax)

        # Apply cutoff and put everything in a fresh neigborlist.
        self.nlist = np.zeros(len(dists), dtype=NLIST_DTYPE)
        self.nlist["iatom0"] = iatoms0
        self.nlist["iatom1"] = iatoms1
        self.nlist["delta"] = deltas
        self.nlist["dist"] = dists

        return cell_lengths


@attrs.define
class NBuildCellLists(NBuild):
    nbin_approx: float = attrs.field()

    def _rebuild(self, atpos: ArrayLike, cell_lengths: ArrayLike):
        """Build a neighborlist with linked cell algorithm."""
        atpos = parse_atpos(atpos)
        cell_lengths = parse_cell_lengths(cell_lengths, self.rmax)

        # Group the atoms into bins
        nbins = _determine_nbins(cell_lengths, self.rmax, self.nbin_approx)
        bins = _assign_atoms_to_bins(atpos, cell_lengths, nbins)

        # Loop over pairs of nearby bins and collect parts for neighborlist.
        iatoms0_parts = []
        iatoms1_parts = []
        deltas_parts = []
        dists_parts = []
        for idx0, bin0 in bins.items():
            parts = [_create_parts_self(atpos, bin0, cell_lengths, self.rmax)]
            for idx1 in _iter_nearby(idx0, nbins):
                bin1 = bins.get(idx1)
                if bin1 is not None:
                    parts.append(_create_parts_nearby(atpos, bin0, bin1, cell_lengths, self.rmax))
            for iatoms0, iatoms1, deltas, dists in parts:
                if len(dists) > 0:
                    iatoms0_parts.append(iatoms0)
                    iatoms1_parts.append(iatoms1)
                    deltas_parts.append(deltas)
                    dists_parts.append(dists)

        # Put everything in a neighborlist array.
        if len(dists_parts) == 0:
            self.nlist = np.zeros(0, dtype=NLIST_DTYPE)
        else:
            dists = np.concatenate(dists_parts)
            self.nlist = np.zeros(len(dists), dtype=NLIST_DTYPE)
            self.nlist["iatom0"] = np.concatenate(iatoms0_parts)
            self.nlist["iatom1"] = np.concatenate(iatoms1_parts)
            self.nlist["delta"] = np.concatenate(deltas_parts)
            self.nlist["dist"] = dists

        return cell_lengths


def _determine_nbins(cell_lengths: NDArray[float], rmax: float, nbin_approx: float) -> NDArray[int]:
    """Determine the number of bins, aiming for a given number of atoms per bin.

    Parameters
    ----------
    cell_lengths
        The lengths of a periodic orthorombic box.
    rmax
        The maximum distance between atoms in the neighborlist.
        It is guaranteed that the opposite faces of a bin are separated
        by a distance not less than rmax.
    nbin_approx
        The target number of bins, may be a floating point number.
        For example, the number of atoms divided by 100.

    Returns
    -------
    nbins
        An array with three values: the number of bins along each Cartesian axis.
        The number of bins is at least two.
    """
    nbin_volume = np.prod(cell_lengths) / nbin_approx
    bin_width = max(nbin_volume ** (1 / 3), rmax)
    return np.floor(np.clip(cell_lengths / bin_width, 2, np.inf)).astype(int)


def _assign_atoms_to_bins(
    atpos: NDArray[float], cell_lengths: NDArray[float], nbins: NDArray[int]
) -> dict[tuple[int, int, int], NDArray[int]]:
    """Create arrays of atom indexes for each bin in the cell.

    Parameters
    ----------
    atpos
        Atomic positions, one atom per row.
        Array shape = (natom, 3).
    cell_lengths
        The lengths of a periodic orthorombic box.
    nbins
        The number of bins along each cell axis.

    Returns
    -------
    bins
        A dictionary whose keys are 3-tuples of integer bin indexes and
        whose values are arrays of atom indexes in the corresponding bins.
    """
    if (nbins < 2).any():
        raise ValueError("The cutoff radius is too large for the given cell lengths.")
    idxs = np.floor(atpos / (cell_lengths / nbins)).astype(int) % nbins
    flat_idxs = (idxs[:, 0] * nbins[1] + idxs[:, 1]) * nbins[2] + idxs[:, 2]
    _flat_unique, firsts, inverse = np.unique(flat_idxs, return_index=True, return_inverse=True)
    return {
        tuple(int(idx) for idx in idxs[first]): (inverse == i).nonzero()[0]
        for i, first in enumerate(firsts)
    }


def _create_parts_self(
    atpos: NDArray[float], bin0: NDArray[int] | None, cell_lengths: NDArray[float], rmax: float
):
    """Prepare parts of a neighborlist for pairs within one cell or bin.

    Parameters
    ----------
    atpos
        Atomic positions, one atom per row.
        Array shape = (natom, 3).
    bin0
        A list of atom indexes to consider (or None if all are relevant.)
    cell_lengths
        The lengths of a periodic orthorombic box.
    rmax
        The maximum radioius, i.e. the cut-off radius for the neighborlist.
        Note that the corresponding sphere must fit in the simulation cell.

    Returns
    -------
    iatoms0, iatoms1
        Indexes of atom pairs.
    deltas
        Relative vectors pointing from 0 to 1.
    dists
        Distances between pairs.
    """
    if bin0 is None:
        natom = atpos.shape[0]
        iatoms0 = np.tile(np.arange(natom), natom)
        iatoms1 = np.repeat(np.arange(natom), natom)
    else:
        iatoms0 = np.tile(bin0, len(bin0))
        iatoms1 = np.repeat(bin0, len(bin0))
    mask = iatoms0 < iatoms1
    iatoms0 = iatoms0[mask]
    iatoms1 = iatoms1[mask]
    deltas = atpos[iatoms1] - atpos[iatoms0]
    dists = _apply_mic(deltas, cell_lengths)
    mask = dists <= rmax
    return iatoms0[mask], iatoms1[mask], deltas[mask], dists[mask]


NEARBY = [
    (-1, -1, -1),
    (-1, 0, -1),
    (-1, 1, -1),
    (0, -1, -1),
    (0, 0, -1),
    (0, 1, -1),
    (1, -1, -1),
    (1, 0, -1),
    (1, 1, -1),
    (-1, -1, 0),
    (0, -1, 0),
    (1, -1, 0),
    (-1, 0, 0),
]


def _iter_nearby(idx, nbins):
    """Iterate over nearby bins in 3D.

    Parameters
    ----------
    idx
        Tuple of three integer indexes identifying a bin.
        The neighbors of this bin will be iterated over.
    nbins
        A vector with the number of bins along each dimension, shape == (3,).
        (This is used to impose periodic boundary conditions.)

    Yields
    ------
    idx_nearby
        A tuple with integer bin indexes of nearby bins.
        Only half of them are considered to avoid double counting.
    """

    def skip(ni, i, di):
        return ni == 2 and ((i == 0 and di == -1) or (i == 1 and di == 1))

    a, b, c = idx
    na, nb, nc = nbins
    for da, db, dc in NEARBY:
        if not (skip(na, a, da) or skip(nb, b, db) or skip(nc, c, dc)):
            yield (a + da) % na, (b + db) % nb, (c + dc) % nc


def _create_parts_nearby(
    atpos: NDArray[float],
    bin0: NDArray[int],
    bin1: NDArray[int],
    cell_lengths: NDArray[float],
    rmax: float,
):
    """Prepare parts of a neighborlist for pairs in nearby cells.

    Parameters
    ----------
    atpos
        The array with all atomic positions. Shape is (natom, 3).
    bin0
        Atom indexes in the current bin.
    bin1
        Atom indexes in the other nearby bin.
    cell_lengths
        The lengths of the periodic cell edges.
    rmax
        The maximum radioius, i.e. the cut-off radius for the neighborlist.
        Note that the corresponding sphere must fit in the simulation cell.

    Returns
    -------
    iatoms0, iatoms1
        Indexes of atom pairs.
    deltas
        Relative vectors pointing from 0 to 1.
    dists
        Distances between pairs.
    """
    iatoms0 = np.repeat(bin0, len(bin1))
    iatoms1 = np.tile(bin1, len(bin0))
    deltas = atpos[iatoms1] - atpos[iatoms0]
    dists = _apply_mic(deltas, cell_lengths)
    mask = dists <= rmax
    return iatoms0[mask], iatoms1[mask], deltas[mask], dists[mask]

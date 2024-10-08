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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import parse_atpos, parse_cell_lengths, parse_rmax

__all__ = ("NLIST_DTYPE", "build_nlist_simple", "build_nlist_linked_cell", "recompute_nlist")

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


def build_nlist_simple(
    atpos: ArrayLike, cell_lengths: ArrayLike, rmax: float
) -> NDArray[NLIST_DTYPE]:
    """Build a neighborlist with a simple N^2 scaling algorithm.

    Parameters
    ----------
    atpos
        Atomic positions, one atom per row.
        Array shape = (natom, 3).
    cell_lengths
        The lengths of a periodic orthorombic box.
    rmax
        The maximum radioius, i.e. the cut-off radius for the neighborlist.
        Note that the corresponding sphere must fit in the simulation cell.

    Returns
    -------
    neighborlist
        Structured array with neighborlist, including pairs up to distance rmax.
    """
    # Process parameters
    atpos = parse_atpos(atpos)
    cell_lengths = parse_cell_lengths(cell_lengths)
    rmax = parse_rmax(rmax, cell_lengths)

    # Generate arrays with all pairs below the cutoff.
    iatoms0, iatoms1, deltas, dists = _create_parts_self(atpos, None, cell_lengths, rmax)

    # Apply cutoff and put everything in a fresh neigborlist.
    nlist = np.zeros(len(dists), dtype=NLIST_DTYPE)
    nlist["iatom0"] = iatoms0
    nlist["iatom1"] = iatoms1
    nlist["delta"] = deltas
    nlist["dist"] = dists
    return nlist


def _mic(
    atpos: NDArray[float],
    iatoms0: NDArray[int],
    iatoms1: NDArray[int],
    cell_lengths: NDArray[float],
) -> tuple[NDArray[float], NDArray[float]]:
    """Compute distances and relative vectors with the minimum image convention.

    Parameters
    ----------
    atpos
        Atomic positions, one atom per row.
        Array shape = (natom, 3).
    iatoms0
        Indexes of atoms where the relative vectors start.
    iatoms1
        Corresponding indexes of atoms where the relative vectors end.
    cell_lengths
        The lengths of a periodic orthorombic box.

    Returns
    -------
    deltas
        The relative vectors of the minimal distances.
    dists
        The corresponding lengths of the relative vectors.
    """
    # Construct the relative vectors
    atrel = atpos[iatoms1] - atpos[iatoms0]

    # Apply the minimum image convention
    frrel = atrel / cell_lengths
    atmic = (frrel - np.round(frrel)) * cell_lengths

    # Compute distances and filter
    atd = np.linalg.norm(atmic, axis=1)

    return atmic, atd


def build_nlist_linked_cell(
    atpos: ArrayLike, cell_lengths: ArrayLike, rmax: float
) -> NDArray[NLIST_DTYPE]:
    """Build a neighborlist with linked cell algorithm.

    Parameters
    ----------
    atpos
        Atomic positions, one atom per row.
        Array shape = (natom, 3).
    cell_lengths
        The lengths of a periodic orthorombic box.
    rmax
        The maximum radioius, i.e. the cut-off radius for the neighborlist.
        Note that the corresponding sphere must fit in the simulation cell.

    Returns
    -------
    neighborlist
        Structured array with neighborlist, including pairs up to distance rmax.
    """
    atpos = parse_atpos(atpos)
    cell_lengths = parse_cell_lengths(cell_lengths)
    rmax = parse_rmax(rmax, cell_lengths)

    # Group the atoms into bins
    bins, nbins = _assign_atoms_to_bins(atpos, cell_lengths, rmax)

    # Loop over pairs of nearby bins and collect parts for neighborlist.
    iatoms0_parts = []
    iatoms1_parts = []
    deltas_parts = []
    dists_parts = []
    for idx0, bin0 in bins.items():
        parts = [_create_parts_self(atpos, bin0, cell_lengths, rmax)]
        for idx1 in _iter_nearby(idx0, nbins):
            bin1 = bins.get(idx1)
            if bin1 is not None:
                parts.append(_create_parts_nearby(atpos, bin0, bin1, cell_lengths, rmax))
        for iatoms0, iatoms1, deltas, dists in parts:
            if len(dists) > 0:
                iatoms0_parts.append(iatoms0)
                iatoms1_parts.append(iatoms1)
                deltas_parts.append(deltas)
                dists_parts.append(dists)

    # Put everything in a neighborlist array.
    if len(dists_parts) == 0:
        return np.zeros(0, dtype=NLIST_DTYPE)
    dists = np.concatenate(dists_parts)
    nlist = np.zeros(len(dists), dtype=NLIST_DTYPE)
    nlist["iatom0"] = np.concatenate(iatoms0_parts)
    nlist["iatom1"] = np.concatenate(iatoms1_parts)
    nlist["delta"] = np.concatenate(deltas_parts)
    nlist["dist"] = dists
    return nlist


def _assign_atoms_to_bins(
    atpos: NDArray[float], cell_lengths: NDArray[float], rmax: float
) -> tuple[dict[tuple[int, int, int], NDArray[int]], NDArray[int]]:
    """Create arrays of atom indexes for each bin in the cell.

    Parameters
    ----------
    atpos
        Atomic positions, one atom per row.
        Array shape = (natom, 3).
    cell_lengths
        The lengths of a periodic orthorombic box.
    rmax
        The maximum radioius, i.e. the cut-off radius for the neighborlist.
        Note that the corresponding sphere must fit in the simulation cell.

    Returns
    -------
    bins
        A dictionary whose keys are 3-tuples of integer bin indexes and
        whose values are arrays of atom indexes in the corresponding bins.
    nbins
        An array with three values: the number of bins along each Cartesian axis.
    """
    nbins = np.floor(cell_lengths / rmax).astype(int)
    if (nbins < 2).any():
        raise ValueError("The cutoff radius is too large for the given cell lengths.")
    idxs = np.floor(atpos / (cell_lengths / nbins)).astype(int) % nbins
    flat_idxs = (idxs[:, 0] * nbins[1] + idxs[:, 1]) * nbins[2] + idxs[:, 2]
    _flat_unique, firsts, inverse = np.unique(flat_idxs, return_index=True, return_inverse=True)
    bins = {
        tuple(int(idx) for idx in idxs[first]): (inverse == i).nonzero()[0]
        for i, first in enumerate(firsts)
    }
    return bins, nbins


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
        iatoms0, iatoms1 = np.triu_indices(atpos.shape[0], 1)
    else:
        i0, i1 = np.triu_indices(len(bin0), 1)
        iatoms0 = bin0[i0]
        iatoms1 = bin0[i1]
    deltas, dists = _mic(atpos, iatoms0, iatoms1, cell_lengths)
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
    deltas, dists = _mic(atpos, iatoms0, iatoms1, cell_lengths)
    mask = dists <= rmax
    return iatoms0[mask], iatoms1[mask], deltas[mask], dists[mask]


def recompute_nlist(atpos: ArrayLike, cell_lengths: ArrayLike, nlist: NDArray[NLIST_DTYPE]):
    """Recompute deltas and distances and reset other parts of the neighborlist in-place.

    Parameters
    ----------
    atpos
        The array with all atomic positions. Shape is (natom, 3).
    cell_lengths
        The lengths of the periodic cell edges.
    nlist
        A previously built neighborlist, using older atomic positions.
        This array will be updated instead of returning a new one.
    """
    # Process parameters.
    atpos = parse_atpos(atpos)
    cell_lengths = parse_cell_lengths(cell_lengths)

    # Do some work.
    deltas, dists = _mic(atpos, nlist["iatom0"], nlist["iatom1"], cell_lengths)

    # Update or reset fields in the neigborlist.
    nlist["delta"] = deltas
    nlist["dist"] = dists
    nlist["gdelta"] = 0.0
    nlist["gdist"] = 0.0
    nlist["energy"] = 0.0

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
"""Utilities used by other modules."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ("parse_atpos", "parse_cell_lengths", "parse_rmax")


def parse_atpos(atpos: ArrayLike, natom: int | None = None) -> NDArray[float]:
    """Check and convert an atomic positions argument."""
    atpos = np.asarray(atpos, dtype=float)
    if atpos.ndim != 2:
        raise TypeError("atpos must be a 2D array.")
    if atpos.shape[1] != 3:
        raise TypeError("atpos must have three columns.")
    if natom is not None and atpos.shape[0] != natom:
        raise TypeError(f"atpos is expected to contain natom={natom} rows.")
    return atpos


def parse_cell_lengths(cell_lengths: ArrayLike) -> NDArray[float]:
    cell_lengths = np.asarray(cell_lengths, dtype=float)
    if cell_lengths.shape == ():
        cell_lengths = np.full(3, cell_lengths)
    elif cell_lengths.shape != (3,):
        raise TypeError("cell_lengths must have three elements.")
    if (cell_lengths <= 0).any():
        raise ValueError("All cell lenghths must be positive.")
    return cell_lengths


def parse_rmax(rmax: float, cell_lengths: NDArray[float]) -> float:
    rmax = float(rmax)
    if rmax <= 0:
        raise ValueError("The maximum radius must be strictly positive.")
    if 2 * rmax > cell_lengths.min():
        raise ValueError("Too large maximum radius for the minimum image convention.")
    return rmax

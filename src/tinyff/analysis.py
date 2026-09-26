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
"""Trajectory analysis functions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, next_fast_len, rfft

from .neighborlist import NBuild

__all__ = ("compute_acf", "compute_rdf")


def compute_rdf(
    traj_atpos: ArrayLike, cell_lengths: ArrayLike, spacing: float, nbuild: NBuild
) -> tuple[NDArray[float], NDArray[float]]:
    """Compute an RDF in post-processing, given a trajectory of atomic positions.

    Parameters
    ----------
    traj_atpos
        Configurational snapshots in a 3D array with shape `(nstep, natom, 3)`.
    cell_lengths
        The length of the edge of a cubic simulation cell,
        or an array of lengths of three cell vectors.
    spacing
        The width of the bins of the histogram.
    nbuild
        The neighborlist build algorithm for the computation of pairwise distances.
        Its internal state is overwritten, so do not pass the `nbuild` of a `ForceField`
        that is still in use.
        Because snapshots in a trajectory are typically far apart in time,
        it should be created with `nlist_reuse=0` (the default).

    Returns
    -------
    bin_mids
        The midpoints of the histogram bins used to count the number of pairs.
    rdf
        The radial distribution function at each bin midpoint.
    """
    traj_atpos = np.asarray(traj_atpos, dtype=float)
    bins = np.arange(int(np.floor(nbuild.rmax / spacing)) + 1) * spacing
    counts = 0
    for atpos in traj_atpos:
        cell_lengths = nbuild.update(atpos, cell_lengths)
        counts += np.histogram(nbuild.nlist["dist"], bins)[0]
    sphere_vols = (4 * np.pi / 3) * bins**3
    delta_vols = sphere_vols[1:] - sphere_vols[:-1]
    rho_pair = counts / (traj_atpos.shape[0] * delta_vols)
    natom = traj_atpos.shape[1]
    rho_pair0 = ((natom - 1) * natom) / (2 * np.prod(cell_lengths))
    bin_mids = (bins[1:] + bins[:-1]) / 2
    return bin_mids, rho_pair / rho_pair0


def compute_acf(traj_data: ArrayLike) -> NDArray[float]:
    """Compute the autocorrelation function of time-dependent data.

    Parameters
    ----------
    traj_data
        An array of which the first index corresponds to an equidistant time step.
        The autocorrelation function is averaged over all remaining indexes.

    Returns
    -------
    acf
        The autocorrelation function, as a function of time lag,
        at the same equidistant time steps of the input.
    """
    traj_data = np.asarray(traj_data, dtype=float)
    traj_data = traj_data.reshape((traj_data.shape[0], -1))
    nstep, ncol = traj_data.shape
    # Zero padding to at least 2 * nstep - 1 avoids wrap-around in the circular correlation.
    # The power spectra of all columns are summed before a single inverse transform.
    size = next_fast_len(2 * nstep - 1, real=True)
    spectra = rfft(traj_data, n=size, axis=0)
    power = (spectra.real**2 + spectra.imag**2).sum(axis=1)
    acf = irfft(power, n=size)[:nstep]
    acf /= ncol
    acf /= np.arange(nstep, 0, -1)
    return acf

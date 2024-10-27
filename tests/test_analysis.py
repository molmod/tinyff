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
"""Unit tests for tinyff.analysis."""

import numpy as np
import pytest

from tinyff.analysis import compute_acf, compute_rdf
from tinyff.neighborlist import NBuildSimple


def test_compute_rdf_simple():
    traj_atpos = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 4.0, 0.0]]])
    r, g = compute_rdf(traj_atpos, cell_lengths=12.1, spacing=0.7, nbuild=NBuildSimple(rmax=6.0))
    assert r == pytest.approx(np.arange(0.35, 5.5, 0.7))
    mask = np.zeros(r.shape, dtype=bool)
    mask[abs(r - 3.0).argmin()] = True
    mask[abs(r - 4.0).argmin()] = True
    mask[abs(r - 5.0).argmin()] = True
    assert (g[mask] != 0).all()
    assert (g[~mask] == 0).all()
    nz = g[mask]
    assert (nz[1:] < nz[:-1]).all()
    assert (abs((g * r**2)[mask] - 67) < 0.5).all()


def test_compute_vaf_simple():
    rng = np.random.default_rng(42)
    traj_atvel = rng.uniform(-1, 1, (100, 10, 3))
    acf = compute_acf(traj_atvel)
    assert acf.shape == (100,)

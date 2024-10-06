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
"""Unit tests for tinyff.atomsmithy."""

import numpy as np
import pytest

from tinyff.atomsmithy import (
    build_bcc_lattice,
    build_cubic_lattice,
    build_fcc_lattice,
    build_random_cell,
)
from tinyff.neighborlist import build_nlist_simple


def test_cubic_lattice():
    atpos0 = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.5],
        [0.0, 2.5, 0.0],
        [0.0, 2.5, 2.5],
        [2.5, 0.0, 0.0],
        [2.5, 0.0, 2.5],
        [2.5, 2.5, 0.0],
        [2.5, 2.5, 2.5],
    ]
    atpos1 = build_cubic_lattice(2.5, 2)
    assert atpos0 == pytest.approx(atpos1)


def test_bcc_lattice():
    atpos0 = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.5],
        [0.0, 2.5, 0.0],
        [0.0, 2.5, 2.5],
        [2.5, 0.0, 0.0],
        [2.5, 0.0, 2.5],
        [2.5, 2.5, 0.0],
        [2.5, 2.5, 2.5],
        [1.25, 1.25, 1.25],
        [1.25, 1.25, 3.75],
        [1.25, 3.75, 1.25],
        [1.25, 3.75, 3.75],
        [3.75, 1.25, 1.25],
        [3.75, 1.25, 3.75],
        [3.75, 3.75, 1.25],
        [3.75, 3.75, 3.75],
    ]
    atpos1 = build_bcc_lattice(2.5, 2)
    assert atpos0 == pytest.approx(atpos1)


def test_fcc_lattice():
    atpos0 = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.5],
        [0.0, 2.5, 0.0],
        [0.0, 2.5, 2.5],
        [2.5, 0.0, 0.0],
        [2.5, 0.0, 2.5],
        [2.5, 2.5, 0.0],
        [2.5, 2.5, 2.5],
        [0.0, 1.25, 1.25],
        [0.0, 1.25, 3.75],
        [0.0, 3.75, 1.25],
        [0.0, 3.75, 3.75],
        [2.5, 1.25, 1.25],
        [2.5, 1.25, 3.75],
        [2.5, 3.75, 1.25],
        [2.5, 3.75, 3.75],
        [1.25, 0.0, 1.25],
        [1.25, 0.0, 3.75],
        [1.25, 2.5, 1.25],
        [1.25, 2.5, 3.75],
        [3.75, 0.0, 1.25],
        [3.75, 0.0, 3.75],
        [3.75, 2.5, 1.25],
        [3.75, 2.5, 3.75],
        [1.25, 1.25, 0.0],
        [1.25, 1.25, 2.5],
        [1.25, 3.75, 0.0],
        [1.25, 3.75, 2.5],
        [3.75, 1.25, 0.0],
        [3.75, 1.25, 2.5],
        [3.75, 3.75, 0.0],
        [3.75, 3.75, 2.5],
    ]
    atpos1 = build_fcc_lattice(2.5, 2)
    assert atpos0 == pytest.approx(atpos1)


def test_random_box():
    rng = np.random.default_rng(42)
    atpos = build_random_cell(10.0, 32, rng)
    nlist = build_nlist_simple(atpos, [10.0, 10.0, 10.0], rcut=4.0)
    assert nlist["dist"].min() > 2.0

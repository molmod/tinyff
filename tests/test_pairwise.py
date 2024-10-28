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
"""Unit tests for tinyff.pairwise."""

import numdifftools as nd
import numpy as np
import pytest

from tinyff.pairwise import CheapRepulsion, CutOffWrapper, LennardJones


def test_lennard_jones_derivative():
    lj = LennardJones(2.5, 0.5)
    dist = np.linspace(0.4, 3.0, 50)
    gdist1 = lj.compute(dist, nderiv=1)[1]
    gdist2 = nd.Derivative(lambda x: lj.compute(x)[0])(dist)
    assert gdist1 == pytest.approx(gdist2)


def test_lennard_jones_cut_derivative():
    lj = CutOffWrapper(LennardJones(2.5, 0.5), 3.5)
    dist = np.linspace(0.4, 5.0, 50)
    gdist1 = lj.compute(dist, nderiv=1)[1]
    gdist2 = nd.Derivative(lambda x: lj.compute(x)[0])(dist)
    assert gdist1 == pytest.approx(gdist2)


def test_lennard_jones_cut_zero_array():
    lj = CutOffWrapper(LennardJones(2.5, 0.5), 3.5)
    e, g = lj.compute([5.0, 3.6], nderiv=1)
    assert (e == 0.0).all()
    assert (g == 0.0).all()


def test_lennard_jones_cut_zero_scalar():
    lj = CutOffWrapper(LennardJones(2.5, 0.5), 3.5)
    e, g = lj.compute(5.0, nderiv=1)
    assert e == 0.0
    assert g == 0.0


def test_cheap_repulsion_derivative():
    cr = CheapRepulsion(2.5)
    dist = np.linspace(0.4, 3.0, 50)
    gdist1 = cr.compute(dist, nderiv=1)[1]
    gdist2 = nd.Derivative(lambda x: cr.compute(x)[0])(dist)
    assert gdist1 == pytest.approx(gdist2)


def test_cheap_repulsion_cutoff():
    rcut = 2.5
    cr = CheapRepulsion(rcut)
    eps = 1e-13
    e, g = cr.compute(rcut - 0.1, nderiv=1)
    assert abs(e) > eps
    assert abs(g) > eps
    e, g = cr.compute(rcut - eps, nderiv=1)
    assert abs(e) < eps
    assert abs(g) < eps
    e, g = cr.compute(rcut + 0.2, nderiv=1)
    assert e == 0.0
    assert g == 0.0

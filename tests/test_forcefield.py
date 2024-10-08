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
"""Unit tests for tinyff.forcefield."""

import numdifftools as nd
import numpy as np
import pytest

from tinyff.forcefield import CutOffWrapper, LennardJones, PairwiseForceField
from tinyff.neighborlist import build_nlist_linked_cell, build_nlist_simple


def test_lennard_jones_derivative():
    lj = LennardJones(2.5, 0.5)
    dist = np.linspace(0.4, 3.0, 50)
    gdist1 = lj(dist)[1]
    gdist2 = nd.Derivative(lambda dist: lj(dist)[0])(dist)
    assert gdist1 == pytest.approx(gdist2)


def test_lennard_jones_cut_derivative():
    lj = CutOffWrapper(LennardJones(2.5, 0.5), 3.5)
    dist = np.linspace(0.4, 5.0, 50)
    gdist1 = lj(dist)[1]
    gdist2 = nd.Derivative(lambda x: lj(x)[0])(dist)
    assert gdist1 == pytest.approx(gdist2)


def test_lennard_jones_cut_zero_array():
    lj = CutOffWrapper(LennardJones(2.5, 0.5), 3.5)
    e, g = lj([5.0, 3.6])
    assert (e == 0.0).all()
    assert (g == 0.0).all()


def test_lennard_jones_cut_zero_scalar():
    lj = CutOffWrapper(LennardJones(2.5, 0.5), 3.5)
    e, g = lj(5.0)
    assert e == 0.0
    assert g == 0.0


@pytest.mark.parametrize("build_nlist", [build_nlist_linked_cell, build_nlist_simple])
def test_pairwise_force_field_two(build_nlist):
    # Build a simple model for testing.
    cell_length = 20.0
    atpos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # Define the force field.
    rcut = 8.0
    lj = CutOffWrapper(LennardJones(2.5, 1.3), rcut)
    pwff = PairwiseForceField(lj, rcut, build_nlist=build_nlist)

    # Compute and check against manual result
    energy, forces, frc_press = pwff(atpos, cell_length)
    d = np.linalg.norm(atpos[0] - atpos[1])
    e, g = lj(d)
    assert energy == pytest.approx(e)
    assert forces == pytest.approx(np.array([[g, 0.0, 0.0], [-g, 0.0, 0.0]]))
    assert frc_press == pytest.approx(-g * d / (3 * cell_length**3))


@pytest.mark.parametrize("build_nlist", [build_nlist_linked_cell, build_nlist_simple])
def test_pairwise_force_field_three(build_nlist):
    # Build a simple model for testing.
    cell_length = 20.0
    atpos = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 2.5], [0.0, 5.0, -2.5]])

    # Define the force field.
    rcut = 8.0
    lj = CutOffWrapper(LennardJones(2.5, 1.3), rcut)
    pwff = PairwiseForceField(lj, rcut, build_nlist=build_nlist)

    # Compute the energy, the forces and the force contribution pressure.
    energy1, forces1, frc_press1 = pwff(atpos, cell_length)

    # Compute the energy manually and compare.
    dists = [
        np.linalg.norm(atpos[1] - atpos[2]),
        np.linalg.norm(atpos[2] - atpos[0]),
        np.linalg.norm(atpos[0] - atpos[1]),
    ]
    energy2 = lj(dists)[0].sum()
    assert energy1 == pytest.approx(energy2)

    # Test forces with numdifftool
    forces2 = -nd.Gradient(lambda x: pwff(x.reshape(-1, 3), cell_length)[0])(atpos)
    forces2.shape = (-1, 3)
    assert forces1 == pytest.approx(forces2.reshape(-1, 3))

    # Test pressure with numdifftool
    def energy_volume(volume):
        my_cell_length = volume ** (1.0 / 3.0)
        scale = my_cell_length / cell_length
        return pwff(atpos * scale, my_cell_length)[0]

    frc_press2 = -nd.Derivative(energy_volume)(cell_length**3)
    assert frc_press1 == pytest.approx(frc_press2)


@pytest.mark.parametrize("build_nlist", [build_nlist_linked_cell, build_nlist_simple])
def test_pairwise_force_field_fifteen(build_nlist):
    # Build a simple model for testing.
    cell_length = 20.0
    atpos = np.array(
        [
            [1.44312518, 19.04105338, 2.40917937],
            [18.56638373, 19.36876523, 1.04082339],
            [15.4648885, 2.89452394, 5.66329753],
            [12.11611309, 19.001517, 17.19418478],
            [6.80418287, 5.65586971, 8.53724665],
            [8.07614612, 17.85301782, 5.96970034],
            [6.08426762, 1.85381157, 8.09270812],
            [9.39155079, 10.29526351, 5.03853033],
            [0.60874926, 4.51273075, 18.02934992],
            [15.41680528, 9.36911558, 18.84660097],
            [14.42910733, 2.2588027, 4.59601648],
            [18.32769468, 10.55508761, 18.54896363],
            [2.64336372, 10.03756966, 9.6377395],
            [14.01553155, 15.43656781, 15.99678273],
            [3.69078799, 16.8481288, 0.78705498],
        ]
    )

    # Define the force field.
    rcut = 8.0
    lj = CutOffWrapper(LennardJones(2.5, 1.3), rcut)
    pwff = PairwiseForceField(lj, rcut, build_nlist=build_nlist)

    # Compute the energy, the forces and the force contribution to the pressure.
    energy, forces1, frc_press1 = pwff(atpos, cell_length)
    assert energy < 0

    # Test forces with numdifftool
    forces2 = -nd.Gradient(lambda x: pwff(x.reshape(-1, 3), cell_length)[0])(atpos)
    forces2.shape = (-1, 3)
    assert forces1 == pytest.approx(forces2.reshape(-1, 3))

    # Test pressure with numdifftool
    def energy_volume(volume):
        my_cell_length = volume ** (1.0 / 3.0)
        scale = my_cell_length / cell_length
        return pwff(atpos * scale, my_cell_length)[0]

    frc_press2 = -nd.Derivative(energy_volume)(cell_length**3)
    assert frc_press1 == pytest.approx(frc_press2)


@pytest.mark.parametrize("build_nlist", [build_nlist_linked_cell, build_nlist_simple])
def test_nlist_reuse(build_nlist):
    # Build a simple model for testing.
    cell_length = 20.0
    atpos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # Define the force field.
    rcut = 8.0
    lj = CutOffWrapper(LennardJones(2.5, 1.3), rcut)
    pwff = PairwiseForceField(lj, rmax=9.0, build_nlist=build_nlist, nlist_reuse=3)
    pwff(atpos, cell_length)
    assert len(pwff.nlist) == 1
    assert pwff.nlist_use_count == 3
    assert pwff.nlist["dist"][0] == pytest.approx(2.0)
    atpos = np.array([[0.0, 0.0, 0.0], [8.5, 0.0, 0.0]])
    pwff(atpos, cell_length)
    assert len(pwff.nlist) == 1
    assert pwff.nlist_use_count == 2
    assert pwff.nlist["dist"][0] == pytest.approx(8.5)
    atpos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    pwff(atpos, cell_length)
    assert pwff.nlist_use_count == 1
    assert len(pwff.nlist) == 1
    assert pwff.nlist["dist"][0] == pytest.approx(10)
    pwff(atpos, cell_length)
    assert pwff.nlist_use_count == 3
    assert len(pwff.nlist) == 0

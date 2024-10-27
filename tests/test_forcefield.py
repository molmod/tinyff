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

from tinyff.forcefield import ForceField
from tinyff.neighborlist import NBuildCellLists, NBuildSimple
from tinyff.pairwise import CheapRepulsion, CutOffWrapper, LennardJones


@pytest.mark.parametrize("nbuild", [NBuildSimple(8.0), NBuildCellLists(8.0, nbin_approx=8)])
def test_pairwise_force_field_two(nbuild):
    # Build a simple model for testing.
    cell_length = 20.0
    atpos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # Define the force field.
    rcut = 8.0
    lj = CutOffWrapper(LennardJones(2.5, 1.3), rcut)
    ff = ForceField([lj], nbuild=nbuild)

    # Compute and check against manual result
    energy, forces, frc_press = ff.compute(atpos, cell_length, do_forces=True, do_press=True)
    d = np.linalg.norm(atpos[0] - atpos[1])
    e, g = lj.compute(d, do_gdist=True)
    assert energy == pytest.approx(e)
    assert forces == pytest.approx(np.array([[g, 0.0, 0.0], [-g, 0.0, 0.0]]))
    assert frc_press == pytest.approx(-g * d / (3 * cell_length**3))


@pytest.mark.parametrize("nbuild", [NBuildSimple(8.0), NBuildCellLists(8.0, nbin_approx=8)])
def test_pairwise_force_field_three(nbuild):
    # Build a simple model for testing.
    cell_length = 20.0
    atpos = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 2.5], [0.0, 5.0, -2.5]])

    # Define the force field.
    rcut = 8.0
    lj = CutOffWrapper(LennardJones(2.5, 1.3), rcut)
    ff = ForceField([lj], nbuild=nbuild)

    # Compute the energy, the forces and the force contribution pressure.
    energy1, forces1, frc_press1 = ff.compute(atpos, cell_length, do_forces=True, do_press=True)

    # Compute the energy manually and compare.
    dists = [
        np.linalg.norm(atpos[1] - atpos[2]),
        np.linalg.norm(atpos[2] - atpos[0]),
        np.linalg.norm(atpos[0] - atpos[1]),
    ]
    energy2 = lj.compute(dists)[0].sum()
    assert energy1 == pytest.approx(energy2)

    # Test forces with numdifftool
    forces2 = -nd.Gradient(lambda x: ff.compute(x.reshape(-1, 3), cell_length)[0])(atpos)
    forces2.shape = (-1, 3)
    assert forces1 == pytest.approx(forces2.reshape(-1, 3))

    # Test pressure with numdifftool
    def energy_volume(volume):
        my_cell_length = volume ** (1.0 / 3.0)
        scale = my_cell_length / cell_length
        return ff.compute(atpos * scale, my_cell_length)[0]

    frc_press2 = -nd.Derivative(energy_volume)(cell_length**3)
    assert frc_press1 == pytest.approx(frc_press2)


@pytest.mark.parametrize("nbuild", [NBuildSimple(8.0), NBuildCellLists(8.0, nbin_approx=8)])
def test_pairwise_force_field_fifteen(nbuild):
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
    ff = ForceField([lj], nbuild=nbuild)

    # Compute the energy, the forces and the force contribution to the pressure.
    energy, forces1, frc_press1 = ff.compute(atpos, cell_length, do_forces=True, do_press=True)
    assert energy < 0

    # Test forces with numdifftool
    forces2 = -nd.Gradient(lambda x: ff.compute(x.reshape(-1, 3), cell_length)[0])(atpos)
    forces2.shape = (-1, 3)
    assert forces1 == pytest.approx(forces2.reshape(-1, 3))

    # Test pressure with numdifftool
    def energy_volume(volume):
        my_cell_length = volume ** (1.0 / 3.0)
        scale = my_cell_length / cell_length
        return ff.compute(atpos * scale, my_cell_length)[0]

    frc_press2 = -nd.Derivative(energy_volume)(cell_length**3)
    assert frc_press1 == pytest.approx(frc_press2)


@pytest.mark.parametrize(
    ("nbuild_class", "kwargs"), [(NBuildSimple, {}), (NBuildCellLists, {"nbin_approx": 8})]
)
def test_superposition(nbuild_class, kwargs):
    atpos = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    cell_length = 10.0

    # Define the force field.
    rcut = 4.9
    lj = CutOffWrapper(LennardJones(2.5, 1.3), rcut)
    cr = CheapRepulsion(rcut)
    ff_lj = ForceField([lj], nbuild=nbuild_class(rcut, **kwargs))
    ff_pp = ForceField([cr], nbuild=nbuild_class(rcut, **kwargs))
    ff_su = ForceField([lj, cr], nbuild=nbuild_class(rcut, **kwargs))

    ener_lj, forces_lj, press_lj = ff_lj.compute(atpos, cell_length, do_forces=True, do_press=True)
    ener_pp, forces_pp, press_pp = ff_pp.compute(atpos, cell_length, do_forces=True, do_press=True)
    ener_su, forces_su, press_su = ff_su.compute(atpos, cell_length, do_forces=True, do_press=True)
    assert ener_lj + ener_pp == pytest.approx(ener_su)
    assert forces_lj + forces_pp == pytest.approx(forces_su)
    assert press_lj + press_pp == pytest.approx(press_su)


@pytest.mark.parametrize("nbuild", [NBuildSimple(5.0), NBuildCellLists(5.0, nbin_approx=8)])
def test_try_accept_move_simple(nbuild):
    atpos0 = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    delta = np.array([1.0, 0.4, -0.3])
    atpos1 = atpos0.copy()
    iatom = 1
    atpos1[iatom] += delta
    cell_length = 10.0
    rcut = 4.9

    ff = ForceField([CheapRepulsion(rcut)], nbuild=nbuild)
    (energy1,) = ff.compute(atpos1, cell_length)
    (energy0,) = ff.compute(atpos0, cell_length)
    assert (ff.nbuild.nlist["gdelta"] == 0).all()

    energy_change, move = ff.try_move(1, delta, cell_length)
    # Check basics
    assert energy_change == pytest.approx(energy1 - energy0)
    assert (move.nlist["iatom0"] == ff.nbuild.nlist["iatom0"][move.select]).all()
    assert (move.nlist["iatom1"] == ff.nbuild.nlist["iatom1"][move.select]).all()
    assert move.nlist["dist"] != pytest.approx(ff.nbuild.nlist["dist"][move.select])
    assert move.nlist["delta"] != pytest.approx(ff.nbuild.nlist["delta"][move.select])
    assert move.nlist["energy"] != pytest.approx(ff.nbuild.nlist["energy"][move.select])
    assert (move.nlist["energy"] != 0).any()
    assert (move.nlist["gdelta"] == 0).all()
    # Check details
    for im, il in enumerate(move.select):
        if move.nlist["iatom0"][im] == iatom:
            assert move.nlist["delta"][im] == pytest.approx(ff.nbuild.nlist["delta"][il] - delta)
        elif move.nlist["iatom1"][im] == iatom:
            assert move.nlist["delta"][im] == pytest.approx(ff.nbuild.nlist["delta"][il] + delta)
        else:
            raise AssertionError("Unrelated nlist row in move.")

    ff.accept_move(move)
    assert move.nlist["dist"] == pytest.approx(ff.nbuild.nlist["dist"][move.select])
    assert move.nlist["delta"] == pytest.approx(ff.nbuild.nlist["delta"][move.select])
    assert move.nlist["energy"] == pytest.approx(ff.nbuild.nlist["energy"][move.select])
    assert (move.nlist["gdelta"] == 0).all()


@pytest.mark.parametrize("nbuild", [NBuildSimple(4.9), NBuildCellLists(4.9, nbin_approx=8)])
def test_try_accept_move_random(nbuild):
    natom = 50
    cell_length = 10.0
    rcut = 3.2
    ff = ForceField([CheapRepulsion(rcut)], nbuild=nbuild)
    rng = np.random.default_rng(42)
    atpos0 = rng.uniform(-cell_length, 2 * cell_length, (natom, 3))
    for _ in range(100):
        delta = rng.uniform(-0.1, 0.1, 3)
        iatom = int(rng.integers(natom))
        atpos1 = atpos0.copy()
        atpos1[iatom] += delta

        (energy1,) = ff.compute(atpos1, cell_length)
        (energy0,) = ff.compute(atpos0, cell_length)
        energy_change, move = ff.try_move(iatom, delta, cell_length)
        assert abs((move.nlist["delta"] <= cell_length / 2).all())
        assert energy_change == pytest.approx(energy1 - energy0)

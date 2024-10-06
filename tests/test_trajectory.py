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
"""Unit tests for tinyff.trajectory."""

import os

import numpy as np
import pytest

from tinyff.trajectory import NPYWriter, PDBWriter

PDB_REF_DATA = """\
CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
HETATM    1 Ar   ATM     1       0.000  20.000   5.000  1.00  1.00          Ar
HETATM    2 Ar   ATM     1      10.000  33.000  24.400  1.00  1.00          Ar
HETATM    3 Ar   ATM     1     -10.000  28.000  17.000  1.00  1.00          Ar
END
CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
HETATM    1 Ar   ATM     1       0.100  20.300   0.000  1.00  1.00          Ar
HETATM    2 Ar   ATM     1      12.000  30.500  32.100  1.00  1.00          Ar
HETATM    3 Ar   ATM     1     -10.300  35.800  20.300  1.00  1.00          Ar
END
"""


def test_pdb_writer(tmpdir):
    cell_length = 3.0
    atpos0 = np.array([[0.0, 2.0, 0.5], [1.0, 3.3, 2.44], [-1.0, 2.8, 1.7]])
    atpos1 = atpos0 + np.array([[0.01, 0.03, -0.5], [0.2, -0.25, 0.77], [-0.03, 0.78, 0.33]])

    # Parameters: (i) distance are in nanometers and (ii) 3 argon atoms
    path_pdb = os.path.join(tmpdir, "test.pdb")
    pdb_writer = PDBWriter(path_pdb, to_angstrom=10.0, atnums=[18, 18, 18])
    pdb_writer.dump(atpos0, cell_length)
    pdb_writer.dump(atpos1, cell_length)

    # Load PDB file to compare with reference result
    with open(path_pdb) as fh:
        data = fh.read()
    assert data == PDB_REF_DATA

    # Start from scratch, should not append.
    del pdb_writer
    pdb_writer = PDBWriter(path_pdb, to_angstrom=10.0, atnums=[18, 18, 18])
    pdb_writer.dump(atpos0, cell_length)
    pdb_writer.dump(atpos1, cell_length)

    # Load PDB file to compare with reference result
    with open(path_pdb) as fh:
        print(data)
    assert data == PDB_REF_DATA


def test_npy_traj(tmpdir):
    traj_atpos = np.array([[[1.2, 1.3], [4.9, 3.1]], [[0.7, 8.1], [-7.9, 0.5]]])
    traj_pressure = np.array([5.0, 4.0])

    for _ in range(2):
        # Write
        npy_writer = NPYWriter(os.path.join(tmpdir, "out"))
        for i in 0, 1:
            npy_writer.dump(pressure=traj_pressure[i], atpos=traj_atpos[i])

        # Load and check
        assert np.load(os.path.join(tmpdir, "out/pressure.npy")) == pytest.approx(traj_pressure)
        assert np.load(os.path.join(tmpdir, "out/atpos.npy")) == pytest.approx(traj_atpos)

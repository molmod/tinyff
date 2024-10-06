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
"""Unit tests for tinyff.neighborlist."""

import numpy as np
import pytest

from tinyff.neighborlist import (
    _assign_atoms_to_bins,
    _create_parts_nearby,
    _create_parts_self,
    _iter_nearby,
    _mic,
    build_nlist_linked_cell,
    build_nlist_simple,
    recompute_nlist,
)


def test_mic():
    atpos = np.array([[0.1, 0.1, 0.2], [1.9, 1.9, 2.0]])
    cell_lengths = np.array([2.0, 2.0, 2.0])
    deltas, dists = _mic(atpos, [0], [1], cell_lengths)
    assert deltas == pytest.approx(np.array([[-0.2, -0.2, -0.2]]))
    assert dists == pytest.approx([np.sqrt(12) / 10])


def test_mic_random():
    rng = np.random.default_rng(42)
    natom = 100
    npair = 200
    atpos = rng.uniform(-50, 50, (natom, 3))
    iatoms0 = rng.integers(natom, size=npair)
    iatoms1 = rng.integers(natom, size=npair)
    cell_lengths = np.array([5.0, 10.0, 20.0])
    deltas, dists = _mic(atpos, iatoms0, iatoms1, cell_lengths)
    assert deltas.shape == (npair, 3)
    assert (abs(deltas) <= cell_lengths / 2).all()
    assert (dists <= np.linalg.norm(cell_lengths) / 2).all()


def test_assign_atoms_to_bins_simple():
    atpos = np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.4], [1.5, 0.1, 4.1]])
    cell_lengths = np.array([2.0, 2.0, 2.0])
    rcut = 0.95
    bins, nbins = _assign_atoms_to_bins(atpos, cell_lengths, rcut)
    assert nbins.shape == (3,)
    assert nbins.dtype == int
    assert (nbins == 2).all()
    assert len(bins) == 2
    for idx in bins:
        assert len(idx) == 3
        assert isinstance(idx[0], int)
        assert isinstance(idx[1], int)
        assert isinstance(idx[2], int)
    assert (bins[(0, 0, 0)] == [0, 1]).all()
    assert (bins[(1, 0, 0)] == [2]).all()


def test_assign_atoms_to_bins_random():
    rng = np.random.default_rng(42)
    natom = 500
    rcut = 0.99
    atpos = rng.uniform(-50, 50, (natom, 3))
    cell_lengths = np.array([5.0, 3.0, 2.0])
    bins, nbins = _assign_atoms_to_bins(atpos, cell_lengths, rcut)
    assert nbins.shape == (3,)
    assert nbins.dtype == int
    assert (nbins == [5, 3, 2]).all()
    for idx, atoms in bins.items():
        assert len(idx) == 3
        assert isinstance(idx[0], int)
        assert isinstance(idx[1], int)
        assert isinstance(idx[2], int)
        assert idx[0] >= 0
        assert idx[0] < 5
        assert idx[1] >= 0
        assert idx[1] < 3
        assert idx[2] >= 0
        assert idx[2] < 2
        assert (atoms[1:] > atoms[:-1]).all()
    assert len(bins) == 5 * 3 * 2
    assert sum(len(bin0) for bin0 in bins.values()) == natom


@pytest.mark.parametrize("with_bin", [True, False])
def test_create_parts_self_simple(with_bin):
    atpos = (
        np.array([[0.0, -1.0, -4.0], [0.0, 0.0, 1.0], [0.0, 1.0, 4.0], [2.0, 5.0, 7.0]])
        if with_bin
        else np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 4.0]])
    )
    cell_lengths = np.array([15.0, 15.0, 15.0])
    rcut = 4.5
    bin0 = np.array([1, 2]) if with_bin else None
    iatoms0, iatoms1, deltas, dists = _create_parts_self(atpos, bin0, cell_lengths, rcut)
    if not with_bin:
        bin0 = [0, 1]
    assert iatoms0.shape == (1,)
    assert iatoms0.dtype == int
    assert iatoms0[0] == bin0[0]
    assert iatoms1.shape == (1,)
    assert iatoms1.dtype == int
    assert iatoms1[0] == bin0[1]
    assert deltas.shape == (1, 3)
    assert deltas[0] == pytest.approx([0.0, 1.0, 3.0])
    assert dists.shape == (1,)
    assert dists[0] == pytest.approx(np.sqrt(10))


@pytest.mark.parametrize("with_bin", [True, False])
def test_create_parts_self_random(with_bin):
    rng = np.random.default_rng(42)
    natom = 200
    atpos = rng.uniform(-50, 50, (natom, 3))
    cell_lengths = np.array([10.0, 15.0, 20.0])
    rcut = 4.5
    if with_bin:
        bin0 = rng.choice(natom, natom // 2, replace=False)
        bin0.sort()
        assert len(np.unique(bin0)) == natom // 2
    else:
        bin0 = None
    iatoms0_a, iatoms1_a, deltas_a, dists_a = _create_parts_self(atpos, bin0, cell_lengths, rcut)
    if with_bin:
        assert all(iatom0 in bin0 for iatom0 in iatoms0_a)
        assert all(iatom1 in bin0 for iatom1 in iatoms1_a)
    npair = len(dists_a)
    assert iatoms0_a.shape == (npair,)
    assert iatoms0_a.dtype == int
    assert iatoms1_a.shape == (npair,)
    assert iatoms1_a.dtype == int
    assert (iatoms0_a < iatoms1_a).all()
    assert len(np.unique(iatoms0_a + natom * iatoms1_a)) == npair
    assert deltas_a.shape == (npair, 3)
    assert (abs(deltas_a) < cell_lengths / 2).all()
    assert dists_a.shape == (npair,)
    assert (dists_a < np.linalg.norm(cell_lengths) / 2).all()

    # Displace all atoms by an integer linear combination of cell vectors.
    # This should not have any influence on the result.
    atpos += rng.integers(-5, 5, size=(natom, 3)) * cell_lengths
    iatoms0_b, iatoms1_b, deltas_b, dists_b = _create_parts_self(atpos, bin0, cell_lengths, rcut)
    assert (iatoms0_a == iatoms0_b).all()
    assert (iatoms1_a == iatoms1_b).all()
    assert deltas_a == pytest.approx(deltas_b)
    assert dists_a == pytest.approx(dists_b)


def test_iter_nearby_simple():
    nbins = np.array([3, 4, 5])
    nearby = list(_iter_nearby((0, 1, 2), nbins))
    assert len(nearby) == len(set(nearby))
    assert nearby == [
        (2, 0, 1),
        (2, 1, 1),
        (2, 2, 1),
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 0, 2),
        (0, 0, 2),
        (1, 0, 2),
        (2, 1, 2),
    ]


def test_iter_nearby_all():
    nbins = np.array([3, 4, 5])
    for n0 in range(nbins[0]):
        for n1 in range(nbins[1]):
            for n2 in range(nbins[2]):
                nearby = list(_iter_nearby((n0, n1, n2), nbins))
                assert len(nearby) == len(set(nearby))
                for o0, o1, o2 in nearby:
                    assert o0 >= 0
                    assert o0 < nbins[0]
                    assert o1 >= 0
                    assert o1 < nbins[1]
                    assert o2 >= 0
                    assert o2 < nbins[2]


def test_create_parts_nearby_simple():
    # Prepare example.
    atpos = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 2.5], [1.1, 1.2, -0.1], [1.9, 1.7, 1.8]])
    bin0 = np.array([0, 1])
    bin1 = np.array([2, 3])
    cell_lengths = np.array([2.0, 2.0, 2.0])
    rcut = 0.999
    iatoms0, iatoms1, deltas, dists = _create_parts_nearby(atpos, bin0, bin1, cell_lengths, rcut)

    # Two combinations are not expected to be present.
    all_dists = _mic(atpos, [0, 0, 1, 1], [2, 3, 2, 3], cell_lengths)[1]
    assert all_dists[0] >= rcut
    assert all_dists[3] >= rcut

    # Check the other two.
    assert iatoms0.shape == (2,)
    assert iatoms0.dtype == int
    assert (iatoms0 == [0, 1]).all()
    assert iatoms1.shape == (2,)
    assert iatoms1.dtype == int
    assert (iatoms1 == [3, 2]).all()
    assert deltas.shape == (2, 3)
    assert dists.shape == (2,)
    assert deltas[0] == pytest.approx([-0.2, -0.4, -0.3])
    assert dists[0] == pytest.approx(np.sqrt(29) / 10)
    assert deltas[1] == pytest.approx([0.2, 0.3, -0.6])
    assert dists[1] == pytest.approx(0.7)


def test_create_parts_nearby_random():
    rng = np.random.default_rng(42)
    natom = 100
    atpos = np.concatenate(
        [
            rng.uniform(0, 1, (natom // 2, 3)),
            rng.uniform(0, 1, (natom // 2, 3)) + [1, 0, 0],  # noqa: RUF005
        ]
    )
    cell_lengths = np.array([3.0, 2.0, 5.0])
    rcut = 0.999
    bin0 = np.arange(natom // 2)
    bin1 = bin0 + natom // 2

    # Run with original input
    iatoms0_a, iatoms1_a, deltas_a, dists_a = _create_parts_nearby(
        atpos, bin0, bin1, cell_lengths, rcut
    )
    npair = len(dists_a)
    assert npair < ((natom // 2) * (natom // 2 - 1)) // 2
    assert iatoms0_a.shape == (npair,)
    assert iatoms1_a.shape == (npair,)
    assert deltas_a.shape == (npair, 3)
    assert dists_a.shape == (npair,)
    assert (dists_a < rcut).all()

    # Displace all atoms by an integer linear combination of cell vectors.
    # This should not have any influence on the result.
    atpos += rng.integers(-5, 5, size=(natom, 3)) * cell_lengths
    iatoms0_b, iatoms1_b, deltas_b, dists_b = _create_parts_nearby(
        atpos, bin0, bin1, cell_lengths, rcut
    )
    assert (iatoms0_a == iatoms0_b).all()
    assert (iatoms1_a == iatoms1_b).all()
    assert deltas_a == pytest.approx(deltas_b)
    assert dists_a == pytest.approx(dists_b)


@pytest.mark.parametrize("build_nlist", [build_nlist_simple, build_nlist_linked_cell])
@pytest.mark.parametrize("cell_length", [1.0, 2.0, 3.0])
def test_build_cubic_simple(build_nlist, cell_length):
    # Build
    atpos = np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])
    atpos[1] += cell_length
    cell_lenghts = [cell_length] * 3
    nlist = build_nlist(atpos, cell_lenghts, 0.4)
    assert len(nlist) == 1
    i, j, delta, _, dist, _, _ = nlist[0]
    if i == 0:
        assert j == 1
        assert delta == pytest.approx([-0.2, -0.2, -0.2])
    else:
        assert i == 1
        assert j == 0
        assert delta == pytest.approx([0.2, 0.2, 0.2])
    assert dist == pytest.approx(np.sqrt(12) / 10)

    # Recompute
    atpos[1] = [0.5, 0.5, 0.5]
    recompute_nlist(atpos, cell_lenghts, nlist)
    assert len(nlist) == 1
    i, j, delta, _, dist, _, _ = nlist[0]
    if i == 0:
        assert j == 1
        assert delta == pytest.approx([0.4, 0.4, 0.4])
    else:
        assert i == 1
        assert j == 0
        assert delta == pytest.approx([-0.4, -0.4, -0.4])
    assert dist == pytest.approx(np.sqrt(48) / 10)


@pytest.mark.parametrize("build_nlist", [build_nlist_simple, build_nlist_linked_cell])
def test_build_empty(build_nlist):
    atpos = np.array([[0.1, 0.1, 0.1], [2.1, 2.1, 2.1]])
    cell_lenghts = 5.0
    nlist = build_nlist(atpos, cell_lenghts, 0.4)
    assert len(nlist) == 0


@pytest.mark.parametrize(
    "cell_lengths", [[10.0, 15.0, 20.0], [15.0, 20.0, 10.0], [20.0, 10.0, 15.0]]
)
def test_build_ortho_random(cell_lengths):
    rcut = 4.999
    rng = np.random.default_rng(42)
    natom = 100
    atpos = rng.uniform(-50.0, 50.0, (natom, 3))

    # Compute with simple algorithm and with linked cell
    nlist1 = build_nlist_simple(atpos, cell_lengths, rcut)
    nlist2 = build_nlist_linked_cell(atpos, cell_lengths, rcut)

    # Compare the results
    assert len(nlist1) == len(nlist2)

    def normalize(nlist):
        """Normalize neigbor lists to enable one-on-one comparison."""
        iatoms0 = nlist["iatom0"].copy()
        iatoms1 = nlist["iatom1"].copy()
        swap = iatoms0 < iatoms1
        nlist["iatom0"][swap] = iatoms1[swap]
        nlist["iatom1"][swap] = iatoms0[swap]
        nlist["delta"][swap] *= -1
        order = np.lexsort([nlist["iatom1"], nlist["iatom0"]])
        return nlist[order]

    # Sort both neighbor lists
    nlist1 = normalize(nlist1)
    nlist2 = normalize(nlist2)

    # Compare each field separately for more readable test outputs
    assert (nlist1["iatom0"] == nlist2["iatom0"]).all()
    assert (nlist1["iatom1"] == nlist2["iatom1"]).all()

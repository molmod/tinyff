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
"""Pairwise interaction terms for use in force fields."""

import attrs
import numpy as np
from numpy.typing import NDArray

from .neighborlist import NLIST_DTYPE

__all__ = ("PairwiseTerm", "LennardJones", "CutOffWrapper", "CheapRepulsion")


@attrs.define
class PairwiseTerm:
    def compute_nlist(
        self, nlist: NDArray[NLIST_DTYPE], do_energy: bool = True, do_gdist: bool = False
    ):
        """Compute energies and derivatives and add them to the neighborlist."""
        results = self.compute(nlist["dist"], do_energy, do_gdist)
        if do_energy:
            nlist["energy"] += results.pop(0)
        if do_gdist:
            nlist["gdist"] += results.pop(0)

    def compute(
        self,
        dist: NDArray[float],
        do_energy: bool = True,
        do_gdist: bool = False,
    ) -> list[NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        raise NotImplementedError  # pragma: nocover


@attrs.define
class LennardJones(PairwiseTerm):
    epsilon: float = attrs.field(default=1.0, converter=float)
    sigma: float = attrs.field(default=1.0, converter=float)

    def compute(
        self,
        dist: NDArray[float],
        do_energy: bool = True,
        do_gdist: bool = False,
    ) -> list[NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        results = []
        dist = np.asarray(dist, dtype=float)
        x = self.sigma / dist
        x3 = x * x * x
        x6 = x3 * x3
        if do_energy:
            energy = (x6 - 1) * x6
            energy *= 4 * self.epsilon
            results.append(energy)
        if do_gdist:
            gdist = (x6 - 0.5) * x6 / dist
            gdist *= -48 * self.epsilon
            results.append(gdist)
        return results


@attrs.define
class CutOffWrapper(PairwiseTerm):
    original: PairwiseTerm = attrs.field()
    rcut: float = attrs.field(converter=float)
    ecut: float = attrs.field(init=False, default=0.0, converter=float)
    gcut: float = attrs.field(init=False, default=0.0, converter=float)

    def __attrs_post_init__(self):
        """Post initialization changes."""
        self.ecut, self.gcut = self.original.compute(self.rcut, do_gdist=True)

    def compute(
        self,
        dist: NDArray[float],
        do_energy: bool = True,
        do_gdist: bool = False,
    ) -> list[NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        dist = np.asarray(dist, dtype=float)
        mask = dist < self.rcut
        results = []
        if mask.ndim == 0:
            # Deal with non-array case
            if mask:
                orig_results = self.original.compute(dist, do_energy, do_gdist)
                if do_energy:
                    energy = orig_results.pop(0)
                    energy -= self.ecut + self.gcut * (dist - self.rcut)
                    results.append(energy)
                if do_gdist:
                    gdist = orig_results.pop(0)
                    gdist -= self.gcut
                    results.append(gdist)
            else:
                if do_energy:
                    results.append(0.0)
                if do_gdist:
                    results.append(0.0)
        else:
            orig_results = self.original.compute(dist, do_energy, do_gdist)
            if do_energy:
                energy = orig_results.pop(0)
                energy -= self.ecut + self.gcut * (dist - self.rcut)
                energy *= mask
                results.append(energy)
            if do_gdist:
                gdist = orig_results.pop(0)
                gdist -= self.gcut
                gdist *= mask
                results.append(gdist)
        return results


@attrs.define
class CheapRepulsion(PairwiseTerm):
    """Simple and cheap repulsive potential that smoothly goes to zero at cutoff."""

    rcut: float = attrs.field(converter=float, validator=attrs.validators.gt(0))

    def compute(
        self,
        dist: NDArray[float],
        do_energy: bool = True,
        do_gdist: bool = False,
    ) -> list[NDArray]:
        """Compute pair potential energy and its derivative towards distance."""
        dist = np.asarray(dist, dtype=float)
        x = dist / self.rcut
        results = []
        common = (x - 1) * (x < 1)
        if do_energy:
            energy = common * common
            results.append(energy)
        if do_gdist:
            gdist = (2 / self.rcut) * common
            results.append(gdist)
        return results

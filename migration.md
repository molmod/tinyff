# Migration Guide

This document lists changes you need to make when upgrading to a new version of TinyFF.
Minor versions upgrades should maintain backward compatibility.
Only when the major version changes, you may need to make changes to code using TinyFF


## Upgrading from TinyFF 1.x to 2.x

The required changes are relatively minor:

- You can import all the relevant functions and classes from the top-level `tinyff` package.
  For example, change from:

    ```python
    from tinyff.atomsmithy import build_fcc_lattice
    from tinyff.forcefield import CutOffWrapper, LennardJones, ForceField
    from tinyff.neighborlist import NBuildSimple
    ```

    to

    ```python
    from tinyff import build_fcc_lattice, CutOffWrapper, LennardJones, ForceField, NBuildSimple
    ```

    Note that the old `tinyff.forcefield` has been split into two modules: `forcefield` and `pairwise`.
    By using the top-level imports, your code becomes independent of such internal details.

- When creating a `ForceField` instance, the `nbuild` argument must be specified with a keyword.
  For example, change from:

    ```python
    ff = ForceField([LennardJones(1.0, 1.0)], NBuildSimple(rmax))
    ```

    to

    ```python
    ff = ForceField([LennardJones(1.0, 1.0)], nbuild=NBuildSimple(rmax))
    ```

- When computing energies (and forces), you need to call the `ff.compute` method,
  instead of using the force field object as a function.
  If you want to compute forces and pressure, this must be requested explicitly.
  By default, version 2 will only compute the energy.
  For example, change from:

    ```python
    energy, forces, pressure = ff(atpos, cell_lengths)
    ```

    to

    ```python
    energy, forces, pressure = ff.compute(atpos, cell_lengths, nderiv=1)
    ```

- The pair potentials have undergone a similar change.
  If you want to calculate the derivative, you need to ask for it explicitly.
  By default, only the energy is computed.
  (They are now all derived from `PairwiseTerm` instead of `PairPotential`)
  For example, change from:

    ```python
    lj = LennardJones(1.0, 1.0)
    e, g = lj.compute(dist)
    ```

    to

    ```python
    lj = LennardJones(1.0, 1.0)
    e, g = lj.compute(dist, nderiv=1)
    ```

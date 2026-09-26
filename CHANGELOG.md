# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0] - 2026-09-26

Code and prose cleanup, small performance improvements, and better error messages.

### Changed

- The tests require `numdifftools>=0.10.1`, `pytest>=8.2.0` and `pytest-cov>=5.0.0`.
- The CI tests Python 3.11 with the oldest supported direct dependencies,
  and Python 3.14 with the most recent ones.

### Fixed

- `NBuild.try_move` (and hence `ForceField.try_move`) accepts NumPy integers for `iatom`,
  e.g. obtained with `rng.integers`.
- `compute_rdf` and `compute_acf` accept any array-like input, not only NumPy arrays.
- `NPYWriter` no longer deletes NPY files from an existing directory
  when it cannot be cleaned up completely.
  It raises a `RuntimeError` before removing anything.
- The element symbol in PDB files is right-justified, as required by the PDB format.
- `ForceField.compute` accepts any array-like input for `atpos`,
  also when computing forces.
- `NBuild` raises an error for negative values of `nlist_reuse`.
- The warning for large `nrep` in `build_general_cubic_lattice`
  points to the correct line in the calling code.
- Faster assignment of atoms to bins in `NBuildCellLists`.
- Faster pair generation in `NBuildSimple`.
- Faster accumulation of atomic forces in `ForceField.compute`.
- Much faster `compute_acf`, using a single FFT for all columns.
- Typos and incorrect docstrings.
- Clarified the limitations of `nlist_reuse` and `try_move`,
  and the recommended value of `nbin_approx` (about 30 atoms per bin).
- `NBuild.try_move` no longer triggers a `DeprecationWarning` with NumPy 2.5
  (setting the shape of an array).

## [2.1.0] - 2025-09-17

NumPy 2 is now required.

### Changed

- Bump NumPy and other dependencies.

## [2.0.0] - 2024-10-28

The main changes of this release include support for Monte Carlo algorithms and performance improvements.

### Added

- The `ForceField` object has a new method `compute` that can selectively compute
  the energy only or the energy, forces and the force-contribution to the pressure.
  This improves the efficiency in applications where derivatives are of interest,
  e.g. in Monte Carlo simulations.
- The `ForceField` object is extended with `try_move` and `accept_move`
  to support (relatively) efficient Monte Carlo algorithms with TinyFF.
- Basic analysis routines for radial distribution functions and autocorrelation functions.

### Changed

- Many performance improvements!
- The `ForceField` class requires the `NBuild` instance to be provided as a keyword argument.
  For example: `ForceField([LennardJones()], nbuild=NBuildSimple(rmax))`.
- The `ForceField.__call__` method is replaced by `ForceField.compute`,
  which has a different API with a new `nderiv=0` argument.
  By default, only the energy is computed.
  You must request forces and pressure explicitly by setting `nderiv=1`.
  The function always returns a list of results, even if `nderiv=0`.
- The `PairwiseTerm.compute` (previously `PairwisePotential.compute`) has a new API:
  it takes an `nderiv` argument to decide what is computed
  (energy or energy and derivative).
  It returns a list of requested results.
  By default, only the energy is computed.
- The `ForceTerm.__call__` method has been replaced by `PairwiseTerm.compute_nlist`.
  (This method is primarily for internal usage.)
- Module reorganization to simplify the usage of TinyFF:
  all relevant functions and classes can be imported from the top-level `tinyff` package.
- Module reorganization: all pairwise potentials are now defined in `tinyff.pairwise`,
  instead of `tinyff.forcefield`.
- The `NBuildCellLists` has an additional mandatory keyword argument: `nbin_approx`,
  which is the approximate number of bins in which the cell is split up.
  The recommended setting is `natom / 100`.


### Removed

- The `ForceTerm` base class has been removed.


## [1.0.0] - 2024-10-10

### Changed

- Refactor `ForceField` class to facilitate future extensions.
- Refactor neighborlist API, to prepare for more efficient implementations.


## [0.2.2] - 2024-10-09

### Fixed

- Fix leaking of wrapped coordinates when writing PDB file.


## [0.2.1] - 2024-10-08

### Fixed

- Fix bug in `PairwiseForceField` class: use `rmax` only, never `rcut`.
- Fix version import.
- Fix pressure-related terminology.


## [0.2.0] - 2024-10-07

### Changed

- Add mandatory `rcut` option to `build_random_cell`.


## [0.1.0] - 2024-10-06

### Added

- Add a `stride` option to the trajectory writers.
- Add method `dump_single` method to `PDBWriter` to write one-off file with a single snapshot.

### Changed

- By default, run only 100 optimization steps in `build_random_cell`.
- Wrap atoms back into the cell when writing PDB trajectory files, for nicer visual.
- Stricter consistency checking between multiple `dump` calls in `NPYWriter`.


## [0.0.0] - 2024-10-06

Initial release. See README.md for a description of all features.


[Unreleased]: https://github.com/molmod/tinyff/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/molmod/tinyff/releases/tag/v2.1.0
[2.0.0]: https://github.com/molmod/tinyff/releases/tag/v2.0.0
[1.0.0]: https://github.com/molmod/tinyff/releases/tag/v1.0.0
[0.2.2]: https://github.com/molmod/tinyff/releases/tag/v0.2.2
[0.2.1]: https://github.com/molmod/tinyff/releases/tag/v0.2.1
[0.2.0]: https://github.com/molmod/tinyff/releases/tag/v0.2.0
[0.1.0]: https://github.com/molmod/tinyff/releases/tag/v0.1.0
[0.0.0]: https://github.com/molmod/tinyff/releases/tag/v0.0.0

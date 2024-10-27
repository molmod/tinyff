# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The `ForceField` object has a new method `compute` that can selectively compute
  the energy, the forces and/or the force-contribution to the pressure.
  This improves the efficiency in applications where not all quantities are of interest,
  such as in Monte Carlo simulations.
- The `ForceField` object is extended with `try_move` and `accept_move`
  to support (relatively) efficient Monte Carlo algorithms with TinyFF.


### Changed

- Many performance improvements!
- The `ForceField` class requires the `NBuild` instance to be provided as a keyword argument.
  For example: `ForceField([LennardJones()], nbuild=NBuildSimple)`.
- The `ForceField.__call__` method is replaced by `ForceField.compute`,
  which has a different API with new `do_*` arguments.
  By default, only the energy is computed.
  You must request forces and pressure explicitly.
  The function always returns a list of results, even if there is only one.
- The `PairwiseTerm.compute` (previously `PairwisePotential.compute`) has a new API:
  it takes `do_*` arguments to decide what is computed (energy and or derivative).
  It returns a list of requested results.
  By default, only the energy is computed.
- The `ForceTerm.__call__` method has been replaced by `PairwiseTerm.compute_nlist`.
  (This method is primarily for internal usage.)
- Module reorganization to simplify the usage of TinyFF:
  all relevant functions and classes can be imported from the top-level `tinyff` package.
- Module reorganization: all pairwise potentials are now defined in `tinyff.pairwise`,
  instead of `tinyff.forcefield`.
- The `NBuildCellLists` has an additional mandatory keyword argument: `nbin_approx`,
  which is the approximate number of binds in which the cell is split up.
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


[Unreleased]: https://github.com//molmod/tinyff
[1.0.0]: https://github.com/molmod/tinyff/tag/v1.0.0
[0.2.2]: https://github.com/molmod/tinyff/tag/v0.2.2
[0.2.1]: https://github.com/molmod/tinyff/tag/v0.2.1
[0.2.0]: https://github.com/molmod/tinyff/tag/v0.2.0
[0.1.0]: https://github.com/molmod/tinyff/tag/v0.1.0
[0.0.0]: https://github.com/molmod/tinyff/tag/v0.0.0

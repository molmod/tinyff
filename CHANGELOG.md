# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## Changed

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


[0.0.0]: https://github.com/molmod/tinyff/tag/v0.0.0
[0.1.0]: https://github.com/molmod/tinyff/tag/v0.1.0

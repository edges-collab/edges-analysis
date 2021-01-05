# Changelog

## Version dev

### Changed

- Internally much easier tracking of metadata for levels, and validation of Level data.

## Version 2.0.0

### Added

- Ability to pass arbitrary settings on the command line to the "level" command

### Changed

- Frequency and GHA averaging now done via averaging model and residuals.

## Version 1.1.0

### Added
- Methods for filtering on Level1 objects, and calling them consistently from Level2.
- Store which files are fully flagged in Level2.
- Beam class for easier handling of beam information
- IndexModel class for defining models of spectral indices on the sphere.
- Parallel filtering at Level2
- Working Total Power Filter and RMS filter
- Faster modeling throughout

### Fixed
- Warnings in xrfi are now suppressed, and a summary is given.

## Version 0.1.0

First version compatible with new `edges-collab` set of repos.

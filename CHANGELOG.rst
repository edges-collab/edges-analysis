Changelog
=========

Version dev
-----------

Added
~~~~~
- Ability to fit fiducial models to frequency-binned spectra (residuals still evaluated
  at raw resolution).
- Removal of ``FilteredData`` in favour of filtering adding flags to pre-existing level
  data.
- New ``filter`` CLI command.
- ``from_cst`` beam function.
- New class for binning after combinedData, called CombinedBinnedData
- CLI steps to handle this new processing steps with the same "bin" command
- Added an option of smoothing the beam over frequency

Fixed
~~~~~

- Total power and RMS filters have been completely reworked and work much better now,
  sharing code for simplicity.
- Fixed the at_freq function to use any model specified by the user

Changed
~~~~~~~
- Internally much easier tracking of metadata for levels, and validation of Level data.

Version 2.0.0
-------------

Added
~~~~~

- Ability to pass arbitrary settings on the command line to the "level" command

Changed
~~~~~~~
- Frequency and GHA averaging now done via averaging model and residuals.

Version 1.1.0
-------------
Added
~~~~~
- Methods for filtering on Level1 objects, and calling them consistently from Level2.
- Store which files are fully flagged in Level2.
- Beam class for easier handling of beam information
- IndexModel class for defining models of spectral indices on the sphere.
- Parallel filtering at Level2
- Working Total Power Filter and RMS filter
- Faster modeling throughout

Fixed
~~~~~
- Warnings in xrfi are now suppressed, and a summary is given.

Version 0.1.0
-------------

First version compatible with new ``edges-collab`` set of repos.

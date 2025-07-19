===========================================
EDGES Calibration File Structure Definition
===========================================
**VERSION:** v2.0.0

This document is based on
`memo #113 "Receiver calibration procedure document" <http://loco.lab.asu.edu/loco-memos/edges_reports/tom_20180523_Calibration_Steps.pdf>`_,
by Tom Mozden. It *defines* the directory structure and file naming
conventions used for calibration of EDGES receivers.
Calibration observations that do *not* conform to this structure are considered broken,
and should be either fixed or removed.
This standard is enforced by the code `edges-io <https://github.com/edges-collab/edges-io>`_.

Format
------
The following format description will be as formal and complete as possible. Pay special
attention to modifiers such as "definitely", "possibly" and "solely". In general, there
is a well-defined list of files/folders that should be in any one place, and extra files
will be considered *erroneous*, as they may confuse file readers into reading incorrect
information silently.

A global exception to this rule are files with the following suffices: ``.old``, ``.invalid``,
``.ignore``. These may be kept in the observation and will always be ignored by the
checker and reader. In addition, a file called ``Notes.txt`` *should* be placed at the
root of the structure (i.e. in the ``25C/`` folder).

Throughout the following description, we will use formats such as ``FileXX_YYY.txt``.
In these, some groups of letters are intended to be placeholders for meta-information.
These will *always* be capital letters. They will *usually* be a repeat set of such letters,
but not always. It will *always* be made explicit in the context which letters are meant
to be placeholders. The number of such letters in a given position is strict: the actual
replacement data must match it. For example, in the above, if ``YYY`` represented the
day of the year, it must be inserted as a three-digit string. If the day was the 40th, it
must be inserted as ``040`` rather than simply ``40``.

*Optional* format entries will be marked with square brackets, eg. ``File[XX]_YYY.txt``.
In this case, one could have a file such as ``File_040.txt``, or a file such as
``File30_040.txt``. Each would be valid and identifiable.

Entries that have a limited number of choices may be identified with the following
notation: ``File_(this|that|the_other).txt``. In this case, the filename ``File_this.txt``
would be equally valid as ``File_the_other.txt``. But not ``File_this_that.txt``.

Outline
~~~~~~~
The following directory tree gives a summary view of the structure of an observation.
Here, curly brackets represent the idea that *all* of a given pattern must be present.
Conversely, angle brackets represent the idea that *any* of a given pattern could be
present (including none)::

    ReceiverXX_YYYY_MM_DD_LLL_to_HHH_MHz/
        <15|25|35>C/
            Resistance/
                {Ambient|HotLoad|LongCableOpen|LongCableShorted}_<NN>_YYYY_DDD_HH_MM_SS_lab.csv
                <AntSimX>_{NN}_YYYY_DDD_HH_MM_SS_lab.csv
            Spectra/
                {Ambient|HotLoad|LongCableOpen|LongCableShorted}_<NN>_YYYY_DDD_HH_MM_SS_lab.<h5|acq|mat|npz>
                <AntSimX>_<NN>_YYYY_DDD_HH_MM_SS_lab.{h5|acq|mat|npz}
            S11/
                ReceiverReading<NN>/
                    {ReceiverReading|Open|Short|Match}<RR>.s1p
                SwitchingState<NN>/
                    {Open|Short|Match|ExternalOpen|ExternalShort|ExternalMatch}<RR>.s1p
                {LongCableOpen|LongCableShort|Ambient|HotLoad}<NN>/
                    {Open|Short|Match|External}<RR>.s1p
                <AntSimX><NN>/
                    {Open|Short|Match|External}<RR>.s1p

**Note:** the various occurrences of ``<NN>`` here signify the "run number". Each load
(Ambient/HotLoad/...) may have a unique run number greater than zero, however, the run
number for that load is then the same between resistance, spectra and s11.

We will go through each format in more detail below to identify any vague points and
also describe the metadata of each entry.

Root
~~~~
Format:
    * ``ReceiverXX_YYYY_MM_DD_LLL_to_HHH_MHz/``

Entries:
    * ``XX``: Receiver version number ``(01|02|03)``
    * ``LLL``: Start frequency in MHz
    * ``HHH``: Stop frequency in MHz
    * ``YYYY_MM_DD``: Calibration start date

Example:
    * ``Receiver03_2019_040_to_200_MHz``

The root directory contains *up to* three subdirectories defining the temperature of the
receiver. This "temperature subdirectory" can be considered part of the root name,
as a calibration observation is fully contained in each.

Format:
    * ``<15|25|35>C``.

The options here are temperatures in Celsius.

Top-Level Subdirectories
~~~~~~~~~~~~~~~~~~~~~~~~
Root (with temperature) definitely consists of three sub folders:
    * ``Resistance/``: Contains temperature measurements of thermistor.
    * ``S11/``: Contains VNA measurements in sub folders for different loads and receiver.
    * ``Spectra/``: Contains digitizer output for all loads.

S11 Folder
~~~~~~~~~~
The ``S11`` directory consists *solely* of the following subdirectories. Any other file
or directory will be flagged as an **error** (besides global exceptions defined above).

Each subdirectory contains a number of ``.s1p`` format files. We define explicitly
which files may be contained in each directory below. Nevertheless, we here
emphasise that the format of these files contains a single entry ``<RR>``, called
the "repeat number", which identifies a chronological ordering of when the data was taken,
within a single hook-up.
It is an integer, and the entries in any given directory for any given file kind must
start at one and increment by one. There may be an arbitrary number of repeat numbers for
any given load and standard.

**Note:** only a single repeat number for all *standards* (open, short, match etc.) within
a given *load* (Ambient, SwitchingState etc.) can be *used*. It is never acceptable to
use for example ``Ambient/Open01.s1p`` and ``Ambient/Short02.s1p`` together (though they
can both exist). Thus, an incomplete set of s1p files for a given run number is flagged
as en *error* -- these files should be removed, or defined with an ``.invalid`` suffix.

Contents Format:
    * ``ReceiverReading<NN>/``
        - ``<NN>``: the "run number" of the observation. An integer. Lowest value
          *must* be ``01``, and it must increment by unity. Any number of directories
          may be present. Each represents a repetition of the entire measurement.
        - Contains ``ReceiverReading<RR>.s1p``, ``Short<RR>.s1p``, ``Open<RR>.s1p``
          and ``Match<RR>.s1p``. See notes on ``<RR>`` above. Each corresponds to the
          measurement of a different standard.
    * ``SwitchingState<NN>/``
        - ``<NN>``: See note for ``ReceiverReading<NN>``.
        - Contains ``{Open|Short|Match|ExternalOpen|ExternalShort|ExternalMatch}<RR>.s1p``.
          These are again all measurements of different internal/external standards. Again,
          see notes on ``<RR>`` above.
    * ``{Ambient|HotLoad|LongCableOpen|LongCableShort}<NN>/``
        - *All* of these options *must* be present. They represent the S11 measurements
          of the four calibration loads. Repeat number must be greater or equal to one.
        - Each contains *all* of ``{External|Short|Open|Match}<RR>.s1p``.
    * ``[AntSim<X>]<NN>/``
        - Any number of Antenna Simulators *may* be present (up to 9). If present, ``X``
          identifies the simulator (an integer from 1-9).
        - The contents of an antenna simulation are the same as a Load. All of:
          ``{External|Short|Open|Match}<RR>.s1p``.
        - Repeat number must be greater or equal one.


Spectra Folder
~~~~~~~~~~~~~~
Contents Format:
    * ``{Ambient|HotLoad|LongCableOpen|LongCableShorted}_<NN>_YYYY_DDD_HH_MM_SS_lab.<h5|acq|mat|npz>``

Entries:
    * ``{Ambient|HotLoad|LongCableOpen|LongCableShorted}``: input calibration load. All must exist.
    * <NN>: "run number". Multiple of these may exist for any given load, and other entries can be different for each run num.
      The lowest value for a given load must be ``01`` and they must increment by unity.
    * ``YYYY``: year of observation (must match root folder)
    * ``DDD``: numbered day of year (need not match root folder, but should be close).
    * ``HH``: hour observation started
    * ``MM``: minute observation started
    * ``SS``: second observation started.
    * ``<h5|acq|mat|npz>``: format of the spectrum file. Any may be present (and different ones
      may be present for different loads and run numbers). Current default is to use acq.

Example:
    * ``Ambient_01_2019_351_12_35_56_lab.acq``

Additional contents: there also *may* exist any number of files with the same format, but
with the load name replaced with ``AntSim<X>``, where ``X`` represents the antenna simulator
number (from 1-9).

Resistance Folder
~~~~~~~~~~~~~~~~~
The contents have exactly the same formatting as the ``Spectra/`` folder, except that
the file extension *must* be ``.csv``. The timing entries for the resistance *do not*
need to be the same as their counterpart in ``Spectra/``, nor do there need to be the
same number of runs of each. Nevertheless, all loads (including simulators) in one
*must* be present in the other.

Version History
---------------
**Note:** this version history reflects changes in this file (not the broader ``edges-io``
code), and therefore the standard itself. Versions are in the form ``MAJOR.MINOR.PATCH``,
which correspond to:

* ``PATCH``: a change to this document intended to clarify a point that was already true
  (or formatting changes). Does not change the standard at all.
* ``MINOR``: standard changed in a backwards-compatible way. Eg. a new possible file
  or convention added for which all possible readers will still give the same value.
* ``MAJOR``: backwards-incompatible change. A change such that the reader itself must
  be changed in order to give the same results, or not error. In this case, all
  observations on disk will require updating.

v2.0.0
~~~~~~
* Fixed a bug in the documentation, in which "run number" and "repeat number" were swapped
  for the S11. Also added a clarifying note that only one run number per load is allowed
  for all kinds of measurements (spectra/resistance/s11).

v1.1.0
~~~~~~
* Specified that ``Notes.txt`` must be placed in the root folder rather than anywhere
  in the structure.

v1.0.1
~~~~~~
* Clarification that run-numbers cannot be mixed and matched within S11 measurements.

v1.0.0
~~~~~~
* First version of format standard, based on original memo #113.

"""A sub-package dedicated to implementing a similar interface to the legacy C-code.

This module is not the suggested way to use the edges package, but it can help with
identifying differences between this analysis code and the original (legacy) C code
that was used to obtain e.g. the results of Bowman+2018.

The idea is that here all the parameter names and defaults are the same as in the
C-code, with functions available that replicate entire C-programs.
"""

from .alanio import (
    LOADMAP,
    SPEC_LOADMAP,
    read_alan_calibrated_temp,
    read_modelled_s11s,
    read_raul_s11_format,
    read_s11_csv,
    read_spe_file,
    read_spec_txt,
    read_specal,
    read_specal_iter,
    write_modelled_s11s,
    write_s11_csv,
    write_spec_txt,
    write_spec_txt_gsd,
    write_specal,
)
from .alanmode import (
    ACQPlot7aMoonParams,
    Edges2CalobsParams,
    Edges3CalobsParams,
    EdgesScriptParams,
    acqplot7amoon,
    alancal,
    corrcsv,
    edges,
)

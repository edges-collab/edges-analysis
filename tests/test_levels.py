from edges_analysis.analysis import CalibratedData, CombinedData, DayAveragedData, BinnedData


def test_level1(level1: CalibratedData):
    assert level1[0].raw_frequencies.shape == (8192,)
    assert level1[1].raw_frequencies.shape == (8192,)


def test_level2(level2: CombinedData):
    assert level2.resids.shape[-1] == len(level2.raw_frequencies)
    assert level2.spectrum.shape == level2.resids.shape

    # just run some plotting methods to make sure they don't error...
    level2.plot_daily_residuals(freq_resolution=1.0, gha_max=18, gha_min=6)


def test_level3(level3: DayAveragedData):
    assert level3.resids.shape[-1] == len(level3.raw_frequencies)
    assert level3.spectrum.shape == level3.resids.shape


def test_level4(level4: BinnedData):
    assert level4.resids.shape[-1] == len(level4.raw_frequencies)
    assert level4.resids.shape == level4.spectrum.shape

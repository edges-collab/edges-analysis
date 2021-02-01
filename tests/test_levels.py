from edges_analysis.analysis import Level1, Level2, Level3, Level4


def test_level1(level1: Level1):
    assert level1[0].raw_frequencies.shape == (8192,)
    assert level1[1].raw_frequencies.shape == (8192,)


def test_level2(level2: Level2):
    assert level2.resids.shape[-1] == len(level2.raw_frequencies)
    assert level2.spectrum.shape == level2.resids.shape


def test_level3(level3: Level3):
    assert level3.resids.shape[-1] == len(level3.raw_frequencies)
    assert level3.spectrum.shape == level3.resids.shape


def test_level4(level4: Level4):
    assert level4.resids.shape[-1] == len(level4.raw_frequencies)
    assert level4.resids.shape == level4.spectrum.shape

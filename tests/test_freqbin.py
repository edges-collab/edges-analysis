from edges_analysis.averaging import freqbin


def test_freq_bin_direct(cal_step):
    new = freqbin.freq_bin_direct(cal_step[0], resolution=2)
    assert len(new.freq_array) == len(cal_step[0].freq_array) // 2

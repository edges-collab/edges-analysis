import numpy as np

from edges_analysis import sky_models as sm


def test_haslam_accuracy():
    """Simply test that the two HASLAM variants are close together."""
    h1 = sm.Haslam408()
    h2 = sm.Haslam408AllNoh()

    # Interpolate the healpix one to the same grid as the theta/phi one.
    h1_interp = h1.healpix.interpolate_bilinear_skycoord(h2.coords, h1.temperature)

    # Less than 2% of the pixels should be more than 10% different.
    assert (
        np.sum(np.abs(h1_interp - h2.temperature) / h2.temperature > 0.1)
        / len(h2.temperature)
        < 0.02
    )

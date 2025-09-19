import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as un

from edges.cal import plots
from edges.cal.s11 import CalibratedS11, S11ModelParams


class TestPlotRawSpectrum:
    def test_plot_raw_spectrum(self, calobs):
        plots.plot_raw_spectrum(calobs.ambient.spectrum)

    @pytest.mark.parametrize("xlabel", [True, False])
    @pytest.mark.parametrize("ylabel", [True, False])
    def test_plot_raw_spectrum_array(self, calobs, xlabel, ylabel):
        plots.plot_raw_spectrum(
            calobs.ambient.averaged_q, freq=calobs.freqs, xlabel=xlabel, ylabel=ylabel
        )
        plt.close()  # close to conserve memory


class TestPlotRawSpectra:
    def test_plot_raw_spectra(self, calobs):
        plots.plot_raw_spectra(calobs)
        plt.close()  # close to conserve memory

    def test_plot_raw_spectra_with_fig_ax(self, calobs):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(4, 1)
        outfig = plots.plot_raw_spectra(calobs, fig=fig, ax=ax)
        plt.close(fig)  # close to conserve memory
        assert outfig is fig


class TestPlotS11Residual:
    def setup_class(self):
        rng = np.random.default_rng()
        self.s11 = CalibratedS11(
            s11=np.exp(rng.uniform(size=100) * 1j),
            freqs=np.linspace(100, 200, 100) * un.MHz,
        )
        self.s11_model_params = S11ModelParams()

    @pytest.mark.parametrize(
        ("decade_ticks", "ylabels", "label", "title"),
        [
            (True, True, "label", None),
            (False, False, None, "title"),
            (False, False, None, False),
        ],
    )
    def test_plot_s11_residual(self, decade_ticks, ylabels, title, label):
        plots.plot_s11_residual(
            raw_s11=self.s11,
            s11_model_params=self.s11_model_params,
            decade_ticks=decade_ticks,
            ylabels=ylabels,
            title=title,
            label=label,
        )
        plt.close()  # close to conserve memory

    def test_with_fig_ax(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(4, 1)
        outfig = plots.plot_s11_residual(
            raw_s11=self.s11, s11_model_params=self.s11_model_params, fig=fig, ax=ax
        )
        assert outfig is fig


class TestPlotS11Models:
    def test_plot_s11_models(self, calobs):
        plots.plot_s11_models(
            calobs,
            s11_model_params=S11ModelParams(),
            receiver_model_params=S11ModelParams(),
        )
        plt.close()  # close to conserve memory


class TestPlotCalibratedTemp:
    @pytest.mark.parametrize(
        ("bins", "xlabel", "ylabel"), [(0, True, True), (2, False, False)]
    )
    def test_plot_calibrated_temp(self, calobs, calibrator, bins, xlabel, ylabel):
        plots.plot_calibrated_temp(
            calobs=calobs,
            calibrator=calibrator,
            load="ambient",
            bins=bins,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        plt.close()  # close to conserve memory

    def test_plot_calibrated_temps(self, calobs, calibrator):
        plots.plot_calibrated_temps(calobs, calibrator)
        plt.close()  # close to conserve memory

    def test_plot_calibrated_temps_with_fix_ax(self, calobs, calibrator):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(4, 1)

        outfig = plots.plot_calibrated_temps(calobs, calibrator, fig=fig, ax=ax)
        assert outfig is fig


class TestPlotCalCoefficients:
    def test_plot_cal_coefficients(self, calibrator):
        plots.plot_cal_coefficients(calibrator)
        plt.close()  # close to conserve memory

    def test_plot_cal_coeff_fig_ax(self, calobs, calibrator):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(5, 1)

        outfig = plots.plot_cal_coefficients(calibrator, fig=fig, ax=ax)
        assert outfig is fig

from .analysis.plots import (
    beam_chromaticity_differences,
    plot_antenna_beam,
    plot_antenna_calibration_params,
    plot_balun_loss,
    plot_balun_loss2,
    plot_beam_chromaticity_correction,
    plot_beam_gain,
    plot_beam_power,
    plot_calibration_parameters,
    plot_data_stats,
    plot_ground_loss,
    plot_low_mid_comparison,
    plot_receiver_calibration_params,
    plot_sky_model,
    plot_sky_model_comparison,
)
from .estimation.plots import (
    plot_absorption_model_comparison,
    plot_foreground_polychord_fit,
    plot_triangle_plot,
)


def plots_midband_paper(plot_number, s11_path="antenna_s11_2018_147_17_04_33.txt"):
    if plot_number == 1:
        plot_receiver_calibration_params()
    elif plot_number == 10:
        plot_data_stats()
    elif plot_number == 11:
        plot_low_mid_comparison()
    elif plot_number == 12:
        plot_sky_model_comparison()
    elif plot_number == 121:
        plot_beam_gain()
    elif plot_number == 13:
        plot_sky_model()
    elif plot_number == 14:
        plot_absorption_model_comparison()
    elif plot_number == 2:
        plot_antenna_calibration_params(s11_path)
    elif plot_number == 3:
        plot_balun_loss2(s11_path)
    elif plot_number == 4:
        plot_antenna_beam()
    elif plot_number == 5:
        plot_beam_chromaticity_correction()
    elif plot_number == 50:
        plot_ground_loss()
    elif plot_number == 500:
        plot_calibration_parameters(s11_path)
    elif plot_number == 501:
        plot_balun_loss()
    elif plot_number == 6:
        beam_chromaticity_differences()
    elif plot_number == 7:
        plot_beam_power()
    elif plot_number == 8:
        plot_foreground_polychord_fit()
    elif plot_number == 9:
        plot_triangle_plot()
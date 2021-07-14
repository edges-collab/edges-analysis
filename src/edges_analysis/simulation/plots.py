"""Plotting utilities."""
import matplotlib.pyplot as plt
import numpy as np

from . import data_models as dm


def plot_monte_carlo_receiver():
    """Plot MC receiver."""
    plt.figure(1)

    f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = dm.MC_error_propagation()

    plt.subplot(4, 2, 1)
    plt.plot(f, t1 - m1)
    plt.plot(f, t2 - m2)
    plt.title("Perturbation 1, Low Foreground")
    plt.xticks(np.arange(60, 121, 10), labels=[])
    plt.ylabel("T [K]")
    plt.legend(["Mid-Band Antenna S11", "Low-Band 3 Antenna S11"])

    plt.subplot(4, 2, 2)
    plt.plot(f, t3 - m3)
    plt.plot(f, t4 - m4)
    plt.title("Perturbation 1, High Foreground")
    plt.xticks(np.arange(60, 121, 10), labels=[])

    f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = dm.MC_error_propagation()

    plt.subplot(4, 2, 3)
    plt.plot(f, t1 - m1)
    plt.plot(f, t2 - m2)
    plt.title("Perturbation 2, Low Foreground")
    plt.xticks(np.arange(60, 121, 10), labels=[])
    plt.ylabel("T [K]")

    plt.subplot(4, 2, 4)
    plt.plot(f, t3 - m3)
    plt.plot(f, t4 - m4)
    plt.title("Perturbation 2, High Foreground")
    plt.xticks(np.arange(60, 121, 10), labels=[])

    f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = dm.MC_error_propagation()

    plt.subplot(4, 2, 5)
    plt.plot(f, t1 - m1)
    plt.plot(f, t2 - m2)
    plt.title("Perturbation 3, Low Foreground")
    plt.xticks(np.arange(60, 121, 10), labels=[])
    plt.ylabel("T [K]")

    plt.subplot(4, 2, 6)
    plt.plot(f, t3 - m3)
    plt.plot(f, t4 - m4)
    plt.title("Perturbation 3, High Foreground")
    plt.xticks(np.arange(60, 121, 10), labels=[])

    f, r1, r2, t1, t2, t3, t4, m1, m2, m3, m4 = dm.MC_error_propagation()

    plt.subplot(4, 2, 7)
    plt.plot(f, t1 - m1)
    plt.plot(f, t2 - m2)
    plt.title("Perturbation 4, Low Foreground")
    plt.xlabel("frequency [MHz]")
    plt.ylabel("T [K]")

    plt.subplot(4, 2, 8)
    plt.plot(f, t3 - m3)
    plt.plot(f, t4 - m4)
    plt.title("Perturbation 4, High Foreground")
    plt.xlabel("frequency [MHz]")

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(f, 20 * np.log10(np.abs(r1)))
    plt.plot(f, 20 * np.log10(np.abs(r2)))
    plt.xlabel("frequency [MHz]")
    plt.ylabel("magnitude [dB]")

    plt.legend(["Mid-Band Antenna S11", "Low-Band 3 Antenna S11"])
    plt.subplot(1, 2, 2)
    plt.plot(f, (180 / np.pi) * np.unwrap(np.angle(r1)))
    plt.plot(f, (180 / np.pi) * np.unwrap(np.angle(r2)))
    plt.xlabel("frequency [MHz]")
    plt.ylabel("phase [deg]")


def plot_simulation_residuals(f, out, folder_plot, name_flag):
    """Plot simulation residuals."""

    def plot_it(key, ylim_1, yticks_1, div_1, ylim_2=None, yticks_2=None, div_2=None):
        fig, ax = plt.subplots(1, 2, figsize=[13, 11], sharex=True)

        for i, val in enumerate(out[key]["low"]):
            ax[0].plot(f, val - div_1 * i, "br"[i % 2])

        for i, val in enumerate(out[key]["high"]):
            ax[1].plot(f, val - (div_2 or div_1) * i, "br"[i % 2])

        for axx in ax:
            axx.set_xlim([60, 150])
            axx.grid()
            axx.set_xlabel("frequency [MHz]")

        ax[0].set_ylim(ylim_1)
        ax[0].set_ylabel(f"GHA\n [{div_1} K per division]")
        ax[0].yaxis.yticks(yticks_1, np.arange(17, 5, -1))

        ax[1].set_ylim(ylim_2 or ylim_1)
        ax[1].set_ylabel(f"GHA\n [{div_2 or div_1} K per division]")
        ax[1].yaxis.yticks(yticks_2 or yticks_1, np.arange(5, -7, -1))

        plt.savefig(
            folder_plot + name_flag + f"_simulated_{key}.pdf", bbox_inches="tight"
        )

    plot_it(
        "residuals",
        ylim_1=(-6, 0.5),
        yticks_1=np.arange(-5.5, 0.1, 0.5),
        div_1=0.5,
        ylim_2=(-24, 2),
        yticks_2=np.arange(-22, 0.1, 2),
        div_2=2,
    )
    plot_it(
        "correction", ylim_1=(-0.2, 1.1), yticks_1=np.arange(-0.1, 1.01, 0.1), div_1=0.1
    )
    plot_it(
        "correction_residuals",
        ylim_1=(-0.06, 0.005),
        yticks_1=np.arange(-0.055, 0.0025, 0.005),
        div_1=0.005,
    )

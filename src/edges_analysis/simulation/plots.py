import matplotlib.pyplot as plt
import numpy as np

from . import data_models as dm


def plot_monte_carlo_receiver():
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

import numpy as np
from examples.edges_polychord import v
from getdist import MCSamples, plots
from matplotlib import pyplot as plt
from src.edges_analysis.analysis.scripts import edges_folder
from src.edges_analysis.estimation.models import model
from src.edges_analysis.simulation import data_models as dm


def load_samples(input_textfile, index_good, label_names=[]):
    r"""
    label_names=[r'A\;[{\rm K}]', r'\nu_0\;[{\rm MHz}]', r'w\;[{\rm MHz}]', r'\tau_1', r'\tau_2',
    r'T_{100}\;[{\rm K}]', r'\beta', r'\gamma', r'\delta', r'\epsilon']
    """
    # TODO: this function is really redundant -- use loadMCSamples instead.

    # Loading data
    dd = np.genfromtxt(input_textfile)
    d = dd[index_good::, :]

    ww = d[:, 0]  # Weights
    ll = d[:, 1] / 2  # Minus Log Likelihood
    ss = d[:, 2::]  # Parameter samples

    # Parameter names and labels
    Npar = len(ss[0, :])
    names = ["x" + str(i + 1) for i in range(Npar)]

    if len(label_names) == 0:
        labels = [r"a_" + str(i) for i in range(Npar)]
    else:
        labels = label_names

    # Convert samples into GETDIST format
    getdist_samples = MCSamples(
        samples=ss, weights=ww, loglikes=ll, names=names, labels=labels, label=r"only"
    )

    # Best fit and covariance matrix
    IX = np.argmin(d[:, 1])  # the maximum likelihood point
    best_fit = d[IX, 2::]

    covariance_matrix = getdist_samples.cov()

    return getdist_samples, ww, ll, best_fit, covariance_matrix


def triangle_plot():
    d = np.genfromtxt("/home/raul/Desktop/hu.txt")

    ss = d[:, 2::]
    ww = d[:, 0]
    ll = d[:, 1] / 2

    # names  = ['x1', 'x2', 'x3', 'x4', 'x5']
    # labels = [r'a_0', r'a_1', r'a_2', r'a_3', r'a_4']

    names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
    labels = [
        r"T_{21}",
        r"\nu_r",
        r"\Delta\nu",
        r"\tau",
        r"a_0",
        r"a_1",
        r"a_2",
        r"a_3",
        r"a_4",
    ]

    samples = MCSamples(
        samples=ss, weights=ww, loglikes=ll, names=names, labels=labels, label=r"only"
    )

    g = plots.getSubplotPlotter(subplot_size=3.3)
    g.settings.legend_fontsize = 15
    g.settings.lab_fontsize = 15
    g.settings.axes_fontsize = 15
    # g.settings.tight_layout=False
    g.triangle_plot(
        [samples], filled=True
    )  # , param_limits={'x1':[1498, 1502], 'x2':[-2.504,
    # -2.496]}, filled=True, legend_loc='upper right')
    g.export("/home/raul/Desktop/output_file_X.pdf")
    plt.close()
    plt.close()

    return samples


def model_plot():
    t = model([1500, -2.5])
    s3 = model([1, -2.5])

    s1 = 2 * np.ones(len(v))
    s2 = 1 * np.ones(len(v))

    plt.close()
    plt.close()

    plt.figure(figsize=[6, 8])
    plt.subplot(2, 1, 1)
    plt.plot(v, t, "k")

    plt.ylabel("temperature [K]")
    plt.ylim([0, 5000])

    plt.subplot(2, 1, 2)
    plt.plot(v, s1, color=[0.7, 0.7, 0.7])
    plt.plot(v, s2, "r")
    plt.plot(v, s3, "b")

    plt.xlabel("frequency [MHz]")
    plt.ylabel("temperature [K]")
    plt.legend(
        [
            r"$\sigma=2\;\;\rm{[K]}$",
            r"$\sigma=1\;\;\rm{[K]}$",
            r"$\sigma = \left(\frac{\nu}{75\;\;\rm{MHz}}\right)^{-2.5}\;\;\rm{[K]}$",
        ]
    )

    plt.ylim([0, 3])

    plt.savefig("models.pdf", bbox_inches="tight")


def chains_plot():
    d3 = np.genfromtxt("/home/ramo7131/Desktop/chains/test3.txt")
    d2 = np.genfromtxt("/home/ramo7131/Desktop/chains/test2.txt")
    d4 = np.genfromtxt("/home/ramo7131/Desktop/chains/test4.txt")

    ss3 = d3[:, 2::]
    ss2 = d2[:, 2::]
    ss4 = d4[:, 2::]

    FS = 15

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ss2[:, 0], color=[0.7, 0.7, 0.7])
    plt.plot(ss3[:, 0], "r")
    plt.plot(ss4[:, 0], "b")
    plt.ylabel(r"$T_{75}\;\; \rm [K]$", fontsize=FS)
    plt.legend(
        [
            r"$\sigma=2\;\;\rm{[K]}$",
            r"$\sigma=1\;\;\rm{[K]}$",
            r"$\sigma = \left(\frac{\nu}{75\;\;\rm{MHz}}\right)^{-2.5}\;\;\rm{[K]}$",
        ]
    )

    plt.subplot(2, 1, 2)
    plt.plot(ss2[:, 1], color=[0.7, 0.7, 0.7])
    plt.plot(ss3[:, 1], "r")
    plt.plot(ss4[:, 1], "b")
    plt.ylabel(r"$\beta$", fontsize=FS)

    plt.xlabel("sample number")
    plt.savefig("chains.pdf", bbox_inches="tight")


def plot_absorption_model_comparison():
    f = np.arange(60, 120.5, 0.5)
    b18 = dm.signal_model("exp", [-0.5, 78, 19, 7], f)
    x1 = dm.signal_model("exp", [-0.5, 78, 19, 7, -0.5], f)
    x2 = dm.signal_model("exp", [-0.5, 78, 19, 7, 0.5], f)
    t1 = dm.signal_model("tanh", [-0.5, 78, 19, 3, 7], f)
    t2 = dm.signal_model("tanh", [-0.5, 78, 19, 7, 3], f)
    x0_top = 0.1
    y0_top = 0.5
    x0_bottom = 0.1
    y0_bottom = 0.1
    dx = 0.85
    dy = 0.4
    fig = plt.figure(figsize=[3.8, 4.3])
    ax = fig.add_axes([x0_top, y0_top, dx, dy])
    ax.plot(f, b18, "k")
    ax.plot(f, x1, "b--")
    ax.plot(f, x2, "b:")
    plt.ylim([-0.7, 0.1])
    plt.yticks(np.arange(-0.6, 0.05, 0.2))
    plt.text(59, -0.65, "(a)", fontsize=15)
    plt.legend(
        [r"Bowman et al. (2018)", r"Exp Model, $\chi=-0.5$", r"Exp Model, $\chi=+0.5$"],
        fontsize=8,
    )
    plt.ylabel("brightness\n temperature [K]")
    ax = fig.add_axes([x0_bottom, y0_bottom, dx, dy])
    ax.plot(f, b18, "k")
    ax.plot(f, t1, "r--")
    ax.plot(f, t2, "r:")
    # plt.plot(f, b18, 'k')
    plt.ylim([-0.7, 0.1])
    plt.yticks(np.arange(-0.6, 0.05, 0.2))
    plt.text(59, -0.65, "(b)", fontsize=15)
    plt.legend(
        [r"Bowman et al. (2018)", r"Tanh Model, $\tau_1=3$", r"Tanh Model, $\tau_2=3$"],
        fontsize=8,
    )
    plt.xlabel(r"$\nu$ [MHz]", fontsize=13)
    plt.ylabel("brightness\n temperature [K]")
    # Saving
    plt.savefig(
        edges_folder + "/plots/20190730/absorption_models.pdf", bbox_inches="tight"
    )


def plot_triangle_plot():
    filename = (
        edges_folder + "mid_band/polychord/20190617/case2"
        "/foreground_exp_signal_exp_4par_60_120MHz/chain.txt"
    )
    label_names = [
        r"A\;[{\rm K}]",
        r"\nu_0\;[{\rm MHz}]",
        r"w\;[{\rm MHz}]",
        r"\tau",
        r"T_{100}\;[{\rm K}]",
        r"\beta",
        r"\gamma",
        r"\delta",
        r"\epsilon",
    ]
    getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
        filename, 0, label_names=label_names
    )
    # reordered_getdist_samples = np.flip(getdist_samples)
    output_pdf_filename = (
        edges_folder + "plots/20190617/triangle_plot_exp_exp_4par_60_120MHz.pdf"
    )
    triangle_plot(
        getdist_samples, output_pdf_filename, legend_FS=10, label_FS=18, axes_FS=7
    )


def plot_foreground_polychord_fit():
    FLOW = 58
    FHIGH = 120
    vr = 90
    path_plot_save = edges_folder + "plots/20190815/"
    figure_plot_save = "powerlog_5par_exp_4par_v1.pdf"
    fg = "powerlog5"
    signal = "exp4"
    d = np.genfromtxt(
        edges_folder + "mid_band/spectra/level5/case_nominal"
        "/integrated_spectrum_case_nominal_days_186_219_58-120MHz.txt"
    )
    v = d[:, 0]
    t = d[:, 1]
    w = d[:, 2]
    s = d[:, 3]
    t[w == 0] = np.nan
    s[w == 0] = np.nan
    # Best-fit foreground model
    if fg == "linlog5":
        filename_foreground = (
            edges_folder
            + "mid_band/polychord/20190815/case_nominal/foreground_linlog_5par_v2"
            "/chain.txt"
        )
        label_foreground = [r"a_0", r"a_1", r"a_2", r"a_3", r"a_4"]
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            filename_foreground, 0, label_names=label_foreground
        )
        model_fg = dm.foreground_model(
            "linlog", best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0
        )
    if fg == "powerlog5":
        filename_foreground = (
            edges_folder
            + "mid_band/polychord/20190815/case_nominal/foreground_powerlog_5par"
            "/chain.txt"
        )
        label_foreground = [
            r"T_{90}\;[{\rm K}]",
            r"\beta",
            r"\gamma",
            r"\delta",
            r"\epsilon",
        ]
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            filename_foreground, 0, label_names=label_foreground
        )
        model_fg = dm.foreground_model(
            "powerlog", best_fit, v, vr, ion_abs_coeff=0, ion_emi_coeff=0
        )
    # Best-fit foreground model + signal model
    full = fg + "_" + signal
    if full == "linlog5_exp4":
        filename_foreground_plus_signal = (
            edges_folder
            + "mid_band/polychord/20190815/case_nominal/foreground_linlog_5par_signal_exp_4par_v2"
            "/chain.txt"
        )
        label_foreground_plus_signal = [
            r"A\;[{\rm K}]",
            r"\nu_0\;[{\rm MHz}]",
            r"w\;[{\rm MHz}]",
            r"\tau",
            r"a_0",
            r"a_1",
            r"a_2",
            r"a_3",
            r"a_4",
        ]
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal
        )
        full_model = dm.full_model(
            best_fit,
            v,
            vr,
            model_type_signal="exp",
            model_type_foreground="linlog",
            N21par=4,
            NFGpar=5,
        )
    if full == "powerlog5_exp4":
        filename_foreground_plus_signal = (
            edges_folder
            + "mid_band/polychord/20190815/case_nominal/foreground_powerlog_5par_signal_exp_4par"
            "/chain.txt"
        )
        label_foreground_plus_signal = [
            r"A\;[{\rm K}]",
            r"\nu_0\;[{\rm MHz}]",
            r"w\;[{\rm MHz}]",
            r"\tau",
            r"T_{90}\;[{\rm K}]",
            r"\beta",
            r"\gamma",
            r"\delta",
            r"\epsilon",
        ]
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal
        )
        full_model = dm.full_model(
            best_fit,
            v,
            vr,
            model_type_signal="exp",
            model_type_foreground="powerlog",
            N21par=4,
            NFGpar=5,
        )
    if full == "powerlog5_tanh5":
        filename_foreground_plus_signal = (
            edges_folder
            + "mid_band/polychord/20190811/case_nominal/foreground_powerlog_5par_signal_tanh_5par"
            "/chain.txt"
        )
        label_foreground_plus_signal = [
            r"A\;[{\rm K}]",
            r"\nu_0\;[{\rm MHz}]",
            r"w\;[{\rm MHz}]",
            r"\tau_1",
            r"\tau_2",
            r"T_{90}\;[{\rm K}]",
            r"\beta",
            r"\gamma",
            r"\delta",
            r"\epsilon",
        ]
        getdist_samples, ww, ll, best_fit, covariance_matrix = load_samples(
            filename_foreground_plus_signal, 0, label_names=label_foreground_plus_signal
        )
        full_model = dm.full_model(
            best_fit,
            v,
            vr,
            model_type_signal="tanh",
            model_type_foreground="powerlog",
            N21par=5,
            NFGpar=5,
        )
    # Best-fit signal model
    if signal == "exp4":
        model_signal = dm.signal_model("exp", best_fit[0:4], v)
    if signal == "tanh5":
        model_signal = dm.signal_model("tanh", best_fit[0:5], v)
    model_edges2018, x2, limits = dm.signal_edges2018_uncertainties(v)
    model_edges2018_A1 = limits[:, 0]
    model_edges2018_A2 = limits[:, 1]
    x0b = 0.35
    y0b = 0.525
    dyb = 0.1
    x0a = 0.35
    y0a = 0.625
    dya = 0.2
    x0c = 0.1
    y0c = 0.37
    dyc = 0.11
    x0d = 0.58
    y0d = 0.37
    dyd = 0.11
    x0e = 0.1
    y0e = 0.1
    dye = 0.25
    x0f = 0.58
    y0f = 0.1
    dx = 0.4
    dyf = 0.25
    f1 = plt.figure(1, figsize=[8, 10])
    # Panel a
    # ---------------------------------
    ax = f1.add_axes([x0a, y0a, dx, dya])
    ax.plot(v, t, "b", linewidth=1)
    ax.set_xlim([FLOW, FHIGH])
    ax.set_ylim([250, 3500])
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(500, 3001, 500))
    ax.set_yticklabels(["500", "1000", "1500", "2000", "2500", "3000"])
    ax.set_ylabel("T$_b$ [K]", fontsize=13)
    ax.text(114, 2900, "(a)", fontsize=14)
    # Panel b
    # ---------------------------------
    ax = f1.add_axes([x0b, y0b, dx, dyb])
    ax.plot(v, s, "b", linewidth=1)
    ax.plot(v, -s, "b", linewidth=1)
    ax.plot(v, np.zeros(len(v)), "k--")
    ax.set_xlim([FLOW, FHIGH])
    ax.set_ylim([-0.07, 0.07])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_ylabel(r"$\sigma_b$ [K]", fontsize=13)
    ax.text(114, 0.03, "(b)", fontsize=14)
    # Panel c
    # ---------------------------------
    ax = f1.add_axes([x0c, y0c, dx, dyc])
    ax.plot(v, t - model_fg, "b", linewidth=1)
    ax.set_xlim([FLOW, FHIGH])
    ax.set_ylim([-0.3, 0.3])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(-0.2, 0.21, 0.2))
    ax.set_yticklabels(["-0.2", "0", "0.2"])
    ax.set_ylabel("T$_b$ [K]", fontsize=13)
    ax.text(114, 0.170, "(c)", fontsize=14)
    # Panel d
    # ---------------------------------
    ax = f1.add_axes([x0d, y0d, dx, dyd])
    ax.plot(v, t - full_model, "b", linewidth=1)
    ax.set_xlim([FLOW, FHIGH])
    ax.set_ylim([-0.3, 0.3])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(-0.2, 0.21, 0.2))
    ax.set_yticklabels(["-0.2", "0", "0.2"])
    ax.text(114, 0.170, "(d)", fontsize=14)
    # Panel e
    # ---------------------------------
    ax = f1.add_axes([x0e, y0e, dx, dye])
    ax.plot(v, model_signal, "b", linewidth=1)
    ax.plot(v, model_edges2018, "r", linewidth=0.5)
    ax.plot(v, model_edges2018_A1, "r--", linewidth=0.5)
    ax.plot(v, model_edges2018_A2, "r--", linewidth=0.5)
    ax.plot(v, model_signal, "b", linewidth=1)
    ax.set_xlim([FLOW, FHIGH])
    ax.set_ylim([-1.3, 0.1])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(-1.2, 0.1, 0.2))
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=13)
    ax.set_ylabel("T$_b$ [K]", fontsize=13)
    ax.text(114, -0.25, "(e)", fontsize=14)  # , fontweight='bold')
    # Panel f
    # ---------------------------------
    ax = f1.add_axes([x0f, y0f, dx, dyf])
    ax.plot(v, model_signal + (t - full_model), "b", linewidth=1)
    ax.plot(v, model_edges2018, "r", linewidth=0.5)
    ax.plot(v, model_edges2018_A1, "r--", linewidth=0.5)
    ax.plot(v, model_edges2018_A2, "r--", linewidth=0.5)
    ax.plot(v, model_signal + (t - full_model), "b", linewidth=1)
    ax.set_xlim([FLOW, FHIGH])
    ax.set_ylim([-1.3, 0.1])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(-1.2, 0.1, 0.2))
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=13)
    ax.text(114, -0.25, "(f)", fontsize=14)
    ax.legend(["Mid-Band", "Bowman et al. (2018)"], fontsize=9, loc=4)
    # Saving plot
    plt.savefig(path_plot_save + figure_plot_save, bbox_inches="tight")
    plt.close()
    plt.close()
    plt.close()
    plt.close()

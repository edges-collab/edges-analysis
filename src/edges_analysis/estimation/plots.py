import numpy as np
from getdist import loadMCSamples, plots
from matplotlib import pyplot as plt
from ..simulation import data_models as dm
from ..config import config


def load_samples(file_root, index_good=0):
    samples = loadMCSamples(file_root, settings={"ignore_row": index_good})
    best_fit = samples.getBestFit().getParamDict()
    best_fit = [best_fit[name] for name in samples.paramNames.names]
    covariance_matrix = samples.cov()

    return samples, samples.weights, samples.loglikes, best_fit, covariance_matrix


def plot_absorption_model_comparison(models, labels, styles=None):
    """
    Plot a comparison of different absorption models with the fiducial Bowman+18 model.

    Parameters
    ----------
    models : list of tuples
        A list of models to compare. Each element should be a tuple, with the first
        element a string -- either 'exp' or 'tanh', indicating the kind of model,
        and the second element being an iterable of parameter values.
    labels : list of str
        Labels for each model for the plot legend.
    styles : list, optional
         A list of strings defining the matplotlib style (eg. 'r-') for each model.
         Note the fiducial model is simply 'k'.

    """
    f = np.arange(60, 120.5, 0.5)

    mdls = {label: dm.signal_model(model[0], model[1], f) for label, model in zip(models, labels)}
    b18 = dm.signal_model("exp", [-0.5, 78, 19, 7], f)

    # x1 = dm.signal_model("exp", [-0.5, 78, 19, 7, -0.5], f)
    # x2 = dm.signal_model("exp", [-0.5, 78, 19, 7, 0.5], f)
    # t1 = dm.signal_model("tanh", [-0.5, 78, 19, 3, 7], f)
    # t2 = dm.signal_model("tanh", [-0.5, 78, 19, 7, 3], f)
    # [r"Bowman et al. (2018)", r"Exp Model, $\chi=-0.5$", r"Exp Model, $\chi=+0.5$"],
    nkinds = len({m[0] for m in models})
    fig, ax = plt.subplots(nkinds, 1, figsize=[3.8, 4.3])

    ax[0].plot(f, b18, "k", label=r"Bowman et al. (2018)")
    if nkinds == 2:
        ax[1].plot(f, b18, "k", label=r"Bowman et al. (2018)")

    if styles is None:
        styles = [None] * len(mdls)

    for (label, mdl), (kind, _), style in zip(mdls.items(), models, styles):
        axi = ["exp", "tanh"].index(kind) if nkinds == 2 else 0
        ax[axi].plot(f, mdl, style, label=label)

    for axx in ax:
        axx.set_ylim([-0.7, 0.1])
        axx.yaxis.set_yticks(np.arange(-0.6, 0.05, 0.2))
        axx.legend(fontsize=8)
        axx.set_ylabel("brightness\n temperature [K]")

    if nkinds == 2:
        ax[0].text(59, -0.65, "(a)", fontsize=15)
        ax[1].text(59, -0.65, "(b)", fontsize=15)

    ax[-1].set_xlabel(r"$\nu$ [MHz]", fontsize=13)

    return fig, ax


def triangle_plot(file_root, output_file):
    samples = load_samples(file_root, 0)[0]

    g = plots.getSubplotPlotter(subplot_size=3.3)
    g.settings.legend_fontsize = 10
    g.settings.lab_fontsize = 18
    g.settings.axes_fontsize = 7
    # g.settings.tight_layout=False
    g.triangle_plot([samples], filled=True)
    g.export(output_file)


def plot_foreground_polychord_fit(
    datafile,
    fg_chain_root,
    full_chain_root,
    f_low,
    f_high,
    save_path,
    fg="powerlog5",
    signal="exp4",
    vr=90,
):

    figure_plot_save = "{}_{}.pdf".format(fg, signal)

    v, t, w, s = np.genfromtxt(datafile).T
    t[w == 0] = np.nan
    s[w == 0] = np.nan

    best_fit_fg = load_samples(fg_chain_root)[-2]
    best_fit_full = load_samples(full_chain_root)[-2]

    n_fg = int(fg[-1])
    n_21 = int(signal[-1])
    fg = fg[:-1]
    signal = signal[:-1]

    model_fg = dm.foreground_model(fg, best_fit_fg, v, vr, ion_abs_coeff=0, ion_emi_coeff=0)
    full_model = dm.full_model(
        best_fit_full,
        v,
        vr,
        model_type_signal=signal,
        model_type_foreground=fg,
        n_21=n_21,
        n_fgpar=n_fg,
    )

    model_signal = dm.signal_model("exp", best_fit_full[:n_21], v)

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
    ax.set_xlim([f_low, f_high])
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
    ax.set_xlim([f_low, f_high])
    ax.set_ylim([-0.07, 0.07])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_ylabel(r"$\sigma_b$ [K]", fontsize=13)
    ax.text(114, 0.03, "(b)", fontsize=14)
    # Panel c
    # ---------------------------------
    ax = f1.add_axes([x0c, y0c, dx, dyc])
    ax.plot(v, t - model_fg, "b", linewidth=1)
    ax.set_xlim([f_low, f_high])
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
    ax.set_xlim([f_low, f_high])
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
    ax.set_xlim([f_low, f_high])
    ax.set_ylim([-1.3, 0.1])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(-1.2, 0.1, 0.2))
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=13)
    ax.set_ylabel("T$_b$ [K]", fontsize=13)
    ax.text(114, -0.25, "(e)", fontsize=14)
    # Panel f
    # ---------------------------------
    ax = f1.add_axes([x0f, y0f, dx, dyf])
    ax.plot(v, model_signal + (t - full_model), "b", linewidth=1)
    ax.plot(v, model_edges2018, "r", linewidth=0.5)
    ax.plot(v, model_edges2018_A1, "r--", linewidth=0.5)
    ax.plot(v, model_edges2018_A2, "r--", linewidth=0.5)
    ax.plot(v, model_signal + (t - full_model), "b", linewidth=1)
    ax.set_xlim([f_low, f_high])
    ax.set_ylim([-1.3, 0.1])
    ax.set_xticks(np.arange(60, 121, 10))
    ax.set_yticks(np.arange(-1.2, 0.1, 0.2))
    ax.set_xlabel(r"$\nu$ [MHz]", fontsize=13)
    ax.text(114, -0.25, "(f)", fontsize=14)
    ax.legend(["Mid-Band", "Bowman et al. (2018)"], fontsize=9, loc=4)
    # Saving plot
    plt.savefig(save_path + figure_plot_save, bbox_inches="tight")

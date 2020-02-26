import src.edges_analysis.estimation.plots

from ..simulation import data_models as dm

edges_folder = ""  # TODO: remove


def plots_midband_polychord(fig):
    # TODO: replace this entire function with a simple script function.

    if fig == 0:
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 1:
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_linlog/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[r"a_0", r"a_1", r"a_2", r"a_3", r"a_4"],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=12, axes_FS=8
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="linlog",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 10:
        # Data used:  60-120
        folder = (
            edges_folder + "mid_band/polychord/20190508/case1_nominal"
            "/foreground_model_exp_signal_model_tanh_60_120MHz/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau_1",
                r"\tau_2",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=13, axes_FS=7
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="tanh",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    elif fig == 11:
        # Data used:  60-120, CASE 2, cterms7, wterms8
        folder = (
            edges_folder + "mid_band/polychord/20190508/case1_nominal"
            "/foreground_model_exp_signal_model_tanh_60_120MHz_case2/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau_1",
                r"\tau_2",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=13, axes_FS=7
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(2, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="tanh",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    elif fig == 12:
        # Data used:  60-120
        folder = (
            edges_folder + "mid_band/polychord/20190508/case1_nominal"
            "/foreground_model_exp_signal_model_exp_60_120MHz/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            2000,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau",
                r"\chi",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=13, axes_FS=7
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 120)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    elif fig == 2:
        # Data used:  60-67, 103-119.5
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 3:
        # Data used:  60-65, 103-119.5
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap2/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 4:
        # Data used:  60-65, 95-119.5
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap3/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, abel_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 119.5)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 5:
        # Data used:  60-65, 95-115
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap4/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 6:
        # Data used:  60-65, 100-115
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap5/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 7:
        # Data used:  60-65, 97-115
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap6/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 8:
        # Data used:  60-65, 100-115, CASE 2, cterms7, wterms8
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_gap7/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(2, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="exp",
            model_type_foreground="exp",
            n_21=0,
            n_fgpar=5,
        )
    elif fig == 9:
        # Data used:  60-115
        folder = (
            edges_folder
            + "mid_band/polychord/20190508/case1_nominal/foreground_model_exp_signal_model_tanh/"
        )
        getdist_samples, ww, ll, best_fit, covariance_matrix = gp.load_samples(
            folder + "chain.txt",
            0,
            label_names=[
                r"A\;[{\rm K}]",
                r"\nu_0\;[{\rm MHz}]",
                r"w\;[{\rm MHz}]",
                r"\tau_1",
                r"\tau_2",
                r"T_{100}\;[{\rm K}]",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\epsilon",
            ],
        )
        src.edges_analysis.estimation.plots.triangle_plot(
            getdist_samples, folder + "result.pdf", legend_FS=10, label_FS=10, axes_FS=5
        )

        v, t, w, sigma, inv_sigma, det_sigma = dm.real_data(10, 60, 115)

        model = dm.full_model(
            best_fit,
            v,
            100,
            model_type_signal="tanh",
            model_type_foreground="exp",
            n_21=5,
            n_fgpar=5,
        )
    return v, t, w, model

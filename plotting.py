"""

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse



def init_matplotlib_params(save_not_show_fig, show_latex_fig):
    fontsize = 8
    linewidth = 1.0
    gridlinewidth = 0.7

    # Global changes
    matplotlib.rcParams.update({
            # Fonts
            'font.size': fontsize,
            'axes.titlesize': fontsize,
            'axes.labelsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize,
            'figure.titlesize': fontsize,
            # Line width
            'lines.linewidth': linewidth,
            'grid.linewidth': gridlinewidth
        })

    # Backend if saving
    if save_not_show_fig:
        matplotlib.use("pgf")

    # Font if saving or ploting in tex mode
    if save_not_show_fig or show_latex_fig:
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",   
            'font.family': 'serif',         # Use serif/main font for text elements
            'text.usetex': True,            # Use inline maths for ticks
            'pgf.rcfonts': False,           # Don't setup fonts from matplotlib rc params
        })
    return

# From https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals) # eigvals positive because covariance is positive semi definite
    return Ellipse(xy=centre, width=width, height=height, angle=np.degrees(theta), **kwargs)


def plot_state_cov(plotter, state_covariance, state, **kwargs):
    state_2d = np.array([state[0], state[2]])
    state_cov_2d = np.array([[state_covariance[0][0], state_covariance[2][0]],
                             [state_covariance[0][2], state_covariance[2][2]]])
    ellipse = get_cov_ellipse(state_cov_2d, state_2d, 2, **kwargs)
    return plotter.add_artist(ellipse)


def plot_single_sim(sim_data):
    fig = plt.figure()
    ax_sim = fig.add_subplot(121)
    ax_trace = fig.add_subplot(122)

    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colour_map = plt.cm.get_cmap('plasma_r')

    # Ground truth
    ax_sim.plot([x[0] for x in sim_data.gt], [x[2] for x in sim_data.gt], c='lightgray', linestyle='-', linewidth=1.0)

    # Nosied measurements
    for s in range(sim_data.num_sensors):
        ax_sim.scatter([x[0] for x in sim_data.zs[s]], [x[1] for x in sim_data.zs[s]], c=colours[s], marker='.', s=1.0)

    # Unpriv
    ax_sim.plot([x[0][0] for x in sim_data.unpriv_filter_results], [x[0][2] for x in sim_data.unpriv_filter_results], c='black', linestyle='-', linewidth=1.0)
    for i in range(sim_data.sim_len):
        if i % 10 == 0:
            plot_state_cov(ax_sim, sim_data.unpriv_filter_results[i][1], sim_data.unpriv_filter_results[i][0], color='black', fill=False, linestyle='-', linewidth=1.0)
    ax_trace.plot([x for x in range(sim_data.sim_len)], [np.trace(x[1]) for x in sim_data.unpriv_filter_results], c='black', linestyle='-', linewidth=1.0)

    # Priv only denoised
    for s in range(sim_data.num_sensors):
        clr = colour_map((s+1)/sim_data.num_sensors)
        ax_sim.plot([x[0][0] for x in sim_data.priv_filters_j_ms_results[s]], [x[0][2] for x in sim_data.priv_filters_j_ms_results[s]], c=clr, linestyle='-', linewidth=1.0)
        for i in range(sim_data.sim_len):
            plot_state_cov(ax_sim, sim_data.priv_filters_j_ms_results[s][i][1], sim_data.priv_filters_j_ms_results[s][i][0], color=clr, fill=False, linestyle='-', linewidth=1.0)
        ax_trace.plot([x for x in range(sim_data.sim_len)], [np.trace(x[1]) for x in sim_data.priv_filters_j_ms_results[s]], c=clr, linestyle='-', linewidth=1.0)

    # Priv all
    for s in range(sim_data.num_sensors):
        clr = colour_map((s+1)/sim_data.num_sensors)
        ax_sim.plot([x[0][0] for x in sim_data.priv_filters_all_ms_results[s]], [x[0][2] for x in sim_data.priv_filters_all_ms_results[s]], c=clr, linestyle='--', linewidth=1.0)
        for i in range(sim_data.sim_len):
            plot_state_cov(ax_sim, sim_data.priv_filters_all_ms_results[s][i][1], sim_data.priv_filters_all_ms_results[s][i][0], color=clr, fill=False, linestyle='--', linewidth=1.0)
        ax_trace.plot([x for x in range(sim_data.sim_len)], [np.trace(x[1]) for x in sim_data.priv_filters_all_ms_results[s]], c=clr, linestyle='--', linewidth=1.0)

    plt.show()
    return


def plot_privilege_differences(avg_sim_data, save_not_show, show_as_tex):
    init_matplotlib_params(save_not_show, show_as_tex)

    # TODO figure size for paper
    fig = plt.figure()

    ax_priv_1 = fig.add_subplot(221)
    ax_priv_2 = fig.add_subplot(222)
    ax_priv_3 = fig.add_subplot(223)
    ax_priv_4 = fig.add_subplot(224)

    # TODO adjust subplots (sizing)

    # Colours
    colour_map = plt.cm.get_cmap('plasma_r')

    # Unpriv in each plot
    ax_priv_1.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filter_errors_avg], linestyle='-', c='black')
    ax_priv_2.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filter_errors_avg], linestyle='-', c='black')
    ax_priv_3.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filter_errors_avg], linestyle='-', c='black')
    ax_priv_4.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filter_errors_avg], linestyle='-', c='black')

    # Priv only denoised at each privilege
    ax_priv_1.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[0]], linestyle='--', c=colour_map(1/4))
    ax_priv_2.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[1]], linestyle='--', c=colour_map(2/4))# TODO looks wrong
    ax_priv_3.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[2]], linestyle='--', c=colour_map(3/4))
    ax_priv_4.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[3]], linestyle='--', c=colour_map(4/4))

    # Priv all at each privilege
    ax_priv_1.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[0]], linestyle='-', c=colour_map(1/4))
    ax_priv_2.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[1]], linestyle='-', c=colour_map(2/4))
    ax_priv_3.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[2]], linestyle='-', c=colour_map(3/4))
    ax_priv_4.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[3]], linestyle='-', c=colour_map(4/4))

    # TODO legend

    # Save or show picture
    if save_not_show:
        plt.savefig('pictures/privilege_differences.pdf')
    else:
        plt.show()

    return



    # # For plotting
    # fig = plt.figure()
    # ax_sim = fig.add_subplot(121)
    # ax_trace = fig.add_subplot(122)
    # gts = []
    # all_zs = []
    # #filter_priv_ests = []
    # filters_priv_ests = []
    # filters_unpriv_ests = []
    # #filter_fused = []
    # colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # colour_map = plt.cm.get_cmap('plasma_r')


    #     # Plotting

    # # Ground truth
    # ax_sim.plot([x[0] for x in gts], [x[2] for x in gts], c='gray', linestyle='-', linewidth=1.0)

    # # Sensors
    # for sen in range(num_sensors):
    #     rel_zs = [z[sen] for z in all_zs]
    #     ax_sim.scatter([x[0] for x in rel_zs], [x[1] for x in rel_zs], c='orange', marker='.', s=1.0)

    # # # Privileged filter
    # # ax_sim.plot([x[0][0] for x in filter_priv_ests], [x[0][2] for x in filter_priv_ests], c='green', linestyle='-', linewidth=1.0)
    # # plot_cntr = 0
    # # for x,P in filter_priv_ests:
    # #     plot_cntr += 1
    # #     if plot_cntr % 5 == 0:
    # #         plot_state_cov(ax_sim, P, x, color='green', fill=False, linestyle='-', linewidth=1.0)
    # # ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in filter_priv_ests], c='green', linestyle='-', linewidth=1.0)

    # # Privileged filters
    # for sen in range(num_sensors):
    #     fil = [x[sen] for x in filters_priv_ests]
    #     clr = colour_map((sen+1)/num_sensors)
    #     # clr = colours[sen % len(colours)]
    #     ax_sim.plot([x[0][0] for x in fil], [x[0][2] for x in fil], linestyle=(0, (5, 1)), linewidth=1.0, c=clr)
    #     plot_cntr = 0
    #     for x,P in fil:
    #         plot_cntr += 1
    #         if plot_cntr % 5 == 0:
    #             plot_state_cov(ax_sim, P, x, fill=False, linestyle=(0, (5, 1)), linewidth=1.0, color=clr)
    #     ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in fil], linestyle=(0, (5, 1)), linewidth=1.0, c=clr)
    
    # # Unprivileged filters
    # for sen in range(num_sensors):
    #     fil = [x[sen] for x in filters_unpriv_ests]
    #     clr = colour_map((sen+1)/num_sensors)
    #     # clr = colours[sen % len(colours)]
    #     ax_sim.plot([x[0][0] for x in fil], [x[0][2] for x in fil], linestyle='-', linewidth=1.0, c=clr)
    #     plot_cntr = 0
    #     for x,P in fil:
    #         plot_cntr += 1
    #         if plot_cntr % 5 == 0:
    #             plot_state_cov(ax_sim, P, x, fill=False, linestyle='-', linewidth=1.0, color=clr)
    #     ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in fil], linestyle='-', linewidth=1.0, c=clr)
    
    # # # Fused filter
    # # ax_sim.plot([x[0][0] for x in filter_fused], [x[0][2] for x in filter_fused], c='blue', linestyle='-', linewidth=1.0)
    # # plot_cntr = 0
    # # for x,P in filter_fused:
    # #     plot_cntr += 1
    # #     if plot_cntr % 5 == 0:
    # #         plot_state_cov(ax_sim, P, x, color='blue', fill=False, linestyle='-', linewidth=1.0)

    # plt.show()
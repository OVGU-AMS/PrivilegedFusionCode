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


def plot_single_priv_sim(sim_data):
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

    # TODO choose layout and adjust subplots (sizing)
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 4), sharex=True, sharey=True)

    # Colours
    colour_map = plt.cm.get_cmap('plasma_r')

    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []

    # Loop and make the plots
    for i,ax in enumerate(axes.flat):
        # TODO change to fit layout and paper notation
        ax.set_title(r'Estimation with $%d$ key%s $(j=%d)$' % (i+1, '' if i==0 else 's', i+1))

        # Unpriv in each plot
        u, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filter_errors_avg], linestyle='-', c='black')
        # Unpriv trace (to check the average MSE above is correct)
        ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.unpriv_filter_results], linestyle='-', c='black')
        
        # Priv only denoised at each privilege
        pd, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[i]], linestyle='--', c=colour_map((i+1)/4))
        # Priv only denoised trace (to check the average MSE above is correct)
        ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_j_ms_results[i]], linestyle='--', c=colour_map((i+1)/4))

        # Priv all at each privilege
        pa, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[i]], linestyle='-', c=colour_map((i+1)/4))
        # Priv all trace (to check the average MSE above is correct)
        ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_all_ms_results[i]], linestyle='-', c=colour_map((i+1)/4))

        unpriv_plots.append(u)
        priv_denoised_plots.append(pd)
        priv_all_plots.append(pa)

    # TODO fix legend (either per-graph or tight fitting figure legend)
    # Legend
    fig.legend((unpriv_plots[0], 
                priv_denoised_plots[0], 
                priv_all_plots[0],
                priv_denoised_plots[1],
                priv_all_plots[1],
                priv_denoised_plots[2],
                priv_all_plots[2],
                priv_all_plots[3]), 
               (r'$(0,4)$ (unprivileged)',
                r'$(4,4)$ (fully privileged)', 
                r'$(1,1)$', 
                r'$(1,4)$',
                r'$(2,2)$',
                r'$(2,4)$',
                r'$(3,3)$',
                r'$(3,4)$'), loc='upper center', ncol=4)

    # Shared axis labels
    fig.supxlabel(r'Simulation Time')   
    fig.supylabel(r'Mean Squared Error (MSE)')

    # TODO hide ticks according to set layout
    # Hide relevant axis ticks
    for a in [axes[0][0], axes[0][1]]:
        a.tick_params(bottom=False)
    for a in [axes[0][1], axes[1][1]]:
        a.tick_params(left=False)

    # Save or show picture
    if save_not_show:
        plt.savefig('pictures/privilege_differences.pdf')
    else:
        plt.show()

    return


def plot_parameter_differences(avg_sim_data, save_not_show, show_as_tex):
    init_matplotlib_params(save_not_show, show_as_tex)

    # TODO choose layout and adjust subplots (sizing)
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 4), sharex=True, sharey=True)

    # Colours
    colour_map = plt.cm.get_cmap('plasma_r')

    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []

    # Loop and make the plots
    for i,ax in enumerate(axes.flat):
        # TODO change to fit layout and paper notation
        ax.set_title(r'Estimation with $%d$ key%s $(j=%d)$' % (i+1, '' if i==0 else 's', i+1))

        # Unpriv in each plot
        u, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filter_errors_avg], linestyle='-', c='black')
        # Unpriv trace (to check the average MSE above is correct)
        ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.unpriv_filter_results], linestyle='-', c='black')
        
        # Priv only denoised at each privilege
        pd, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[i]], linestyle='--', c=colour_map((i+1)/4))
        # Priv only denoised trace (to check the average MSE above is correct)
        ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_j_ms_results[i]], linestyle='--', c=colour_map((i+1)/4))

        # Priv all at each privilege
        pa, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[i]], linestyle='-', c=colour_map((i+1)/4))
        # Priv all trace (to check the average MSE above is correct)
        ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_all_ms_results[i]], linestyle='-', c=colour_map((i+1)/4))

        unpriv_plots.append(u)
        priv_denoised_plots.append(pd)
        priv_all_plots.append(pa)

    # TODO fix legend (either per-graph or tight fitting figure legend)
    # Legend
    fig.legend((unpriv_plots[0], 
                priv_denoised_plots[0], 
                priv_all_plots[0],
                priv_denoised_plots[1],
                priv_all_plots[1],
                priv_denoised_plots[2],
                priv_all_plots[2],
                priv_all_plots[3]), 
               (r'$(0,4)$ (unprivileged)',
                r'$(4,4)$ (fully privileged)', 
                r'$(1,1)$', 
                r'$(1,4)$',
                r'$(2,2)$',
                r'$(2,4)$',
                r'$(3,3)$',
                r'$(3,4)$'), loc='upper center', ncol=4)

    # Shared axis labels
    fig.supxlabel(r'Simulation Time')   
    fig.supylabel(r'Mean Squared Error (MSE)')

    # TODO hide ticks according to set layout
    # Hide relevant axis ticks
    for a in [axes[0][0], axes[0][1]]:
        a.tick_params(bottom=False)
    for a in [axes[0][1], axes[1][1]]:
        a.tick_params(left=False)

    # Save or show picture
    if save_not_show:
        plt.savefig('pictures/privilege_differences.pdf')
    else:
        plt.show()

    return

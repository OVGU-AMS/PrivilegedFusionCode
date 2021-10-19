"""

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
 
  .d8888b.  8888888888 88888888888 888     888 8888888b.  
 d88P  Y88b 888            888     888     888 888   Y88b 
 Y88b.      888            888     888     888 888    888 
  "Y888b.   8888888        888     888     888 888   d88P 
     "Y88b. 888            888     888     888 8888888P"  
       "888 888            888     888     888 888        
 Y88b  d88P 888            888     Y88b. .d88P 888        
  "Y8888P"  8888888888     888      "Y88888P"  888        
                                                          
                                                          
                                                          
 
"""

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
        preamble = r"\usepackage{bm}"
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",   
            'font.family': 'serif',             # Use serif/main font for text elements
            'text.usetex': True,                # Use inline maths for ticks
            'pgf.rcfonts': False,               # Don't setup fonts from matplotlib rc params
            'text.latex.preamble' : preamble,   # Latex preamble when displaying
            'pgf.preamble': preamble            # Latex preamble when saving (pgf)
        })

    return

"""
 
 888    888 8888888888 888      8888888b.  8888888888 8888888b.   .d8888b.  
 888    888 888        888      888   Y88b 888        888   Y88b d88P  Y88b 
 888    888 888        888      888    888 888        888    888 Y88b.      
 8888888888 8888888    888      888   d88P 8888888    888   d88P  "Y888b.   
 888    888 888        888      8888888P"  888        8888888P"      "Y88b. 
 888    888 888        888      888        888        888 T88b         "888 
 888    888 888        888      888        888        888  T88b  Y88b  d88P 
 888    888 8888888888 88888888 888        8888888888 888   T88b  "Y8888P"  
                                                                            
                                                                            
                                                                            
 
"""

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
    # Centre
    state_2d = np.array([state[0], state[2]])

    # 2d covariance
    state_cov_2d = np.array([[state_covariance[0][0], state_covariance[2][0]],
                             [state_covariance[0][2], state_covariance[2][2]]])
    
    # Create and plot ellipse
    ellipse = get_cov_ellipse(state_cov_2d, state_2d, 2, **kwargs)
    return plotter.add_artist(ellipse)

"""
 
  .d8888b. 8888888 888b    888  .d8888b.  888      8888888888       .d8888b. 8888888 888b     d888 
 d88P  Y88b  888   8888b   888 d88P  Y88b 888      888             d88P  Y88b  888   8888b   d8888 
 Y88b.       888   88888b  888 888    888 888      888             Y88b.       888   88888b.d88888 
  "Y888b.    888   888Y88b 888 888        888      8888888          "Y888b.    888   888Y88888P888 
     "Y88b.  888   888 Y88b888 888  88888 888      888                 "Y88b.  888   888 Y888P 888 
       "888  888   888  Y88888 888    888 888      888                   "888  888   888  Y8P  888 
 Y88b  d88P  888   888   Y8888 Y88b  d88P 888      888             Y88b  d88P  888   888   "   888 
  "Y8888P" 8888888 888    Y888  "Y8888P88 88888888 8888888888       "Y8888P" 8888888 888       888 
                                                                                                   
                                                                                                   
                                                                                                   
 
"""

def plot_single_priv_sim(sim_data):
    fig = plt.figure()
    ax_sim = fig.add_subplot(121)
    ax_trace = fig.add_subplot(122)

    # Useful for colour consistency
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

def plot_single_param_sim(sim_data):
    # Single sim will still have 4 plots due to the varying of simulation parameters
    fig, axes = plt.subplots(4, 2)

    # Useful for colour consistency
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colour_map = plt.cm.get_cmap('plasma_r')

    # Loop parameters and treat each combination as its own single plot
    ind = 0
    for Y in sim_data.Ys:
        for Z in sim_data.Zs:
            # Params to identify where to plot
            ax_sim = axes.flat[ind]
            ax_trace = axes.flat[ind+1]
            clr = colour_map(((ind+1)/2)/(len(sim_data.Ys)+len(sim_data.Zs)))

            # Label the plots so it's clear which is which
            ax_trace.set_title(r'$Y=%.2lf$, $Z=%.2lf$' % (Y, Z))
            plt.subplots_adjust(hspace=0.5)

            # Ground truth
            ax_sim.plot([x[0] for x in sim_data.gt], [x[2] for x in sim_data.gt], c='lightgray', linestyle='-', linewidth=1.0)

            # Nosied measurements
            for s in range(sim_data.num_sensors):
                ax_sim.scatter([x[0] for x in sim_data.zs[Y][Z][s]], [x[1] for x in sim_data.zs[Y][Z][s]], c=colours[s], marker='.', s=1.0)

            # Unpriv
            ax_sim.plot([x[0][0] for x in sim_data.unpriv_filters_results[Y][Z]], [x[0][2] for x in sim_data.unpriv_filters_results[Y][Z]], c='black', linestyle='-', linewidth=1.0)
            for i in range(sim_data.sim_len):
                if i % 10 == 0:
                    plot_state_cov(ax_sim, sim_data.unpriv_filters_results[Y][Z][i][1], sim_data.unpriv_filters_results[Y][Z][i][0], color='black', fill=False, linestyle='-', linewidth=1.0)
            ax_trace.plot([x for x in range(sim_data.sim_len)], [np.trace(x[1]) for x in sim_data.unpriv_filters_results[Y][Z]], c='black', linestyle='-', linewidth=1.0)

            # Priv only denoised
            ax_sim.plot([x[0][0] for x in sim_data.priv_filters_j_ms_results[Y][Z]], [x[0][2] for x in sim_data.priv_filters_j_ms_results[Y][Z]], c=clr, linestyle='-', linewidth=1.0)
            for i in range(sim_data.sim_len):
                plot_state_cov(ax_sim, sim_data.priv_filters_j_ms_results[Y][Z][i][1], sim_data.priv_filters_j_ms_results[Y][Z][i][0], color=clr, fill=False, linestyle='-', linewidth=1.0)
            ax_trace.plot([x for x in range(sim_data.sim_len)], [np.trace(x[1]) for x in sim_data.priv_filters_j_ms_results[Y][Z]], c=clr, linestyle='-', linewidth=1.0)

            # Priv all
            ax_sim.plot([x[0][0] for x in sim_data.priv_filters_all_ms_results[Y][Z]], [x[0][2] for x in sim_data.priv_filters_all_ms_results[Y][Z]], c=clr, linestyle='--', linewidth=1.0)
            for i in range(sim_data.sim_len):
                plot_state_cov(ax_sim, sim_data.priv_filters_all_ms_results[Y][Z][i][1], sim_data.priv_filters_all_ms_results[Y][Z][i][0], color=clr, fill=False, linestyle='--', linewidth=1.0)
            ax_trace.plot([x for x in range(sim_data.sim_len)], [np.trace(x[1]) for x in sim_data.priv_filters_all_ms_results[Y][Z]], c=clr, linestyle='--', linewidth=1.0)
            
            ind+=2

    plt.show()
    return

"""
 
 888b     d888        d8888 8888888 888b    888      8888888b.  888      .d88888b. 88888888888 .d8888b.  
 8888b   d8888       d88888   888   8888b   888      888   Y88b 888     d88P" "Y88b    888    d88P  Y88b 
 88888b.d88888      d88P888   888   88888b  888      888    888 888     888     888    888    Y88b.      
 888Y88888P888     d88P 888   888   888Y88b 888      888   d88P 888     888     888    888     "Y888b.   
 888 Y888P 888    d88P  888   888   888 Y88b888      8888888P"  888     888     888    888        "Y88b. 
 888  Y8P  888   d88P   888   888   888  Y88888      888        888     888     888    888          "888 
 888   "   888  d8888888888   888   888   Y8888      888        888     Y88b. .d88P    888    Y88b  d88P 
 888       888 d88P     888 8888888 888    Y888      888        88888888 "Y88888P"     888     "Y8888P"  
                                                                                                         
                                                                                                         
                                                                                                         
 
"""

def plot_privilege_differences(avg_sim_data, save_not_show, show_as_tex):
    init_matplotlib_params(save_not_show, show_as_tex)

    # TODO choose layout and adjust subplots (sizing)
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 4), sharex=True, sharey=True)

    # Colours
    colour_map = plt.cm.get_cmap('plasma_r')

    # Loop and make the plots
    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []
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

    # TODO change legend to fit layout (either per-graph or one tight fitting one)
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

    # TODO hide ticks according to layout
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

    # Loop and make the plots
    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []
    ind = 0
    for Y in avg_sim_data.Ys:
        for Z in avg_sim_data.Zs:
            ax = axes.flat[ind]
            # TODO change to fit layout and paper notation
            ax.set_title(r'$Y=%.2lf\bm{I}$, $Z=%.2lf\bm{I}$' % (Y, Z))

            # Colour is per-plot
            clr = colour_map((ind+1)/4)

            # Unpriv in each plot
            u, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filters_errors_avg[Y][Z]], linestyle='-.', c=clr)
            # Unpriv trace (to check the average MSE above is correct)
            ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.unpriv_filters_results[Y][Z]], linestyle='-.', c=clr)
            
            # Priv only denoised at each privilege
            pd, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[Y][Z]], linestyle='--', c=clr)
            # Priv only denoised trace (to check the average MSE above is correct)
            ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_j_ms_results[Y][Z]], linestyle='--', c=clr)

            # Priv all at each privilege
            pa, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[Y][Z]], linestyle='-', c=clr)
            # Priv all trace (to check the average MSE above is correct)
            ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_all_ms_results[Y][Z]], linestyle='-', c=clr)

            unpriv_plots.append(u)
            priv_denoised_plots.append(pd)
            priv_all_plots.append(pa)

            ind+=1

    # TODO change legend to fit layout (either per-graph or one tight fitting one)
    # Legend
    fig.legend((unpriv_plots[0], 
                priv_denoised_plots[0], 
                priv_all_plots[0]), 
               (r'$(0,4)$ (unprivileged)',
                r'$(2,2)$',
                r'$(2,4)$'), loc='upper center', ncol=3)

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
        plt.savefig('pictures/parameter_differences.pdf')
    else:
        plt.show()

    return


def plot_parameter_scan(sim_data, save_not_show, show_as_tex):
    init_matplotlib_params(save_not_show, show_as_tex)

    # TODO choose layout and adjust subplots (sizing)
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 4), sharex=True, sharey=True)

    # Colours
    colour_map = plt.cm.get_cmap('plasma_r')

    # Loop and make the plots
    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []
    ind = 0
    for priv in sim_data.privileges:
        for fixed in ["Y_fixed", "Z_fixed"]:

            ax = axes.flat[ind]
            # TODO change to fit layout and paper notation
            ax.set_title(r'$%d$ keys, Fixed $%s$' % (priv, 'Y' if ind%2==0 else 'Z'))

            if fixed == 'Y_fixed':
                x = sim_data.Ys
            else:
                x = sim_data.Zs

            # Unpriv in each plot
            u, = ax.plot(x, [t for t in sim_data.unpriv_filters_traces[priv][fixed]], linestyle='-.', c='black')
            
            # Priv only denoised at each privilege
            pd, = ax.plot(x, [t for t in sim_data.priv_filters_j_ms_traces[priv][fixed]], linestyle='--', c=colour_map(1/4))

            # Priv all at each privilege
            pa, = ax.plot(x, [t for t in sim_data.priv_filters_all_ms_traces[priv][fixed]], linestyle='-', c=colour_map(3/4))

            unpriv_plots.append(u)
            priv_denoised_plots.append(pd)
            priv_all_plots.append(pa)

            ind+=1

    # TODO change legend to fit layout (either per-graph or one tight fitting one)
    # Legend
    fig.legend((unpriv_plots[0], 
                priv_denoised_plots[0], 
                priv_all_plots[0]), 
               (r'$(0,4)$ (unprivileged)',
                r'$(2,2)$',
                r'$(2,4)$'), loc='upper center', ncol=3)

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
        plt.savefig('pictures/parameter_differences.pdf')
    else:
        plt.show()

    return
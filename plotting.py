"""

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)

def plot_state_cov(plotter, state_covariance, state, **kwargs):
    state_2d = np.array([state[0], state[2]])
    state_cov_2d = np.array([[state_covariance[0][0], state_covariance[2][0]],
                             [state_covariance[0][2], state_covariance[2][2]]])
    ellipse = get_cov_ellipse(state_cov_2d, state_2d, 2, **kwargs)
    return plotter.add_artist(ellipse)





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
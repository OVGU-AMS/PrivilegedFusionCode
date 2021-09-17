
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import estimation as est


def add_correlated_noises():
    return

def fuse_correlated_covariances(P1, P2, P12):
    P = P1 - (P1 - P12)@np.linalg.inv(P1 + P2 - P12 - P12.T)@(P1 - P12.T)
    return P

def fuse_correlated_estimates(x1, x2, P1, P2, P12):
    x = x1 + (P1 - P12)@np.linalg.inv(P1 + P2 - P12 - P12.T)@(x2 - x1)
    return x

def fuse_correlated(x1, x2, P1, P2, P12):
    x = fuse_correlated_estimates(x1, x2, P1, P2, P12)
    P = fuse_correlated_covariances(P1, P2, P12)
    return x, P

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


def main():
    # State dimension
    n = 4

    # Measurement dimension
    m = 2

    # Process model (q = noise strength, t = timestep)
    q = 0.01
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])

    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])

    # Measurement models
    H = np.array([[1, 0, 0, 0], 
                   [0, 0, 1, 0]])

    R = np.array([[5, 2], 
                  [2, 5]])

    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0]])
    
    # Ground truth init
    gt_init_state = np.array([0, 1, 0, 1])

    # Added noise covariance
    Z = np.array([[15, 5],
                  [5, 15]])

    # Correlation matrix of added measurement noises
    # added_noise_cov_corr = np.array([[35, 2, 15, 5],
    #                                  [2, 35, 5, 15],
    #                                  [15, 5, 35, 2],
    #                                  [5, 15, 2, 35]])
    
    #fused_added_noise_cov = fuse_correlated_covariances(added_noise_cov_corr[:2,:2], added_noise_cov_corr[:2,:2], added_noise_cov_corr[:2,2:])

    ground_truth = est.GroundTruth(F, Q, gt_init_state)
    sensors = []
    num_sensors = 10
    for _ in range(num_sensors):
        sensors.append(est.SensorPure(n, m, H, R))

    #filter_priv = est.KFilter(n, m, F, Q, H, R, init_state, init_cov)
    filters_priv_varying_sensors = []
    for sens in range(1, num_sensors+1):
        stacked_H = np.block([[H] for _ in range(sens)])
        stacked_R = np.block([[R if c==r else np.zeros((2,2)) for c in range(sens)] for r in range(sens)])
        f = est.KFilter(n, sens*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        filters_priv_varying_sensors.append(f)
    
    filters_unpriv_varying_sensors = []
    for sens in range(1, num_sensors+1):
        stacked_H = np.block([[H] for _ in range(sens)])
        stacked_R = np.block([[R+Z if c==r else Z for c in range(sens)] for r in range(sens)])
        f = est.KFilter(n, sens*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        filters_unpriv_varying_sensors.append(f)
    #est_filter_unprivileged_zs = est.KFilter(n, 2*m, F, Q, H2, R2+added_noise_cov_corr, init_state, init_cov)
    #est_filter_fused_unprivileged_zs = est.KFilter(n, m, F, Q, H, R+fused_added_noise_cov, init_state, init_cov)

    # For plotting
    fig = plt.figure()
    ax_sim = fig.add_subplot(121)
    ax_trace = fig.add_subplot(122)
    gts = []
    all_zs = []
    #filter_priv_ests = []
    filters_priv_ests = []
    filters_unpriv_ests = []
    #filter_fused = []
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colour_map = plt.cm.get_cmap('plasma_r')

    sim_steps = 100
    for _ in range(sim_steps):
        gt = ground_truth.update()
        gts.append(gt)

        # Make all measurements
        zs = []
        for sen in range(num_sensors):
            zs.append(sensors[sen].measure(gt))
        all_zs.append(zs)

        # # Privileged filter on true measurements
        # filter_priv.predict()
        # for sen in range(num_sensors):
        #     filter_priv.update(zs[sen])
        # filter_priv_ests.append((filter_priv.x, filter_priv.P))

        # Privileged estimators for varying number of sensors
        priv_ests = []
        for sen in range(num_sensors):
            filters_priv_varying_sensors[sen].predict()
            filters_priv_varying_sensors[sen].update(np.block([z for z in zs[:sen+1]]))
            priv_ests.append((filters_priv_varying_sensors[sen].x, filters_priv_varying_sensors[sen].P))
        filters_priv_ests.append(priv_ests)

        # Add same Gaussian noise to all measurements
        noise = np.random.multivariate_normal(np.zeros(2), Z)
        noised_zs = []
        for sen in range(num_sensors):
            noised_zs.append(zs[sen] + noise)
        
        # Unprivileged estimators for varying number of sensors
        unpriv_ests = []
        for sen in range(num_sensors):
            filters_unpriv_varying_sensors[sen].predict()
            filters_unpriv_varying_sensors[sen].update(np.block([z for z in noised_zs[:sen+1]]))
            unpriv_ests.append((filters_unpriv_varying_sensors[sen].x, filters_unpriv_varying_sensors[sen].P))
        filters_unpriv_ests.append(unpriv_ests)

        # z1 = z1 + corr_noise[:2]
        # z2 = z2 + corr_noise[2:]
        # z_fused = fuse_correlated_estimates(z1, z2, (R2+added_noise_cov_corr)[:2,:2], (R2+added_noise_cov_corr)[:2,:2], (R2+added_noise_cov_corr)[:2,2:])
        # est_filter_fused_unprivileged_zs.predict()
        # e = est_filter_fused_unprivileged_zs.update(z_fused)
        # filter_fused.append(e)
    
    # Plotting

    # Ground truth
    ax_sim.plot([x[0] for x in gts], [x[2] for x in gts], c='gray', linestyle='-', linewidth=1.0)

    # Sensors
    for sen in range(num_sensors):
        rel_zs = [z[sen] for z in all_zs]
        ax_sim.scatter([x[0] for x in rel_zs], [x[1] for x in rel_zs], c='orange', marker='.', s=1.0)

    # # Privileged filter
    # ax_sim.plot([x[0][0] for x in filter_priv_ests], [x[0][2] for x in filter_priv_ests], c='green', linestyle='-', linewidth=1.0)
    # plot_cntr = 0
    # for x,P in filter_priv_ests:
    #     plot_cntr += 1
    #     if plot_cntr % 5 == 0:
    #         plot_state_cov(ax_sim, P, x, color='green', fill=False, linestyle='-', linewidth=1.0)
    # ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in filter_priv_ests], c='green', linestyle='-', linewidth=1.0)

    # Privileged filters
    for sen in range(num_sensors):
        fil = [x[sen] for x in filters_priv_ests]
        clr = colour_map((sen+1)/num_sensors)
        # clr = colours[sen % len(colours)]
        ax_sim.plot([x[0][0] for x in fil], [x[0][2] for x in fil], linestyle=(0, (5, 1)), linewidth=1.0, c=clr)
        plot_cntr = 0
        for x,P in fil:
            plot_cntr += 1
            if plot_cntr % 5 == 0:
                plot_state_cov(ax_sim, P, x, fill=False, linestyle=(0, (5, 1)), linewidth=1.0, color=clr)
        ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in fil], linestyle=(0, (5, 1)), linewidth=1.0, c=clr)
    
    # Unprivileged filters
    for sen in range(num_sensors):
        fil = [x[sen] for x in filters_unpriv_ests]
        clr = colour_map((sen+1)/num_sensors)
        # clr = colours[sen % len(colours)]
        ax_sim.plot([x[0][0] for x in fil], [x[0][2] for x in fil], linestyle='-', linewidth=1.0, c=clr)
        plot_cntr = 0
        for x,P in fil:
            plot_cntr += 1
            if plot_cntr % 5 == 0:
                plot_state_cov(ax_sim, P, x, fill=False, linestyle='-', linewidth=1.0, color=clr)
        ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in fil], linestyle='-', linewidth=1.0, c=clr)
    
    # # Fused filter
    # ax_sim.plot([x[0][0] for x in filter_fused], [x[0][2] for x in filter_fused], c='blue', linestyle='-', linewidth=1.0)
    # plot_cntr = 0
    # for x,P in filter_fused:
    #     plot_cntr += 1
    #     if plot_cntr % 5 == 0:
    #         plot_state_cov(ax_sim, P, x, color='blue', fill=False, linestyle='-', linewidth=1.0)

    plt.show()
        
    return


if __name__ == '__main__':
    main()








# for Pc in np.arange(0,1,0.1):
#     P = 0.5
#     P2 = 0.5
#     P_list = []
#     for i in range(50):
#         P_list.append(P)
#         P = P - (P - Pc)*((P + P2 - Pc - Pc)**-1)*(P - Pc)
#     plt.plot(P_list, label="cP: %1.1lf" % Pc)
# plt.legend()
# plt.show()
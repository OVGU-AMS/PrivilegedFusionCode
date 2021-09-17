
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import estimation as est

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
    state_cov_2d = np.array([[state_covariance[0][0], state_covariance[0][2]],
                             [state_covariance[2][0], state_covariance[2][2]]])
    ellipse = get_cov_ellipse(state_cov_2d, state_2d, 2, **kwargs)
    return plotter.add_artist(ellipse)

class KFilterTimeCorr(est.FilterAbs):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, init_measurement):
        self.n = n
        self.m = m
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = init_state
        self.P = init_cov
        self.P_pred = init_cov
        self.prev_measurement = init_measurement
        return
    
    def predict(self):
        K = self.P_pred@self.H.T@np.linalg.inv(self.H@self.P_pred@self.H.T + self.R)
        M = self.Q@self.H.T
        C = M@np.linalg.inv(self.H@self.P_pred@self.H.T + self.R)
        self.x = self.F@self.x + C@(self.prev_measurement - self.H@self.x)
        self.P = self.F@self.P@self.F.T + self.Q - C@M.T - self.F@K@M - M.T@K.T@self.F.T
        self.P_pred = self.P
        return self.x, self.P
    
    def update(self, measurement):
        K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T + self.R)
        self.x = self.x + self.P@(measurement - self.H@self.x)
        self.P = (np.eye(self.n) - K@self.H)@self.P@(np.eye(self.n) - K@self.H).T + K@self.R@K.T
        self.prev_measurement = measurement
        return self.x, self.P


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
    

    fig = plt.figure()
    ax_sim = fig.add_subplot(121)
    ax_trace = fig.add_subplot(122)
    
    # For computing real errors over many simulations
    all_filter_priv_stacked_errors = []
    all_filter_unpriv_uncorr_errors = []
    all_filter_unpriv_corr_errors = []
    all_filter_unpriv_corr_mdiff_errors = []

    num_sims = 30
    for s in range(num_sims):
        if s%10 == 0:
            print('Sim progress:', s, '/', num_sims)


        ground_truth = est.GroundTruth(F, Q, gt_init_state)
        sensors = []
        num_sensors = 2
        num_stacked_states = 2
        for _ in range(num_sensors):
            sensors.append(est.SensorPure(n, m, H, R))


        stacked_H = np.block([[H] for _ in range(num_sensors)])
        stacked_R = np.block([[R if c==r else np.zeros((2,2)) for c in range(num_sensors)] for r in range(num_sensors)])
        filter_priv_stacked = est.KFilter(n, num_sensors*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        #filter_priv_stacked = est.KFilter(n, m, F, Q, H, R, init_state, init_cov)
        # filters_priv_varying_sensors = []
        # for sens in range(1, num_sensors+1):
        #     stacked_H = np.block([[H] for _ in range(sens)])
        #     stacked_R = np.block([[R if c==r else np.zeros((2,2)) for c in range(sens)] for r in range(sens)])
        #     f = est.KFilter(n, sens*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        #     filters_priv_varying_sensors.append(f)
        
        stacked_H = np.block([[H] for _ in range(num_sensors)])
        stacked_R = np.block([[R+Z if c==r else np.zeros((2,2)) for c in range(num_sensors)] for r in range(num_sensors)])
        filter_unpriv_uncorr = est.KFilter(n, num_sensors*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        #filter_unpriv_uncorr = est.KFilter(n, m, F, Q, H, R+Z, init_state, init_cov)
        # filters_unpriv_varying_sensors = []
        # for sens in range(1, num_sensors+1):
        #     stacked_H = np.block([[H] for _ in range(sens)])
        #     stacked_R = np.block([[R+Z if c==r else Z for c in range(sens)] for r in range(sens)])
        #     f = est.KFilter(n, sens*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        #     filters_unpriv_varying_sensors.append(f)





        # TODO Given changable offsets stacked_r would need to change
        # TODO replace dimension numbers with n and m

        stacked_H = np.block([[H if c==r else np.zeros((2,4)) for c in range(num_stacked_states)] for r in range(num_sensors)])
        stacked_R = np.block([[R+Z if c==r else Z for c in range(num_sensors)] for r in range(num_sensors)])
        stacked_init_state = np.block([init_state if i==0 else np.zeros(4) for i in range(num_stacked_states)])
        stacked_init_cov = np.block([[init_cov if c==0 and r==0 else np.zeros((4,4)) for c in range(num_stacked_states)] for r in range(num_stacked_states)])
        stacked_Q = np.block([[Q if c==0 and r==0 else np.zeros((4,4)) for c in range(num_stacked_states)] for r in range(num_stacked_states)])
        stacked_F = np.block([[F if c==0 and r==0 else np.eye(4,4) if c==r-1 else np.zeros((4,4)) for c in range(num_stacked_states)] for r in range(num_stacked_states)])
        filter_unpriv_corr = est.KFilter(num_stacked_states*n, num_sensors*m, stacked_F, stacked_Q, stacked_H, stacked_R, stacked_init_state, stacked_init_cov)



        phi = np.block([[np.eye(2) if (r==0 and c==3) or (r==2 and c==3) else np.zeros((2,2)) for c in range(4)] for r in range(4)])
        stacked_H = np.block([[H] if i in [0,1] else [np.zeros((2,4))] for i in range(4)])
        H_dash = stacked_H@F - phi@stacked_H
        tmp = [[np.zeros((2,2)) for c in range(4)] for r in range(4)]
        tmp[0][0] = R
        tmp[1][1] = R+Z
        tmp[1][3] = Z
        tmp[3][1] = Z
        tmp[3][3] = Z
        stacked_R = np.block(tmp)
        R_dash = stacked_H@Q@stacked_H.T + stacked_R
        init_measurement = np.zeros(4*m)
        filter_unpriv_corr_mdiff = KFilterTimeCorr(n, 4*m, F, Q, H_dash, R_dash, init_state, init_cov, init_measurement)




        
        gts = []
        all_zs = []
        all_noised_zs = []
        filter_priv_stacked_ests = []
        filter_unpriv_uncorr_ests = []
        filter_unpriv_corr_ests = []
        filter_unpriv_corr_mdiff_ests = []
        colour_map = plt.cm.get_cmap('plasma_r')

        sim_steps = 300
        noise_stream = [np.random.multivariate_normal(np.zeros(2), Z) for _ in range(sim_steps+1)] # TODO +1 should be the longest time offset
        for k in range(sim_steps):
            gt = ground_truth.update()
            gts.append(gt)

            # Make all measurements
            zs = []
            for sen in range(num_sensors):
                zs.append(sensors[sen].measure(gt))
            all_zs.append(zs)

            # Privileged filter
            filter_priv_stacked.predict()
            x,P = filter_priv_stacked.update(np.block([z for z in zs]))
            filter_priv_stacked_ests.append((x, P))

            # Adding nosies
            noised_zs = []
            for sen in range(num_sensors):
                noised_zs.append(zs[sen] + noise_stream[k+sen])
            all_noised_zs.append(noised_zs)
            
            # Unprivileged filter assuming no correlation
            filter_unpriv_uncorr.predict()
            x,P = filter_unpriv_uncorr.update(np.block([z for z in noised_zs]))
            filter_unpriv_uncorr_ests.append((x, P))

            # Unprivileged filter assuming correlation
            filter_unpriv_corr.predict()
            if k>0:
                time_corr_zs = np.block([noised_zs[0], all_noised_zs[-2][-1]])
            else:
                time_corr_zs = np.block([noised_zs[0], noise_stream[0]])
            x,P = filter_unpriv_corr.update(time_corr_zs)
            filter_unpriv_corr_ests.append((x, P))

            # Unprivileged assuming correlation - measurement difference filter
            if k>0:
                measurement_diff = np.block([z for z in zs]+noise_stream[k:k+2]) - phi@filter_unpriv_corr_mdiff.prev_measurement
                filter_unpriv_corr_mdiff.update(measurement_diff)
            x,P = filter_unpriv_corr_mdiff.predict()
            filter_unpriv_corr_mdiff_ests.append((x, P))
            
        
        all_filter_priv_stacked_errors.append([np.linalg.norm(gt-e[0]) for gt,e in zip(gts, filter_priv_stacked_ests)])
        all_filter_unpriv_uncorr_errors.append([np.linalg.norm(gt-e[0]) for gt,e in zip(gts, filter_unpriv_uncorr_ests)])
        #all_filter_unpriv_uncorr_errors.append([np.linalg.norm(gt-e[0])+np.linalg.norm(gt_prev-e_prev[0]) for gt_prev,gt,e_prev,e in zip([np.zeros(4)] + gts[:-1], gts, [(np.zeros(4), None)]+filter_unpriv_uncorr_ests[:-1], filter_unpriv_uncorr_ests)])
        #all_filter_unpriv_corr_errors.append([np.linalg.norm(np.block([gt, gt_prev])-e[0]) for gt_prev,gt,e in zip([np.zeros(4)] + gts[:-1], gts, filter_unpriv_corr_ests)])
        all_filter_unpriv_corr_errors.append([np.linalg.norm(gt-e[0][4:]) for gt,e in zip(gts, filter_unpriv_corr_ests[1:] + [filter_unpriv_corr_ests[-1]])])
        all_filter_unpriv_corr_mdiff_errors.append([np.linalg.norm(gt-e[0]) for gt,e in zip(gts, filter_unpriv_corr_mdiff_ests)])

    # Plotting

    # Ground truth
    ax_sim.plot([x[0] for x in gts], [x[2] for x in gts], c='gray', linestyle='-', linewidth=1.0)

    # Sensors
    # for sen in range(num_sensors):
    #     rel_zs = [z[sen] for z in all_zs]
    #     ax_sim.scatter([x[0] for x in rel_zs], [x[1] for x in rel_zs], c='orange', marker='.', s=1.0)

    # Privileged filter
    clr = 'green'
    ax_sim.plot([x[0][0] for x in filter_priv_stacked_ests], [x[0][2] for x in filter_priv_stacked_ests], linestyle='-', linewidth=1.0, c=clr)
    plot_cntr = 0
    for x,P in filter_priv_stacked_ests:
        plot_cntr += 1
        if plot_cntr % 5 == 0:
            plot_state_cov(ax_sim, P, x, fill=False, linestyle='-', linewidth=1.0, color=clr)
    #ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in filter_priv_stacked_ests], linestyle='-', linewidth=1.0, c=clr)
    #priv_stacked_sim_rmses = [[np.linalg.norm(e) for e in sim_es] for sim_es in all_filter_priv_stacked_errors]
    priv_stacked_rmses = [np.mean([es[i] for es in all_filter_priv_stacked_errors]) for i in range(sim_steps)]
    priv_stacked_rmse_stds = [np.std([es[i] for es in all_filter_priv_stacked_errors]) for i in range(sim_steps)]

    ax_trace.fill_between([i for i in range(sim_steps)], 
                          [m-sd for m,sd in zip(priv_stacked_rmses, priv_stacked_rmse_stds)], 
                          [m+sd for m,sd in zip(priv_stacked_rmses, priv_stacked_rmse_stds)],
                          linewidth=0, color=clr, alpha=0.2)
    ax_trace.plot([i for i in range(sim_steps)], priv_stacked_rmses, linestyle='-', linewidth=1.0, c=clr)
    
    # Unprivileged filters
    clr = 'red'
    ax_sim.plot([x[0][0] for x in filter_unpriv_uncorr_ests], [x[0][2] for x in filter_unpriv_uncorr_ests], linestyle='-', linewidth=1.0, c=clr)
    plot_cntr = 0
    for x,P in filter_unpriv_uncorr_ests:
        plot_cntr += 1
        if plot_cntr % 5 == 0:
            plot_state_cov(ax_sim, P, x, fill=False, linestyle='-', linewidth=1.0, color=clr)
    #ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in filter_unpriv_uncorr_ests], linestyle='-', linewidth=1.0, c=clr)
    #unpriv_uncorr_sim_rmses = [[np.linalg.norm(e) for e in sim_es] for sim_es in all_filter_unpriv_uncorr_errors]
    unpriv_uncorr_rmses = [np.mean([es[i] for es in all_filter_unpriv_uncorr_errors]) for i in range(sim_steps)]
    unpriv_uncorr_rmse_stds = [np.std([es[i] for es in all_filter_unpriv_uncorr_errors]) for i in range(sim_steps)]

    ax_trace.fill_between([i for i in range(sim_steps)], 
                          [m-sd for m,sd in zip(unpriv_uncorr_rmses, unpriv_uncorr_rmse_stds)], 
                          [m+sd for m,sd in zip(unpriv_uncorr_rmses, unpriv_uncorr_rmse_stds)],
                          linewidth=0, color=clr, alpha=0.2)
    ax_trace.plot([i for i in range(sim_steps)], unpriv_uncorr_rmses, linestyle='-', linewidth=1.0, c=clr)

    clr = 'orange'
    ax_sim.plot([x[0][0] for x in filter_unpriv_corr_ests], [x[0][2] for x in filter_unpriv_corr_ests], linestyle='-', linewidth=1.0, c=clr)
    plot_cntr = 0
    for x,P in filter_unpriv_corr_ests:
        plot_cntr += 1
        if plot_cntr % 5 == 0:
            plot_state_cov(ax_sim, P, x, fill=False, linestyle='-', linewidth=1.0, color=clr)
    #ax_trace.plot([i for i in range(sim_steps)], [np.trace(x[1]) for x in filter_unpriv_corr_ests], linestyle='-', linewidth=1.0, c=clr)
    #unpriv_corr_sim_rmses = [[np.linalg.norm(e) for e in sim_es] for sim_es in all_filter_unpriv_corr_errors]
    unpriv_corr_rmses = [np.mean([es[i] for es in all_filter_unpriv_corr_errors]) for i in range(sim_steps)]
    unpriv_corr_rmse_stds = [np.std([es[i] for es in all_filter_unpriv_corr_errors]) for i in range(sim_steps)]

    ax_trace.fill_between([i for i in range(sim_steps)], 
                          [m-sd for m,sd in zip(unpriv_corr_rmses, unpriv_corr_rmse_stds)], 
                          [m+sd for m,sd in zip(unpriv_corr_rmses, unpriv_corr_rmse_stds)],
                          linewidth=0, color=clr, alpha=0.2)
    ax_trace.plot([i for i in range(sim_steps)], unpriv_corr_rmses, linestyle='-', linewidth=1.0, c=clr)


    # Unpriv measurement difference

    clr = 'blue'
    ax_sim.plot([x[0][0] for x in filter_unpriv_corr_mdiff_ests], [x[0][2] for x in filter_unpriv_corr_mdiff_ests], linestyle='-', linewidth=1.0, c=clr)
    plot_cntr = 0
    for x,P in filter_unpriv_corr_mdiff_ests:
        plot_cntr += 1
        if plot_cntr % 5 == 0:
            plot_state_cov(ax_sim, P, x, fill=False, linestyle='-', linewidth=1.0, color=clr)
    unpriv_corr_mdiff_rmses = [np.mean([es[i] for es in all_filter_unpriv_corr_mdiff_errors]) for i in range(sim_steps)]
    unpriv_corr_mdiff_rmse_stds = [np.std([es[i] for es in all_filter_unpriv_corr_mdiff_errors]) for i in range(sim_steps)]

    ax_trace.fill_between([i for i in range(sim_steps)], 
                          [m-sd for m,sd in zip(unpriv_corr_mdiff_rmses, unpriv_corr_mdiff_rmse_stds)], 
                          [m+sd for m,sd in zip(unpriv_corr_mdiff_rmses, unpriv_corr_mdiff_rmse_stds)],
                          linewidth=0, color=clr, alpha=0.2)
    ax_trace.plot([i for i in range(sim_steps)], unpriv_corr_mdiff_rmses, linestyle='-', linewidth=1.0, c=clr)

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
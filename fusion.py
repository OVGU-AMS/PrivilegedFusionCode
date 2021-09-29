
import numpy as np
import plotting as pltng
import estimation as estmtn
import key_stream as kystrm

class SimData:
    def __init__(self, ident):
        self.ident = ident
        self.gt = []
        self.zs = []
        self.unpriv_filter_results = []
        self.priv_filters_j_ms_results = {}
        self.priv_filters_all_ms_results = {}
    
    @staticmethod
    def average_sims(sim_list):
        return


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

    # Number of present sensors
    num_sensors = 4

    # Pseudorandom correleated and uncorrelated covariances
    Z = 10*np.eye(2)
    Y = 8*np.eye(2)

    # Synced cryptographically random number generators
    generator_pairs = [kystrm.SharedKeyStreamFactory.make_shared_key_streams(2) for _ in range(num_sensors)]

    # Creating simulation objects
    ground_truth = estmtn.GroundTruth(F, Q, gt_init_state)
    sensors = []
    for _ in range(num_sensors):
        sensors.append(estmtn.SensorPure(n, m, H, R))

    unpriv_filter = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, [], 0, True)
    priv_filters_j_ms = []
    priv_filters_all_ms = []
    for j in range(num_sensors):
        gens = [g[0] for g in generator_pairs[:j+1]]
        priv_filters_j_ms.append(estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, j+1, False))
        priv_filters_all_ms.append(estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, j+1, True))

    # #filter_priv = est.KFilter(n, m, F, Q, H, R, init_state, init_cov)
    # filters_priv_varying_sensors = []
    # for sens in range(1, num_sensors+1):
    #     stacked_H = np.block([[H] for _ in range(sens)])
    #     stacked_R = np.block([[R if c==r else np.zeros((2,2)) for c in range(sens)] for r in range(sens)])
    #     f = estmtn.KFilter(n, sens*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
    #     filters_priv_varying_sensors.append(f)
    
    # filters_unpriv_varying_sensors = []
    # for sens in range(1, num_sensors+1):
    #     stacked_H = np.block([[H] for _ in range(sens)])
    #     stacked_R = np.block([[R+Z if c==r else Z for c in range(sens)] for r in range(sens)])
    #     f = estmtn.KFilter(n, sens*m, F, Q, stacked_H, stacked_R, init_state, init_cov)
    #     filters_unpriv_varying_sensors.append(f)
    # #est_filter_unprivileged_zs = est.KFilter(n, 2*m, F, Q, H2, R2+added_noise_cov_corr, init_state, init_cov)
    # #est_filter_fused_unprivileged_zs = est.KFilter(n, m, F, Q, H, R+fused_added_noise_cov, init_state, init_cov)

    # Simulation
    sim_runs = 1000
    sim_steps = 100
    sims = []
    for s in range(sim_runs):
        sim = SimData(s)
        sims.append(sim)
        for _ in range(sim_steps):
            
            # Update ground truth
            gt = ground_truth.update()
            sim.gt.append(gt)

            # Generate correlated noise
            

            # Make all measurements and add pseudorandom noises
            zs = []
            for sen in range(num_sensors):
                zs.append(sensors[sen].measure(gt))
                # TODO add noises here
            sim.zs.append(zs)



            # # Privileged filter on true measurements
            # filter_priv.predict()
            # for sen in range(num_sensors):
            #     filter_priv.update(zs[sen])
            # filter_priv_ests.append((filter_priv.x, filter_priv.P))

            # # Privileged estimators for varying number of sensors
            # priv_ests = []
            # for sen in range(num_sensors):
            #     filters_priv_varying_sensors[sen].predict()
            #     filters_priv_varying_sensors[sen].update(np.block([z for z in zs[:sen+1]]))
            #     priv_ests.append((filters_priv_varying_sensors[sen].x, filters_priv_varying_sensors[sen].P))
            # filters_priv_ests.append(priv_ests)

            # # Add same Gaussian noise to all measurements
            # noise = np.random.multivariate_normal(np.zeros(2), Z)
            # noised_zs = []
            # for sen in range(num_sensors):
            #     noised_zs.append(zs[sen] + noise)
            
            # # Unprivileged estimators for varying number of sensors
            # unpriv_ests = []
            # for sen in range(num_sensors):
            #     filters_unpriv_varying_sensors[sen].predict()
            #     filters_unpriv_varying_sensors[sen].update(np.block([z for z in noised_zs[:sen+1]]))
            #     unpriv_ests.append((filters_unpriv_varying_sensors[sen].x, filters_unpriv_varying_sensors[sen].P))
            # filters_unpriv_ests.append(unpriv_ests)

            # z1 = z1 + corr_noise[:2]
            # z2 = z2 + corr_noise[2:]
            # z_fused = fuse_correlated_estimates(z1, z2, (R2+added_noise_cov_corr)[:2,:2], (R2+added_noise_cov_corr)[:2,:2], (R2+added_noise_cov_corr)[:2,2:])
            # est_filter_fused_unprivileged_zs.predict()
            # e = est_filter_fused_unprivileged_zs.update(z_fused)
            # filter_fused.append(e)
    

        
    return

# Run main
if __name__ == '__main__':
    main()

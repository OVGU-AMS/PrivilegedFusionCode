
import numpy as np
import plotting as pltng
import estimation as estmtn
import key_stream as kystrm

class SimData:
    def __init__(self, ident, num_sensors):
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []
        self.zs = dict(((s, []) for s in range(num_sensors)))
        self.unpriv_filter_results = []
        self.priv_filters_j_ms_results = dict(((p, []) for p in range(num_sensors)))
        self.priv_filters_all_ms_results = dict(((p, []) for p in range(num_sensors)))
        return
    
    def compute_errors(self):
        self.sim_len = len(self.gt)
        self.priv_filters_j_ms_errors = {}
        self.priv_filters_all_ms_errors = {}

        for p in range(self.num_sensors-1):
            self.priv_filters_j_ms_errors[p] = [np.linalg.norm(self.priv_filters_j_ms_results[p][i][0] - self.gt[i]) for i in range(self.sim_len)]
            self.priv_filters_all_ms_errors[p] = [np.linalg.norm(self.priv_filters_all_ms_results[p][i][0] - self.gt[i]) for i in range(self.sim_len)]

        return

class AvgSimData:
    def __init__(self, sim_list):
        self.num_sensors = sim_list[0].num_sensors

        self.priv_filters_j_ms_errors_avg = {}
        self.priv_filters_all_ms_errors_avg = {}

        for p in range(self.num_sensors-1):
            self.priv_filters_j_ms_errors_avg[p] = np.mean([s.priv_filters_j_ms_errors[p] for s in sim_list], axis=0)
            self.priv_filters_all_ms_errors_avg[p] = np.mean([s.priv_filters_all_ms_errors[p] for s in sim_list], axis=0)
        
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
    sensor_correlated_covariance = np.block([[Z+Y if r==c else Z for c in range(num_sensors)] for r in range(num_sensors)])

    # Synced cryptographically random number generators
    generator_pairs = [kystrm.SharedKeyStreamFactory.make_shared_key_streams(2) for _ in range(num_sensors)]

    # Creating simulation objects (ground truth, sensors and filters)
    ground_truth = estmtn.GroundTruth(F, Q, gt_init_state)
    sensors = []
    for _ in range(num_sensors):
        sensors.append(estmtn.SensorPure(n, m, H, R))

    unpriv_filter = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, [], num_sensors)
    priv_filters_j_ms = []
    priv_filters_all_ms = []
    for j in range(num_sensors):
        gens = [g[0] for g in generator_pairs[:j+1]]
        priv_filters_j_ms.append(estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, j+1))
        priv_filters_all_ms.append(estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, num_sensors))

    # Simulation
    sim_runs = 1
    sim_steps = 100
    sims = []
    for s in range(sim_runs):

        if s % 1 == 0:
            print("Running Simulation %d ..." % s)

        sim = SimData(s, num_sensors)
        sims.append(sim)
        for _ in range(sim_steps):
            
            # Update ground truth
            gt = ground_truth.update()
            sim.gt.append(gt)

            # Generate correlated noise
            std_normals = np.block([g[1].next_n_as_std_gaussian(m) for g in generator_pairs])
            correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

            # Make all measurements and add pseudorandom noises
            zs = []
            for sen in range(num_sensors):
                true_z = sensors[sen].measure(gt)
                z = true_z + correlated_noises[sen*m:sen*m+m]
                sim.zs[sen].append(z)
                zs.append(z)

            # Unprivileged filter estimate
            unpriv_filter.predict()
            res = unpriv_filter.update(np.block(zs))
            sim.unpriv_filter_results.append(res)

            # Privileged and additional measurement privileged filters' estimates
            for j in range(num_sensors):
                priv_filters_j_ms[j].predict()
                res_j = priv_filters_j_ms[j].update(np.block(zs[:j+1]))
                sim.priv_filters_j_ms_results[j].append(res_j)

                priv_filters_all_ms[j].predict()
                res_all = priv_filters_all_ms[j].update(np.block(zs))
                sim.priv_filters_all_ms_results[j].append(res_all)
    
        # Compute errors of the filters
        sim.compute_errors()
        #pltng.plot_single_sim(sim)

    # Average simulations
    avg_sim_data = AvgSimData(sims)

    # Plot simulations
    

    return

# Run main
if __name__ == '__main__':
    main()

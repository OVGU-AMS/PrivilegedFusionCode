
import numpy as np
import plotting as pltng
import estimation as estmtn
import key_stream as kystrm

class PrivilegeSimData:
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
        
        self.unpriv_filter_errors = [np.linalg.norm(self.unpriv_filter_results[i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        self.priv_filters_j_ms_errors = {}
        self.priv_filters_all_ms_errors = {}
        for p in range(self.num_sensors):
            self.priv_filters_j_ms_errors[p] = [np.linalg.norm(self.priv_filters_j_ms_results[p][i][0] - self.gt[i])**2 for i in range(self.sim_len)]
            self.priv_filters_all_ms_errors[p] = [np.linalg.norm(self.priv_filters_all_ms_results[p][i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        return

class AvgPrivilegeSimData:
    def __init__(self, sim_list):
        self.num_sensors = sim_list[0].num_sensors
        self.sim_len = sim_list[0].sim_len

        # Used to be able to plot covariance trace as a comparison
        self.last_sim = sim_list[0]

        self.unpriv_filter_errors_avg = np.mean([s.unpriv_filter_errors for s in sim_list], axis=0)

        self.priv_filters_j_ms_errors_avg = {}
        self.priv_filters_all_ms_errors_avg = {}
        for p in range(self.num_sensors):
            self.priv_filters_j_ms_errors_avg[p] = np.mean([s.priv_filters_j_ms_errors[p] for s in sim_list], axis=0)
            self.priv_filters_all_ms_errors_avg[p] = np.mean([s.priv_filters_all_ms_errors[p] for s in sim_list], axis=0)
        return


class ParameterSimData:
    def __init__(self, ident, num_sensors, Ys, Zs):
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []
        self.zs = dict(((s, []) for s in range(num_sensors)))

        self.Ys = Ys
        self.Zs = Zs

        self.unpriv_filter_results = []
        self.priv_filters_j_ms_results = []
        self.priv_filters_all_ms_results = []
        for Y in Ys:
            self.priv_filters_j_ms_results.append([])
            self.priv_filters_all_ms_results.append([])
            for Z in Zs:
                self.priv_filters_j_ms_results[Y].append([])
                self.priv_filters_all_ms_results[Y].append([])
        
        return
    
    def compute_errors(self):
        self.sim_len = len(self.gt)
        
        self.unpriv_filter_errors = [np.linalg.norm(self.unpriv_filter_results[i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        self.priv_filters_j_ms_errors = []
        self.priv_filters_all_ms_errors = []
        for Y in self.Ys:
            self.priv_filters_j_ms_errors.append([])
            self.priv_filters_all_ms_errors.append([])
            for Z in self.Zs:
                self.priv_filters_j_ms_errors[Y].append([np.linalg.norm(self.priv_filters_j_ms_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)])
                self.priv_filters_all_ms_errors[Y].append([np.linalg.norm(self.priv_filters_all_ms_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)])

        return



class AvgParameterSimData:
    def __init__(self, sim_list):
        self.num_sensors = sim_list[0].num_sensors
        self.sim_len = sim_list[0].sim_len
        self.Ys = sim_list[0].Ys
        self.Zs = sim_list[0].Zs

        # Used to be able to plot covariance trace as a comparison
        self.last_sim = sim_list[0]

        self.unpriv_filter_errors_avg = np.mean([s.unpriv_filter_errors for s in sim_list], axis=0)

        self.priv_filters_j_ms_errors_avg = []
        self.priv_filters_all_ms_errors_avg = []
        for Y in self.Ys:
            self.priv_filters_j_ms_errors_avg.append([])
            self.priv_filters_all_ms_errors_avg.append([])
            for Z in self.Zs:
                self.priv_filters_j_ms_errors_avg[Y][Z] = np.mean([s.priv_filters_j_ms_errors[p] for s in sim_list], axis=0)
                self.priv_filters_all_ms_errors_avg[Y][Z] = np.mean([s.priv_filters_all_ms_errors[p] for s in sim_list], axis=0)
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

    # Simulations
    sim_runs = 1000
    sim_steps = 100
    sims = []
    for s in range(sim_runs):
        if s % 20 == 0:
            print("Running Simulation %d ..." % s)
        sim = PrivilegeSimData(s, num_sensors)
        sims.append(sim)

        # Synced cryptographically random number generators. Makes exactly the amount of each required by the simulation.
        # Made for each simulation to make popping from generator list easier
        sensor_generators = [kystrm.SharedKeyStreamFactory.make_shared_key_streams(1+2*num_sensors-2*s) for s in range(num_sensors)]

        # Creating simulation objects (ground truth, sensors and filters)
        ground_truth = estmtn.GroundTruth(F, Q, gt_init_state)
        sensors = []
        for _ in range(num_sensors):
            sensors.append(estmtn.SensorPure(n, m, H, R))

        unpriv_filter = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, [], num_sensors)
        priv_filters_j_ms = []
        priv_filters_all_ms = []
        for j in range(num_sensors):
            gens = [g.pop() for g in sensor_generators[:j+1]]
            priv_filters_j_ms.append(estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, j+1))

            gens = [g.pop() for g in sensor_generators[:j+1]]
            priv_filters_all_ms.append(estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, num_sensors))

        # Run simulation
        for _ in range(sim_steps):
            
            # Update ground truth
            gt = ground_truth.update()
            sim.gt.append(gt)

            # Generate correlated noise
            std_normals = np.block([g[0].next_n_as_std_gaussian(m) for g in sensor_generators])
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

        if s == sim_runs-1:
            #pltng.plot_single_sim(sim)
            pass

    # Average simulations
    avg_sim_data = AvgPrivilegeSimData(sims)

    # Plot results
    pltng.plot_privilege_differences(avg_sim_data, False, True)

    return

# Run main
if __name__ == '__main__':
    main()

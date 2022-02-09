
import numpy as np
import plotting as pltng
import estimation as estmtn
import key_stream as kystrm


SIM_RUNS = 1000
SIM_STEPS = 100
PROGRESS_PRINTS = 25
SHOW_SINGLE_SIM_PLOT = False

MAKE_PRIV_PLOT = True
MAKE_PARAM_PLOT = True
MAKE_PARAM_SCAN_PLOT = True

SAVE_NOT_SHOW_PLOTS = True


"""
 
  .d8888b. 8888888 888b     d888      8888888b.        d8888 88888888888     d8888 
 d88P  Y88b  888   8888b   d8888      888  "Y88b      d88888     888        d88888 
 Y88b.       888   88888b.d88888      888    888     d88P888     888       d88P888 
  "Y888b.    888   888Y88888P888      888    888    d88P 888     888      d88P 888 
     "Y88b.  888   888 Y888P 888      888    888   d88P  888     888     d88P  888 
       "888  888   888  Y8P  888      888    888  d88P   888     888    d88P   888 
 Y88b  d88P  888   888   "   888      888  .d88P d8888888888     888   d8888888888 
  "Y8888P" 8888888 888       888      8888888P" d88P     888     888  d88P     888 
                                                                                   
                                                                                   
                                                                                   
 
"""

class PrivilegeSimData:
    def __init__(self, ident, num_sensors):
        # Store general simulation information
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []
        self.zs = dict(((s, []) for s in range(num_sensors)))
        # Results from unprivileged and privileged filters for all the privileges considered
        self.unpriv_filter_results = []
        self.priv_filters_j_ms_results = dict(((p, []) for p in range(num_sensors)))
        self.priv_filters_all_ms_results = dict(((p, []) for p in range(num_sensors)))
        return
    
    def compute_errors(self):
        self.sim_len = len(self.gt)
        
        # Compute unprivileged errors
        self.unpriv_filter_errors = [np.linalg.norm(self.unpriv_filter_results[i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        # Compute privileges errors for all privileges considered
        self.priv_filters_j_ms_errors = {}
        self.priv_filters_all_ms_errors = {}
        for p in range(self.num_sensors):
            self.priv_filters_j_ms_errors[p] = [np.linalg.norm(self.priv_filters_j_ms_results[p][i][0] - self.gt[i])**2 for i in range(self.sim_len)]
            self.priv_filters_all_ms_errors[p] = [np.linalg.norm(self.priv_filters_all_ms_results[p][i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        return

class AvgPrivilegeSimData:
    def __init__(self, sim_list):
        # Copy general information from the first simulation
        self.num_sensors = sim_list[0].num_sensors
        self.sim_len = sim_list[0].sim_len

        # Save first simulation to be able to plot covariance trace as a comparison (doesn't matter which sim is actually saved)
        self.last_sim = sim_list[0]

        # Average the results of unprivileged and privileged filters for all the privileges considered
        self.unpriv_filter_errors_avg = np.mean([s.unpriv_filter_errors for s in sim_list], axis=0)
        self.priv_filters_j_ms_errors_avg = {}
        self.priv_filters_all_ms_errors_avg = {}
        for p in range(self.num_sensors):
            self.priv_filters_j_ms_errors_avg[p] = np.mean([s.priv_filters_j_ms_errors[p] for s in sim_list], axis=0)
            self.priv_filters_all_ms_errors_avg[p] = np.mean([s.priv_filters_all_ms_errors[p] for s in sim_list], axis=0)
        return


class ParameterSimData:
    def __init__(self, ident, num_sensors, Ys, Zs):
        # Store general simulation information
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []

        # Store the parameter values that will be varied
        self.Ys = Ys
        self.Zs = Zs

        # As params are varied, store all measurements, unprivielged and privileged filters' estimates in a 2-D dictionary of parameters
        self.zs = {}
        self.unpriv_filters_results = {}
        self.priv_filters_j_ms_results = {}
        self.priv_filters_all_ms_results = {}
        for Y in Ys:
            self.zs[Y] = {}
            self.unpriv_filters_results[Y] = {}
            self.priv_filters_j_ms_results[Y] = {}
            self.priv_filters_all_ms_results[Y] = {}
            for Z in Zs:
                self.zs[Y][Z] = dict(((s, []) for s in range(num_sensors)))
                self.unpriv_filters_results[Y][Z] = []
                self.priv_filters_j_ms_results[Y][Z] = []
                self.priv_filters_all_ms_results[Y][Z] = []
        
        return
    
    def compute_errors(self):
        self.sim_len = len(self.gt)

        # Compute errors of unprivileged, privileged with denoised measurements only and privileged with all measuremets filters, for all parameter combinations
        self.unpriv_filters_errors = {}
        self.priv_filters_j_ms_errors = {}
        self.priv_filters_all_ms_errors = {}
        for Y in self.Ys:
            self.unpriv_filters_errors[Y] = {}
            self.priv_filters_j_ms_errors[Y] = {}
            self.priv_filters_all_ms_errors[Y] = {}
            for Z in self.Zs:
                self.unpriv_filters_errors[Y][Z] = [np.linalg.norm(self.unpriv_filters_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)]
                self.priv_filters_j_ms_errors[Y][Z] = [np.linalg.norm(self.priv_filters_j_ms_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)]
                self.priv_filters_all_ms_errors[Y][Z] = [np.linalg.norm(self.priv_filters_all_ms_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        return



class AvgParameterSimData:
    def __init__(self, sim_list):
        # Copy general information from the first simulation
        self.num_sensors = sim_list[0].num_sensors
        self.sim_len = sim_list[0].sim_len
        self.Ys = sim_list[0].Ys
        self.Zs = sim_list[0].Zs

        # Save first simulation to be able to plot covariance trace as a comparison (doesn't matter which sim is actually saved)
        self.last_sim = sim_list[0]

        # Compute average errors of all considered filters for all parameter combinations
        self.unpriv_filters_errors_avg = {}
        self.priv_filters_j_ms_errors_avg = {}
        self.priv_filters_all_ms_errors_avg = {}
        for Y in self.Ys:
            self.unpriv_filters_errors_avg[Y] = {}
            self.priv_filters_j_ms_errors_avg[Y] = {}
            self.priv_filters_all_ms_errors_avg[Y] = {}
            for Z in self.Zs:
                self.unpriv_filters_errors_avg[Y][Z] = np.mean([s.unpriv_filters_errors[Y][Z] for s in sim_list], axis=0)
                self.priv_filters_j_ms_errors_avg[Y][Z] = np.mean([s.priv_filters_j_ms_errors[Y][Z] for s in sim_list], axis=0)
                self.priv_filters_all_ms_errors_avg[Y][Z] = np.mean([s.priv_filters_all_ms_errors[Y][Z] for s in sim_list], axis=0)
        return

class ParameterScanSimData:
    def __init__(self, ident, num_sensors, Y_fixed, Z_fixed, Ys, Zs, privileges):
        # Store general simulation information
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []

        # Store the values that will be varied
        self.Y_fixed = Y_fixed
        self.Z_fixed = Z_fixed
        self.Ys = Ys
        self.Zs = Zs
        self.privileges = privileges

        # As params are varied, store all measurements, unprivielged and privileged filters' estimates in a 2-D dictionary of parameters
        self.zs = {}
        self.unpriv_filters_results = {}
        self.priv_filters_j_ms_results = {}
        self.priv_filters_all_ms_results = {}
        for priv in privileges:
            self.zs[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            self.unpriv_filters_results[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            self.priv_filters_j_ms_results[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            self.priv_filters_all_ms_results[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            for Y in Ys:
                self.zs[priv]["Z_fixed"][Y] = dict(((s, []) for s in range(num_sensors)))
                self.unpriv_filters_results[priv]["Z_fixed"][Y] = []
                self.priv_filters_j_ms_results[priv]["Z_fixed"][Y] = []
                self.priv_filters_all_ms_results[priv]["Z_fixed"][Y] = []
            for Z in Zs:
                self.zs[priv]["Y_fixed"][Z] = dict(((s, []) for s in range(num_sensors)))
                self.unpriv_filters_results[priv]["Y_fixed"][Z] = []
                self.priv_filters_j_ms_results[priv]["Y_fixed"][Z] = []
                self.priv_filters_all_ms_results[priv]["Y_fixed"][Z] = []
        
        return
    
    def compute_steady_state_traces(self):
        self.sim_len = len(self.gt)

        # Compute errors of unprivileged, privileged with denoised measurements only and privileged with all measuremets filters, for all parameter combinations
        self.unpriv_filters_traces = {}
        self.priv_filters_j_ms_traces = {}
        self.priv_filters_all_ms_traces = {}
        for priv in self.privileges:
            self.unpriv_filters_traces[priv] = {}
            self.priv_filters_j_ms_traces[priv] = {}
            self.priv_filters_all_ms_traces[priv] = {}
            
            self.unpriv_filters_traces[priv]["Z_fixed"] = [np.trace(self.unpriv_filters_results[priv]["Z_fixed"][Y][-1][1]) for Y in self.Ys]
            self.priv_filters_j_ms_traces[priv]["Z_fixed"] = [np.trace(self.priv_filters_j_ms_results[priv]["Z_fixed"][Y][-1][1]) for Y in self.Ys]
            self.priv_filters_all_ms_traces[priv]["Z_fixed"] = [np.trace(self.priv_filters_all_ms_results[priv]["Z_fixed"][Y][-1][1]) for Y in self.Ys]
            
            self.unpriv_filters_traces[priv]["Y_fixed"] = [np.trace(self.unpriv_filters_results[priv]["Y_fixed"][Z][-1][1]) for Z in self.Zs]
            self.priv_filters_j_ms_traces[priv]["Y_fixed"] = [np.trace(self.priv_filters_j_ms_results[priv]["Y_fixed"][Z][-1][1]) for Z in self.Zs]
            self.priv_filters_all_ms_traces[priv]["Y_fixed"] = [np.trace(self.priv_filters_all_ms_results[priv]["Y_fixed"][Z][-1][1]) for Z in self.Zs]
        return


"""
 
 888b     d888        d8888 8888888 888b    888 
 8888b   d8888       d88888   888   8888b   888 
 88888b.d88888      d88P888   888   88888b  888 
 888Y88888P888     d88P 888   888   888Y88b 888 
 888 Y888P 888    d88P  888   888   888 Y88b888 
 888  Y8P  888   d88P   888   888   888  Y88888 
 888   "   888  d8888888888   888   888   Y8888 
 888       888 d88P     888 8888888 888    Y888 
                                                
                                                
                                                
 
"""

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

    """
    
    8888888b.  8888888b.  8888888 888     888       .d8888b. 8888888 888b     d888 
    888   Y88b 888   Y88b   888   888     888      d88P  Y88b  888   8888b   d8888 
    888    888 888    888   888   888     888      Y88b.       888   88888b.d88888 
    888   d88P 888   d88P   888   Y88b   d88P       "Y888b.    888   888Y88888P888 
    8888888P"  8888888P"    888    Y88b d88P           "Y88b.  888   888 Y888P 888 
    888        888 T88b     888     Y88o88P              "888  888   888  Y8P  888 
    888        888  T88b    888      Y888P         Y88b  d88P  888   888   "   888 
    888        888   T88b 8888888     Y8P           "Y8888P" 8888888 888       888 
                                                                                    
                                                                                    
                                                                                    
    
    """
    if MAKE_PRIV_PLOT:
        sims = []
        print("\nMaking Privilege Plot ...\n")
        for s in range(SIM_RUNS):
            # Progress printing
            if s % PROGRESS_PRINTS == 0:
                print("Running Simulation %d ..." % s)
            
            # Pseudorandom correleated and uncorrelated covariances
            Z = 2*np.eye(2)
            Y = 10*np.eye(2)
            sensor_correlated_covariance = np.block([[Z+Y if r==c else Z for c in range(num_sensors)] for r in range(num_sensors)])
            
            # Sim data storage
            sim = PrivilegeSimData(s, num_sensors)
            sims.append(sim)

            # Synced cryptographically random number generators. Makes exactly the amount of each required by the simulation.
            # Remade for each individual simulation to make popping from generator lists easier
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
            for _ in range(SIM_STEPS):
                
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

                # Privileged with denoised measurements only and with all measurements filters' estimates
                for j in range(num_sensors):
                    priv_filters_j_ms[j].predict()
                    res_j = priv_filters_j_ms[j].update(np.block(zs[:j+1]))
                    sim.priv_filters_j_ms_results[j].append(res_j)

                    priv_filters_all_ms[j].predict()
                    res_all = priv_filters_all_ms[j].update(np.block(zs))
                    sim.priv_filters_all_ms_results[j].append(res_all)
        
            # Compute errors of the filters
            sim.compute_errors()

            # Print the final simulation if flag is set. Primarily for debugging
            if SHOW_SINGLE_SIM_PLOT and s == SIM_RUNS-1:
                pltng.plot_single_priv_sim(sim)

        # Average simulations
        avg_sim_data = AvgPrivilegeSimData(sims)

        # Plot results
        pltng.plot_privilege_differences(avg_sim_data, SAVE_NOT_SHOW_PLOTS, True)

    """
    
    8888888b.     d8888 8888888b.         d8888 888b     d888       .d8888b. 8888888 888b     d888 
    888   Y88b   d88888 888   Y88b       d88888 8888b   d8888      d88P  Y88b  888   8888b   d8888 
    888    888  d88P888 888    888      d88P888 88888b.d88888      Y88b.       888   88888b.d88888 
    888   d88P d88P 888 888   d88P     d88P 888 888Y88888P888       "Y888b.    888   888Y88888P888 
    8888888P" d88P  888 8888888P"     d88P  888 888 Y888P 888          "Y88b.  888   888 Y888P 888 
    888      d88P   888 888 T88b     d88P   888 888  Y8P  888            "888  888   888  Y8P  888 
    888     d8888888888 888  T88b   d8888888888 888   "   888      Y88b  d88P  888   888   "   888 
    888    d88P     888 888   T88b d88P     888 888       888       "Y8888P" 8888888 888       888 
                                                                                                    
                                                                                                    
                                                                                                    
    
    """

    if MAKE_PARAM_PLOT:
        sims = []
        print("\nMaking Parameter Plot ...\n")
        for s in range(SIM_RUNS):
            # Progress printing
            if s % PROGRESS_PRINTS == 0:
                print("Running Simulation %d ..." % s)
            
            # Varying pseudorandom correlated and uncorrelated covariances
            Ys = [2, 10]
            Zs = [2, 10]

            # Only one privilege (number of keys) considered in this plot
            fixed_privilege = 2

            # Sim data storage
            sim = ParameterSimData(s, num_sensors, Ys, Zs)
            sims.append(sim)

            # Synced cryptographically random number generators. Makes exactly the amount of each required by the simulation.
            # Remade for each individual simulation to make popping from generator lists easier
            sensor_generators = [kystrm.SharedKeyStreamFactory.make_shared_key_streams(9-4*(s-(s%2))) for s in range(num_sensors)]

            # Creating simulation objects (ground truth, sensors and filters)
            ground_truth = estmtn.GroundTruth(F, Q, gt_init_state)
            sensors = []
            for _ in range(num_sensors):
                sensors.append(estmtn.SensorPure(n, m, H, R))

            # As correlation parameters are being changed, store all filters and correlation matrices in 2-D dictionaries
            sensor_correlated_covariances = {}
            unpriv_filters = {}
            priv_filters_j_ms = {}
            priv_filters_all_ms = {}
            for Y in Ys:
                sensor_correlated_covariances[Y] = {}
                unpriv_filters[Y] = {}
                priv_filters_j_ms[Y] = {}
                priv_filters_all_ms[Y] = {}
                for Z in Zs:
                    # Correlation matrix
                    Y_mat = Y*np.eye(2)
                    Z_mat = Z*np.eye(2)
                    sensor_correlated_covariances[Y][Z] = np.block([[Z_mat+Y_mat if r==c else Z_mat for c in range(num_sensors)] for r in range(num_sensors)])

                    # Unpriv filter
                    unpriv_filters[Y][Z] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, [], num_sensors)

                    # Priv filter, denoisable measurements only
                    gens = [g.pop() for g in sensor_generators[:fixed_privilege]]
                    priv_filters_j_ms[Y][Z] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, fixed_privilege)

                    # Priv filter, all measurements
                    gens = [g.pop() for g in sensor_generators[:fixed_privilege]]
                    priv_filters_all_ms[Y][Z] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, num_sensors)

            # Run simulation
            for _ in range(SIM_STEPS):
                
                # Update ground truth
                gt = ground_truth.update()
                sim.gt.append(gt)

                # Generate noise
                std_normals = np.block([g[0].next_n_as_std_gaussian(m) for g in sensor_generators])

                # For each of the parameter combinations compute estimates accordingly
                for Y in Ys:
                    for Z in Zs:
                        # Variables names of ease of reading (and likeness to other plot)
                        sensor_correlated_covariance = sensor_correlated_covariances[Y][Z]
                        unpriv_filter = unpriv_filters[Y][Z]

                        # Correlate noise
                        correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

                        # Make all measurements and add pseudorandom noises
                        zs = []
                        for sen in range(num_sensors):
                            true_z = sensors[sen].measure(gt)
                            z = true_z + correlated_noises[sen*m:sen*m+m]
                            sim.zs[Y][Z][sen].append(z)
                            zs.append(z)

                        # Unpriv filter estimate
                        unpriv_filter.predict()
                        res = unpriv_filter.update(np.block(zs))
                        sim.unpriv_filters_results[Y][Z].append(res)

                        # Priv filter with denoise measurements only estimate
                        priv_filters_j_ms[Y][Z].predict()
                        res_j = priv_filters_j_ms[Y][Z].update(np.block(zs[:fixed_privilege]))
                        sim.priv_filters_j_ms_results[Y][Z].append(res_j)

                        # Priv filter with all measurements estimate
                        priv_filters_all_ms[Y][Z].predict()
                        res_all = priv_filters_all_ms[Y][Z].update(np.block(zs))
                        sim.priv_filters_all_ms_results[Y][Z].append(res_all)
        
            # Compute errors of the filters
            sim.compute_errors()

            # Print the final simulation if flag is set. Primarily for debugging
            if SHOW_SINGLE_SIM_PLOT and s == SIM_RUNS-1:
                pltng.plot_single_param_sim(sim)

        # Average simulations
        avg_sim_data = AvgParameterSimData(sims)

        # Plot results
        pltng.plot_parameter_differences(avg_sim_data, SAVE_NOT_SHOW_PLOTS, True)

    """
    
    8888888b.     d8888 8888888b.         d8888 888b     d888       .d8888b.   .d8888b.        d8888 888b    888 
    888   Y88b   d88888 888   Y88b       d88888 8888b   d8888      d88P  Y88b d88P  Y88b      d88888 8888b   888 
    888    888  d88P888 888    888      d88P888 88888b.d88888      Y88b.      888    888     d88P888 88888b  888 
    888   d88P d88P 888 888   d88P     d88P 888 888Y88888P888       "Y888b.   888           d88P 888 888Y88b 888 
    8888888P" d88P  888 8888888P"     d88P  888 888 Y888P 888          "Y88b. 888          d88P  888 888 Y88b888 
    888      d88P   888 888 T88b     d88P   888 888  Y8P  888            "888 888    888  d88P   888 888  Y88888 
    888     d8888888888 888  T88b   d8888888888 888   "   888      Y88b  d88P Y88b  d88P d8888888888 888   Y8888 
    888    d88P     888 888   T88b d88P     888 888       888       "Y8888P"   "Y8888P" d88P     888 888    Y888 
                                                                                                                
                                                                                                                
                                                                                                                
    
    """

    if MAKE_PARAM_SCAN_PLOT:
        print("\nMaking Parameter Scan Plot ...\n")
        
        # # Progress printing
        # if s % PROGRESS_PRINTS == 0:
        #     print("Running Simulation %d ..." % s)
        
        # Varying pseudorandom correlated and uncorrelated covariances

        Y_fixed = 5
        Z_fixed = 5
        Ys = np.arange(0.25,10.25,0.25)
        Zs = np.arange(0.25,10.25,0.25)

        # Only one privilege (number of keys) considered in this plot

        fixed_privileges = [1,2]

        # Sim data storage
        sim = ParameterScanSimData(0, num_sensors, Y_fixed, Z_fixed, Ys, Zs, fixed_privileges)

        # Synced cryptographically random number generators. Makes exactly the amount of each required by the simulation.
        l = len(Ys)+len(Zs)
        sensor_generators = [kystrm.SharedKeyStreamFactory.make_shared_key_streams(1+4*l-2*l*(s*(s-(s%2))//3)) for s in range(num_sensors)]

        # Creating simulation objects (ground truth, sensors and filters)
        ground_truth = estmtn.GroundTruth(F, Q, gt_init_state)
        sensors = []
        for _ in range(num_sensors):
            sensors.append(estmtn.SensorPure(n, m, H, R))

        # As correlation parameters are being changed, store all filters and correlation matrices in 2-D dictionaries
        sensor_correlated_covariances = {}
        unpriv_filters = {}
        priv_filters_j_ms = {}
        priv_filters_all_ms = {}
        for priv in fixed_privileges:
            sensor_correlated_covariances[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            unpriv_filters[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            priv_filters_j_ms[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            priv_filters_all_ms[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))

            for Y in Ys:
                # Correlation matrix
                Y_mat = Y*np.eye(2)
                Z_mat = Z_fixed*np.eye(2)
                sensor_correlated_covariances[priv]["Z_fixed"][Y] = np.block([[Z_mat+Y_mat if r==c else Z_mat for c in range(num_sensors)] for r in range(num_sensors)])

                # Unpriv filter
                unpriv_filters[priv]["Z_fixed"][Y] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, [], num_sensors)

                # Priv filter, denoisable measurements only
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_j_ms[priv]["Z_fixed"][Y] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, priv)

                # Priv filter, all measurements
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_all_ms[priv]["Z_fixed"][Y] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, num_sensors)
            
            for Z in Zs:
                # Correlation matrix
                Y_mat = Y_fixed*np.eye(2)
                Z_mat = Z*np.eye(2)
                sensor_correlated_covariances[priv]["Y_fixed"][Z] = np.block([[Z_mat+Y_mat if r==c else Z_mat for c in range(num_sensors)] for r in range(num_sensors)])

                # Unpriv filter
                unpriv_filters[priv]["Y_fixed"][Z] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, [], num_sensors)

                # Priv filter, denoisable measurements only
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_j_ms[priv]["Y_fixed"][Z] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, priv)

                # Priv filter, all measurements
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_all_ms[priv]["Y_fixed"][Z] = estmtn.PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, num_sensors)



        # Run simulation
        for _ in range(SIM_STEPS):
            
            # Update ground truth
            gt = ground_truth.update()
            sim.gt.append(gt)

            # Generate noise
            std_normals = np.block([g[0].next_n_as_std_gaussian(m) for g in sensor_generators])

            # For each of the parameter combinations compute estimates accordingly
            for priv in fixed_privileges:
                for Y in Ys:
                    # Variables names of ease of reading (and likeness to other plot)
                    sensor_correlated_covariance = sensor_correlated_covariances[priv]["Z_fixed"][Y]
                    unpriv_filter = unpriv_filters[priv]["Z_fixed"][Y]

                    # Correlate noise
                    correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

                    # Make all measurements and add pseudorandom noises
                    zs = []
                    for sen in range(num_sensors):
                        true_z = sensors[sen].measure(gt)
                        z = true_z + correlated_noises[sen*m:sen*m+m]
                        sim.zs[priv]["Z_fixed"][Y][sen].append(z)
                        zs.append(z)

                    # Unpriv filter estimate
                    unpriv_filter.predict()
                    res = unpriv_filter.update(np.block(zs))
                    sim.unpriv_filters_results[priv]["Z_fixed"][Y].append(res)

                    # Priv filter with denoise measurements only estimate
                    priv_filters_j_ms[priv]["Z_fixed"][Y].predict()
                    res_j = priv_filters_j_ms[priv]["Z_fixed"][Y].update(np.block(zs[:priv]))
                    sim.priv_filters_j_ms_results[priv]["Z_fixed"][Y].append(res_j)

                    # Priv filter with all measurements estimate
                    priv_filters_all_ms[priv]["Z_fixed"][Y].predict()
                    res_all = priv_filters_all_ms[priv]["Z_fixed"][Y].update(np.block(zs))
                    sim.priv_filters_all_ms_results[priv]["Z_fixed"][Y].append(res_all)
                
                for Z in Zs:
                    # Variables names of ease of reading (and likeness to other plot)
                    sensor_correlated_covariance = sensor_correlated_covariances[priv]["Y_fixed"][Z]
                    unpriv_filter = unpriv_filters[priv]["Y_fixed"][Z]

                    # Correlate noise
                    correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

                    # Make all measurements and add pseudorandom noises
                    zs = []
                    for sen in range(num_sensors):
                        true_z = sensors[sen].measure(gt)
                        z = true_z + correlated_noises[sen*m:sen*m+m]
                        sim.zs[priv]["Y_fixed"][Z][sen].append(z)
                        zs.append(z)

                    # Unpriv filter estimate
                    unpriv_filter.predict()
                    res = unpriv_filter.update(np.block(zs))
                    sim.unpriv_filters_results[priv]["Y_fixed"][Z].append(res)

                    # Priv filter with denoise measurements only estimate
                    priv_filters_j_ms[priv]["Y_fixed"][Z].predict()
                    res_j = priv_filters_j_ms[priv]["Y_fixed"][Z].update(np.block(zs[:priv]))
                    sim.priv_filters_j_ms_results[priv]["Y_fixed"][Z].append(res_j)

                    # Priv filter with all measurements estimate
                    priv_filters_all_ms[priv]["Y_fixed"][Z].predict()
                    res_all = priv_filters_all_ms[priv]["Y_fixed"][Z].update(np.block(zs))
                    sim.priv_filters_all_ms_results[priv]["Y_fixed"][Z].append(res_all)
    
        # Compute errors of the filters
        sim.compute_steady_state_traces()

        # Print the final simulation if flag is set. Primarily for debugging
        # if SHOW_SINGLE_SIM_PLOT and s == SIM_RUNS-1:
        #     pltng.plot_single_param_sim(sim)

        # Plot results
        pltng.plot_parameter_scan(sim, SAVE_NOT_SHOW_PLOTS, True)

    return

# Run main
if __name__ == '__main__':
    main()

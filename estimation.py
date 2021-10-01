"""

"""

import numpy as np

"""
 
  ######   ######## 
 ##    ##     ##    
 ##           ##    
 ##   ####    ##    
 ##    ##     ##    
 ##    ##     ##    
  ######      ##    
 
"""

class GroundTruth:
    def __init__(self, F, Q, init_state):
        self.F = F
        self.Q = Q
        self.state = init_state
        return
    
    def update(self):
        w = np.random.multivariate_normal(np.array([0, 0, 0, 0]), self.Q)
        self.state = self.F@self.state + w
        return self.state

"""
 
  ######  ######## ##    ##  ######   #######  ########  
 ##    ## ##       ###   ## ##    ## ##     ## ##     ## 
 ##       ##       ####  ## ##       ##     ## ##     ## 
  ######  ######   ## ## ##  ######  ##     ## ########  
       ## ##       ##  ####       ## ##     ## ##   ##   
 ##    ## ##       ##   ### ##    ## ##     ## ##    ##  
  ######  ######## ##    ##  ######   #######  ##     ## 
 
"""

class SensorAbs:
    def measure(self, ground_truth):
        raise NotImplementedError

class SensorPure(SensorAbs):
    def __init__(self, n, m, H, R):
        self.n = n
        self.m = m
        self.H = H
        self.R = R
        return
    
    def measure(self, ground_truth):
        v = np.random.multivariate_normal(np.array([0, 0]), self.R)
        return self.H@ground_truth + v

"""
 
 ######## #### ##       ######## ######## ########   ######  
 ##        ##  ##          ##    ##       ##     ## ##    ## 
 ##        ##  ##          ##    ##       ##     ## ##       
 ######    ##  ##          ##    ######   ########   ######  
 ##        ##  ##          ##    ##       ##   ##         ## 
 ##        ##  ##          ##    ##       ##    ##  ##    ## 
 ##       #### ########    ##    ######## ##     ##  ######  
 
"""

class FilterAbs:
    def predict(self):
        raise NotImplementedError
    
    def update(self, measurement):
        raise NotImplementedError

class KFilter(FilterAbs):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov):
        self.n = n
        self.m = m
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = init_state
        self.P = init_cov
        return
    
    def predict(self):
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)
        return self.x, self.P
    
    def update(self, measurement):
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)
        return self.x, self.P

class PrivFusionFilter(KFilter):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, Z, Y, generators, num_measurements):
        self.Z = Z
        self.Y = Y
        self.generators = generators
        self.num_measurements = num_measurements

        self.single_m = m
        self.privilege = len(generators)
        self.correlated_noise_covariance = None
        if self.privilege > 0:
            self.correlated_noise_covariance = np.block([[Z+Y if c==r else Z for c in range(self.privilege)] for r in range(self.privilege)])

        stacked_H = np.block([[H] for _ in range(num_measurements)])
        # TODO modified stacked_R when num_measurements > len(generators)
        stacked_R = np.block([[R if c==r else np.zeros((2,2)) for c in range(num_measurements)] for r in range(num_measurements)])
        stacked_m = m*num_measurements

        super().__init__(n, stacked_m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        return
    
    def update(self, measurements):
        # Only generate noises if holding any keys
        if self.privilege > 0:

            # Generate the known noises
            std_normals = np.block([g.next_n_as_std_gaussian(self.single_m) for g in self.generators])
            correlated_noises = np.linalg.cholesky(self.correlated_noise_covariance)@std_normals

            # Remove the noises from the recieved measurements
            padding = np.array([0 for _ in range(self.single_m*(self.num_measurements - self.privilege))])
            padded_correlated_noises = np.block([correlated_noises, padding])
            measurements = measurements - padded_correlated_noises

        # Run filter udpate
        super().update(measurements)
        return self.x, self.P


# =====

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

# =====

# class SensorWithPrivileges(SensorPure):
#     def __init__(self, n, m, H, R, covars_to_remove, generators):
#         assert (len(covars_to_remove) == len(generators)), "Number of privilege classes does not match number of generators! Aborting."
#         super().__init__(n, m, H, R)
#         self.covars_to_remove = covars_to_remove
#         self.generators = generators
#         self.num_privs = len(covars_to_remove)
#         return
    
#     def measure(self, ground_truth):
#         return super().measure(ground_truth) + self.get_sum_of_additional_noises()
    
#     def get_sum_of_additional_noises(self):
#         noise = 0
#         for i in range(self.num_privs):
#             n = self.generators[i].next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covars_to_remove[i])
#             noise += n
#             #print("Sensor noise %d: " % i, n)
#         #print("Sensor noise sum: ", noise)
#         return noise

# =====

# class UnprivFilter(KFilter):
#     def __init__(self, n, m, F, Q, H, R, init_state, init_cov, covars_to_remove):
#         super().__init__(n, m, F, Q, H, R, init_state, init_cov)
#         self.R = self.R + sum(covars_to_remove)
#         return

# class PrivFilter(KFilter):
#     def __init__(self, n, m, F, Q, H, R, init_state, init_cov, priv_covar, covar_to_remove, generator):
#         super().__init__(n, m, F, Q, H, R, init_state, init_cov)
#         self.R = self.R + priv_covar
#         self.covar_to_remove = covar_to_remove
#         self.generator = generator
#         return
    
#     def update(self, measurement):
#         super().update(measurement - self.get_additional_noise())
#         return self.x, self.P
    
#     def get_additional_noise(self):
#         return self.generator.next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covar_to_remove)

# class MultKeyPrivFilter(KFilter):
#     def __init__(self, n, m, F, Q, H, R, init_state, init_cov, priv_covar, covars_to_remove, generators):
#         assert (len(covars_to_remove) == len(generators)), "Number of privilege classes does not match number of generators! Aborting."
#         super().__init__(n, m, F, Q, H, R, init_state, init_cov)
#         self.R = self.R + priv_covar
#         self.covars_to_remove = covars_to_remove
#         self.generators = generators
#         self.num_privs = len(covars_to_remove)
#         return
    
#     def update(self, measurement):
#         super().update(measurement - self.get_sum_of_additional_noises())
#         return self.x, self.P
    
#     def get_sum_of_additional_noises(self):
#         noise = 0
#         for i in range(self.num_privs):
#             n = self.generators[i].next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covars_to_remove[i])
#             noise += n
#             #print("Added noise %d: " % i, n)
#         #print("Added noise sum: ", noise)
#         return noise
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

class SensorWithPrivileges(SensorPure):
    def __init__(self, n, m, H, R, covars_to_remove, generators):
        assert (len(covars_to_remove) == len(generators)), "Number of privilege classes does not match number of generators! Aborting."
        super().__init__(n, m, H, R)
        self.covars_to_remove = covars_to_remove
        self.generators = generators
        self.num_privs = len(covars_to_remove)
        return
    
    def measure(self, ground_truth):
        return super().measure(ground_truth) + self.get_sum_of_additional_noises()
    
    def get_sum_of_additional_noises(self):
        noise = 0
        for i in range(self.num_privs):
            n = self.generators[i].next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covars_to_remove[i])
            noise += n
            #print("Sensor noise %d: " % i, n)
        #print("Sensor noise sum: ", noise)
        return noise
    

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

class UnprivFilter(KFilter):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, covars_to_remove):
        super().__init__(n, m, F, Q, H, R, init_state, init_cov)
        self.R = self.R + sum(covars_to_remove)
        return

class PrivFilter(KFilter):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, priv_covar, covar_to_remove, generator):
        super().__init__(n, m, F, Q, H, R, init_state, init_cov)
        self.R = self.R + priv_covar
        self.covar_to_remove = covar_to_remove
        self.generator = generator
        return
    
    def update(self, measurement):
        super().update(measurement - self.get_additional_noise())
        return self.x, self.P
    
    def get_additional_noise(self):
        return self.generator.next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covar_to_remove)

class MultKeyPrivFilter(KFilter):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, priv_covar, covars_to_remove, generators):
        assert (len(covars_to_remove) == len(generators)), "Number of privilege classes does not match number of generators! Aborting."
        super().__init__(n, m, F, Q, H, R, init_state, init_cov)
        self.R = self.R + priv_covar
        self.covars_to_remove = covars_to_remove
        self.generators = generators
        self.num_privs = len(covars_to_remove)
        return
    
    def update(self, measurement):
        super().update(measurement - self.get_sum_of_additional_noises())
        return self.x, self.P
    
    def get_sum_of_additional_noises(self):
        noise = 0
        for i in range(self.num_privs):
            n = self.generators[i].next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covars_to_remove[i])
            noise += n
            #print("Added noise %d: " % i, n)
        #print("Added noise sum: ", noise)
        return noise
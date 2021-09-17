"""

"""

import numpy as np
import key_stream



def gen_corr_noise(m_dim, add_noises_whole_cov, inds_obtained, measurements, ind_wanted, key_stream):
    
    cov_b = add_noises_whole_cov[ind_wanted*m_dim:ind_wanted*m_dim+m_dim, ind_wanted*m_dim:ind_wanted*m_dim+m_dim]

    if len(inds_obtained)>0:
        cov_a = np.block([[add_noises_whole_cov[s_row*m_dim:s_row*m_dim+m_dim, s_col*m_dim:s_col*m_dim+m_dim] for s_col in inds_obtained] for s_row in inds_obtained])
        cov_c = np.block([[add_noises_whole_cov[s_row*m_dim:s_row*m_dim+m_dim, ind_wanted*m_dim:ind_wanted*m_dim+m_dim]] for s_row in inds_obtained])
        ms = np.block(measurements).T
    
        mean = cov_c.T@np.linalg.inv(cov_a)@ms
        cov = cov_b - cov_c.T@np.linalg.inv(cov_a)@cov_c

        # print(cov_a)
        # print(cov_c)
        # print(cov_b)
        # print(ms)

    else:
        mean = np.zeros(m_dim)
        cov = cov_b

    return key_stream.next_n_as_gaussian(m_dim, mean, cov)






num_sensors = 4
measurement_dim = 2


add_noise_covs = [np.array([[5, 2],
                            [2, 5]]),
                  np.array([[6, 3],
                            [3, 5]]),
                  np.array([[4, 3],
                            [3, 4]]),
                  np.array([[6, 5],
                            [5, 6]])]
add_noise_cross_cov = np.array([[2, 2],
                                [2, 2]])
add_noises_whole_cov = np.block([[add_noise_covs[c] if r==c else add_noise_cross_cov for c in range(num_sensors)] for r in range(num_sensors)])
add_noises_whole_mean = np.zeros(measurement_dim*num_sensors)
print(add_noises_whole_cov)



sensor_keystreams_groups = [key_stream.SharedKeyStreamFactory.make_shared_key_streams(2) for _ in range(num_sensors)]

std_gaussians = np.array([])
for s in range(num_sensors):
    std_gaussians = np.append(std_gaussians, sensor_keystreams_groups[s][0].next_n_as_std_gaussian(measurement_dim))

A = np.linalg.cholesky(add_noises_whole_cov)
add_noises_allgen = A@std_gaussians
print(add_noises_allgen)


# Written for fixed sensors
# ===
# sen1_std_gaussians = sensor_keystreams_groups[0][1].next_n_as_std_gaussian(measurement_dim)
# B = np.linalg.cholesky(add_noise_covs[0])
# sen1_noise = B@sen1_std_gaussians
# print(sen1_noise)

# sen2_std_gaussians = sensor_keystreams_groups[1][1].next_n_as_std_gaussian(measurement_dim)
# C = np.linalg.cholesky(add_noise_covs[1] - add_noise_cross_cov@np.linalg.inv(add_noise_covs[0])@add_noise_cross_cov)
# sen2_noise = add_noise_cross_cov@np.linalg.inv(add_noise_covs[0])@sen1_noise + C@sen2_std_gaussians
# print(sen2_noise)

# sen4_noise = gen_corr_noise(measurement_dim, add_noises_whole_cov, [0,1], [sen1_noise, sen2_noise], 3, sensor_keystreams_groups[3][1])
# print(sen4_noise)

gen_order = [0,1,2,3] # Must be in order :(
noise_lst = []
for i,s in enumerate(gen_order):
    noise_lst.append(gen_corr_noise(measurement_dim, add_noises_whole_cov, gen_order[:i], noise_lst, s, sensor_keystreams_groups[s][1]))
noises = np.block(noise_lst)
print(noises)


# ===


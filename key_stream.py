"""

"""

import numpy as np
import scipy.stats as sp
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import matplotlib.pyplot as plt

class SharedKeyStreamFactory:
    @staticmethod
    def make_shared_key_streams(n):
        key = get_random_bytes(16)
        ciphers = []
        ciphers.append(AES.new(key, AES.MODE_CTR))
        for _ in range(n-1):
         ciphers.append(AES.new(key, AES.MODE_CTR, nonce=ciphers[0].nonce))
        return [KeyStream(cipher) for cipher in ciphers]


class KeyStream:
    def __init__(self, cipher):
        self.cipher = cipher
        self.bytes_to_read = 16
        self.max_read_int = 2**(16*8) - 1
        return
    
    def next(self):
        next_read = self.cipher.encrypt(self.bytes_to_read*b'\x00')
        next_int = int.from_bytes(next_read, byteorder='big', signed=False)
        return next_int
    
    def next_as_std_uniform(self):
        next_int = self.next()
        next_unif = (next_int/self.max_read_int)
        return next_unif
    
    def next_n_as_std_gaussian(self, n):
        # Box-Muller transform for n standard normals from generated uniforms
        std_normals = []
        for i in range(n):
            if i%2 == 0:
                u1=self.next_as_std_uniform()
                u2=self.next_as_std_uniform()
                u1_cmp = np.sqrt(-2*np.log(u1))
                u2_cmp = 2*np.pi*u2
                std_normals.append(u1_cmp*np.cos(u2_cmp))
            else:
                std_normals.append(u1_cmp*np.sin(u2_cmp))
        return np.array(std_normals)

    def next_n_as_gaussian(self, n, mean, covariance):
        std_normals = self.next_n_as_std_gaussian(n)
        # Conversion to samples from multivariate normal
        A = np.linalg.cholesky(covariance)
        return mean + A@std_normals


# Quick testing

# a, b = KeyStreamPairFactory.make_pair()

# print(b.next_n_as_gaussian(2, np.array([0,0]), np.array([[4,0],[0,4]])))
# print(a.next_n_as_gaussian(2, np.array([0,0]), np.array([[4,0],[0,4]])))
# print(b.next_n_as_gaussian(2, np.array([0,0]), np.array([[4,0],[0,4]])))
# print(a.next_n_as_gaussian(2, np.array([0,0]), np.array([[4,0],[0,4]])))

# a_list = []
# for i in range(1000):
#     a_list.append(a.next_n_as_gaussian(2, np.array([0,0]), np.array([[4,0],[0,4]])))
# plt.scatter([x[0] for x in a_list], [x[1] for x in a_list])
# plt.show()
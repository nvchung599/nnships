import numpy as np
import scipy.special
import random
from defs import *
from general import *

class Net:

    def __init__(self, num_in, num_h_1, num_h_2, num_out):
        self.num_L1 = num_in
        self.num_L2 = num_h_1
        self.num_L3 = num_h_2
        self.num_L4 = num_out
        self.Theta_1 = np.random.uniform(-5, 5, size=(self.num_L2, self.num_L1))
        self.Theta_2 = np.random.uniform(-5, 5, size=(self.num_L3, self.num_L2))
        self.Theta_3 = np.random.uniform(-5, 5, size=(self.num_L4, self.num_L3))
        self.activation_function = lambda x: scipy.special.expit(x)

    def feed_fwd(self, input_list):
        """input nx1, output kx1"""
        x = np.array(input_list, ndmin=2)
        Z_2 = np.matmul(self.Theta_1, x)
        A_2 = self.activation_function(Z_2)
        Z_3 = np.matmul(self.Theta_2, A_2)
        A_3 = self.activation_function(Z_3)
        Z_4 = np.matmul(self.Theta_3, A_3)
        A_4 = self.activation_function(Z_4)
        return A_4

    def modify_weights(self):
        modify_matrix(self.Theta_1)
        modify_matrix(self.Theta_2)
        modify_matrix(self.Theta_3)

    def create_mixed_weights(self, net_1, net_2):
        self.Theta_1 = get_mix_from_matrices(net_1.Theta_1, net_2.Theta_1)
        self.Theta_2 = get_mix_from_matrices(net_1.Theta_2, net_2.Theta_2)
        self.Theta_3 = get_mix_from_matrices(net_1.Theta_3, net_2.Theta_3)


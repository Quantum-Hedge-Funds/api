import numpy as np
import fastdtw

def calculate_similarity_matrix(data, n):
    rho = np.zeros((n, n))
    for i_i in range(0, n):
        rho[i_i, i_i] = 1.0
        for j_j in range(i_i + 1, n):
            this_rho, _ = fastdtw.fastdtw(data[i_i], data[j_j])
            this_rho = 1.0 / this_rho
            rho[i_i, j_j] = this_rho
            rho[j_j, i_i] = this_rho
    return rho
import numpy as np
from scipy.spatial.distance import cdist


def igd(pareto_front, F, normalize=False):
    pareto_front = pareto_front

    if normalize:
        N = np.max(pareto_front, axis=0) - np.min(pareto_front, axis=0)

        def dist(A, B):
            return np.sqrt(np.sum(np.square((A - B) / N), axis=1))
        D = vectorized_cdist(pareto_front, F, dist)
    else:
        D = cdist(pareto_front, F)

    return np.mean(np.min(D, axis=1))


def igd_plus(pareto_front, F):
    D = modified_cdist(pareto_front, F)
    return np.mean(np.min(D, axis=1))


def vectorized_cdist(A, B, func_dist):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = func_dist(u, v)
    M = np.reshape(D, (A.shape[0], B.shape[0]))
    return M


def modified_cdist(pareto_front, F):
    # Accelerated by Numpy
    size_pareto_front = np.shape(pareto_front)[0]
    size_solution_set = np.shape(F)[0]
    a = np.transpose(np.tile(pareto_front, (size_solution_set, 1, 1)), (1, 0, 2))
    b = np.tile(F, (size_pareto_front, 1, 1))
    objwise_difference = np.subtract(b, a)
    dist_matrix = np.sqrt(
                    np.sum(
                        np.square(
                            np.maximum(objwise_difference, np.zeros(np.shape(objwise_difference)))), axis=2))

    return dist_matrix

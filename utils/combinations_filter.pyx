# combinations_filter.pyx

cimport cython
import numpy as np
cimport numpy as np
from libc.time cimport time
from cython.parallel import prange
from libc.math cimport sqrt
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double compute_cv_distance(np.int64_t[:, :] nodes_view, np.int64_t i, np.int64_t j,
                                double[:, :] distance_matrix_view, np.int64_t N) nogil:
    cdef np.int64_t k
    cdef np.int64_t idx_i, idx_j
    cdef double distances_sum = 0.0
    cdef double distances_sq_sum = 0.0
    cdef double distance
    cdef double mean_distance, std_distance, cv_distance
    cdef double temp

    for k in range(N):
        idx_i = nodes_view[i, k]
        idx_j = nodes_view[j, k]
        distance = distance_matrix_view[idx_i, idx_j]
        if distance <= 0:
            return -1.0 

        distances_sum += distance
        distances_sq_sum += distance * distance

    mean_distance = distances_sum / N
    temp = (distances_sq_sum / N) - (mean_distance * mean_distance)
    if temp < 0:
        temp = 0.0 
    std_distance = sqrt(temp)
    cv_distance = (std_distance / mean_distance) * 100.0

    return cv_distance

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filtered_combinations(np.ndarray[np.int64_t, ndim=2, mode='c'] nodes not None,
                          np.ndarray[np.float64_t, ndim=2, mode='c'] distance_matrix not None,
                          double max_cv=15):
    cdef np.int64_t len_nodes = nodes.shape[0]
    cdef np.int64_t N = nodes.shape[1]
    cdef np.int64_t i, j, k
    cdef bint all_diff
    cdef double cv_distance
    cdef double start_time, elapsed_time
    cdef double[:, :] distance_matrix_view = distance_matrix
    cdef np.int64_t[:, :] nodes_view = nodes
    cdef vector[long long] local_results 

    cdef list result = []

    start_time = time(NULL) 

    with nogil:
        for i in prange(len_nodes, schedule='dynamic'):
            local_results.clear()  # Limpa o vector antes de usar

            for j in range(i + 1, len_nodes):

                # Verifica se todos os elementos de nodes_view[i, :] são diferentes de nodes_view[j, :]
                all_diff = True
                for k in range(N):
                    if nodes_view[i, k] == nodes_view[j, k]:
                        all_diff = False
                        break

                if all_diff:
                    # Calcula o cv_distance usando a função separada
                    cv_distance = compute_cv_distance(nodes_view, i, j, distance_matrix_view, N)

                    if cv_distance >= 0.0 and cv_distance < max_cv:
                        # Armazena os índices i, j
                        local_results.push_back(i)
                        local_results.push_back(j)

            # Unindo os resultados locais no resultado final
            with gil:
                for idx in range(0, local_results.size(), 2):
                    result.append((nodes[local_results[idx]], nodes[local_results[idx+1]]))

            # Atualizando o progresso
            if i % 100 == 0:
                elapsed_time = time(NULL) - start_time
                with gil:
                    print(f"Progress: {i}/{len_nodes}, Elapsed time: {elapsed_time:.2f} seconds")

    # Retorna a lista final de pares de nós
    return result

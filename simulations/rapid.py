import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from .dmp import dmp_prob

def global_preheat_steps(S, I, R, in_neighbors, beta, gamma, steps=10):
    N = len(S)
    for _ in range(steps):
        S_new = np.copy(S)
        I_new = np.copy(I)
        R_new = np.copy(R)

        for v in range(N):
            s_prev = S[v]
            i_prev = I[v]
            neighbor_I = I[in_neighbors[v]]

            prod_term = np.prod(1.0 - beta * neighbor_I)

            s_cur = s_prev * prod_term
            i_cur = (s_prev - s_cur) + i_prev * (1.0 - gamma)
            r_cur = R[v] + i_prev * gamma

            S_new[v] = s_cur
            I_new[v] = i_cur
            R_new[v] = r_cur

        S[:] = S_new
        I[:] = I_new
        R[:] = R_new


def rapid_prob(
        G, beta, gamma, initial_infected_nodes, tol=0.001,
        preheat_steps=20, preheat_method='global'):
    """
    preheat_method:
        'dmp' → 用 DMP 预热 (default)
        'global' → 用原 global_preheat_steps
        None → 不预热
    """
    N = G.number_of_nodes()
    S = np.ones(N)
    I = np.zeros(N)
    R = np.zeros(N)
    for node in initial_infected_nodes:
        S[node] = 0
        I[node] = 1

    in_neighbors = {v: list(G.predecessors(v)) for v in G.nodes()}
    out_neighbors = {v: list(G.successors(v)) for v in G.nodes()}

    if preheat_steps > 0 and preheat_method == 'dmp':
        # DMP 预热
        dmp_results = dmp_prob(G, beta, gamma, initial_infected_nodes, tol=1e-3, max_steps=preheat_steps)
        last_result = dmp_results[-1]
        for node in G.nodes():
            S[node] = last_result['susceptible'][node]
            I[node] = last_result['infected'][node]
            R[node] = last_result['recovered'][node]
    elif preheat_steps > 0 and preheat_method == 'global':
        # 原全局预热
        global_preheat_steps(S, I, R, in_neighbors, beta, gamma, steps=preheat_steps)
    # 如果 preheat_method=None 或 preheat_steps<=0，则不做预热

    residuals = {v: 0.0 for v in G.nodes()}
    for u in G.nodes():
        if I[u] > 0:
            for w in out_neighbors[u]:
                residuals[w] += beta * I[u]

    in_heap = {v: False for v in G.nodes()}
    heap = []

    def push_node(u):
        if residuals[u] > tol and not in_heap[u]:
            in_heap[u] = True
            heapq.heappush(heap, (-residuals[u], u))

    for v in G.nodes():
        if residuals[v] > tol:
            in_heap[v] = True
            heapq.heappush(heap, (-residuals[v], v))

    iteration_results = []
    t = 0
    iteration_results.append({
        'iteration': t,
        'susceptible': {node: S[node] for node in G.nodes()},
        'infected': {node: I[node] for node in G.nodes()},
        'recovered': {node: R[node] for node in G.nodes()}
    })

    while heap:
        t += 1
        while heap:
            neg_residual, v = heapq.heappop(heap)
            in_heap[v] = False
            residual_val = -neg_residual
            if residual_val <= tol:
                continue

            S_prev = S[v]
            I_prev = I[v]
            R_prev = R[v]

            neighbor_I = I[in_neighbors[v]]

            prod_term = np.prod(1.0 - beta * neighbor_I)

            s_new = S_prev * prod_term
            i_new = (S_prev - s_new) + I_prev * (1 - gamma)
            r_new = R_prev + I_prev * gamma

            if i_new > tol:
                S[v] = s_new
                I[v] = i_new
                R[v] = r_new
                delta_i = (I[v] - I_prev)

                for nbr in out_neighbors[v]:
                    old_val = residuals[nbr]
                    residuals[nbr] += beta * delta_i
                    if residuals[nbr] > tol and (residuals[nbr] > old_val):
                        push_node(nbr)
                residuals[v] = 0.0
            else:
                residuals[v] = 0.0

        iteration_results.append({
            'iteration': t,
            'susceptible': {node: S[node] for node in G.nodes()},
            'infected': {node: I[node] for node in G.nodes()},
            'recovered': {node: R[node] for node in G.nodes()}
        })

    return iteration_results
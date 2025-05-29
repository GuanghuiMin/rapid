import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq


def global_preheat_steps(S, I, R, in_neighbors, beta, gamma, steps=20):
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
        preheat_steps=20):
    N = G.number_of_nodes()
    S = np.ones(N)
    I = np.zeros(N)
    R = np.zeros(N)
    for node in initial_infected_nodes:
        S[node] = 0
        I[node] = 1

    in_neighbors = {v: list(G.predecessors(v)) for v in G.nodes()}
    out_neighbors = {v: list(G.successors(v)) for v in G.nodes()}
    if preheat_steps > 0:
        global_preheat_steps(S, I, R, in_neighbors, beta, gamma, steps=preheat_steps)

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
        # print(len(heap))
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

        S_temp = S_prev
        I_temp = I_prev
        R_temp = R_prev

        s_new = S_temp * prod_term
        i_new = (S_temp - s_new) + I_temp * (1 - gamma)
        r_new = R_temp + I_temp * gamma
        S_temp, I_temp, R_temp = s_new, i_new, r_new

        S_new = S_temp
        I_new = I_temp
        R_new = R_temp

        if I_new > tol:
            s = S[v]
            S[v] = S_new
            I[v] = I_new
            R[v] = R_new
            # delta_i = I[v]
            # delta_i = abs(I[v] - I_prev)
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


def plot_sir_trends(results, G):
    iterations = [result['iteration'] for result in results]
    S_counts = [sum(result['susceptible'].values()) for result in results]
    I_counts = [sum(result['infected'].values()) for result in results]
    R_counts = [sum(result['recovered'].values()) for result in results]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, S_counts, label='Susceptible (S)', color='blue')
    plt.plot(iterations, I_counts, label='Infected (I)', color='red')
    plt.plot(iterations, R_counts, label='Recovered (R)', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Number of individuals (Expected)')
    plt.title('SIR Model: Expected Number of Individuals in Each State')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_conditional_probability_iteration_estimation():
    G = nx.barabasi_albert_graph(100, 2).to_directed()
    beta = 0.03
    gamma = 0.01
    initial_infected_nodes = [0, 1]
    tol = 0.001

    results = rapid_prob(
        G, beta, gamma, initial_infected_nodes, tol,
        preheat_steps=0)
    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Susceptible: {result['susceptible']}")
        print(f"  Infected: {result['infected']}")
        print(f"  Recovered: {result['recovered']}\n")

    plot_sir_trends(results, G)


if __name__ == '__main__':
    test_conditional_probability_iteration_estimation()
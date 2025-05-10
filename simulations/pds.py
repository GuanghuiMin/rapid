import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math


def pds_prob(G, beta, gamma, initial_infected_nodes, tol=0.001):
    N = G.number_of_nodes()
    status = np.zeros((N, 3))

    for node in G.nodes():
        if node in initial_infected_nodes:
            status[node, 1] = 1
        else:
            status[node, 0] = 1

    s = status[:, 0]
    i = status[:, 1]
    r = status[:, 2]

    iteration_results = []
    t = 0

    iteration_results.append({
        'iteration': t,
        'susceptible': {node: s[node] for node in G.nodes()},
        'infected': {node: i[node] for node in G.nodes()},
        'recovered': {node: r[node] for node in G.nodes()}
    })

    while max(i) > tol:
        t += 1
        for node in G.nodes():
            in_neighbors = list(G.predecessors(node))
            s_new = np.prod([1 - beta * i[x] for x in in_neighbors]) * s[node]
            i_new = s[node] - s_new + i[node] * (1 - gamma)
            r_new = r[node] + i[node] * gamma

            s[node] = s_new
            i[node] = i_new
            r[node] = r_new

        iteration_results.append({
            'iteration': t,
            'susceptible': {node: s[node] for node in G.nodes()},
            'infected': {node: i[node] for node in G.nodes()},
            'recovered': {node: r[node] for node in G.nodes()}
        })

    return iteration_results


def calculate_theoretical_curve(G, beta, gamma, initial_infected_nodes, l, T):
    A = nx.to_numpy_array(G)
    A_tilde = A.copy()
    for j in initial_infected_nodes:
        A_tilde[j, :] = 0
        A_tilde[:, j] = 0

    S = []
    for k in range(1, l + 1):
        if k == 1:
            M_k = A
        else:
            M_k = A.dot(np.linalg.matrix_power(A_tilde, k - 1))
        S_k = np.sum(M_k[initial_infected_nodes, :])
        S.append(S_k)

    curve = []
    for t in range(T + 1):
        val = 0
        for k in range(1, l + 1):
            if t >= k:
                term = S[k - 1] * (beta ** k) * ((1 - gamma) ** (t - k)) * ((t - k) ** k) / math.factorial(k)
                val += term
        curve.append(val)
    return curve


def plot_sir_trends(results, G, theoretical_curve=None):
    iterations = [result['iteration'] for result in results]

    S_counts = []
    I_counts = []
    R_counts = []

    for result in results:
        S_counts.append(sum(result['susceptible'].values()))
        I_counts.append(sum(result['infected'].values()))
        R_counts.append(sum(result['recovered'].values()))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, S_counts, label='Susceptible (S)', color='blue')
    plt.plot(iterations, I_counts, label='Infected (I)', color='red')
    plt.plot(iterations, R_counts, label='Recovered (R)', color='green')
    if theoretical_curve is not None:
        plt.plot(iterations, theoretical_curve, '--', label='Theoretical Curve', color='orange')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Time (iterations)")
    plt.ylabel("Number of nodes")
    plt.title("SIR dynamics with theoretical curve")
    plt.show()


def test_pds_prob():
    G = nx.barabasi_albert_graph(1000, 2)
    G = G.to_directed()

    beta = 0.01
    gamma = 0.008
    initial_infected_nodes = range(10)
    tol = 1e-3
    l = 2

    results = pds_prob(G, beta, gamma, initial_infected_nodes, tol)

    for result in results:
        print(f"Iteration {result['iteration']}:")
        print(f"  Susceptible: {result['susceptible']}")
        print(f"  Infected: {result['infected']}")
        print(f"  Recovered: {result['recovered']}\n")

    T_final = results[-1]['iteration']
    theoretical_curve = calculate_theoretical_curve(G, beta, gamma, initial_infected_nodes, l, T_final)
    plot_sir_trends(results, G, theoretical_curve)

    print(f"lower bound: {np.log(tol) / np.log(1 - gamma)}")
    print(f"max in_degree: {max(dict(G.in_degree()).values())}")
    print(f"diameter: {nx.diameter(G)}")
    print(-results[-1]['iteration'] / np.log(gamma))


if __name__ == '__main__':
    test_pds_prob()
import numpy as np
import networkx as nx

def cave_index(src_nodes, tar_nodes):
    edge_list = [(int(s), int(t)) for s, t in zip(src_nodes, tar_nodes)]
    E = len(edge_list)
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    attr = {edge: idx for idx, edge in enumerate(edge_list)}
    nx.set_edge_attributes(G, attr, "idx")

    cave = []
    for edge in edge_list:
        rev = (edge[1], edge[0])
        if G.has_edge(*rev):
            cave.append(G.edges[rev]["idx"])
        else:
            cave.append(E)
    return np.array(cave, dtype=int)


class DMP_SIR:
    def __init__(self, weight_adj: np.ndarray, nodes_gamma: np.ndarray):
        self.N = weight_adj.shape[0]
        row, col = np.nonzero(weight_adj)
        self.E = len(row)

        self.src = row
        self.dst = col
        self.w = weight_adj[row, col]
        self.cav = cave_index(self.src, self.dst)

        self.gamma_nodes = nodes_gamma

    def mulmul(self, Theta: np.ndarray) -> np.ndarray:
        prod_to_node = np.ones(self.N, dtype=float)
        for e in range(self.E):
            prod_to_node[self.dst[e]] *= Theta[e]
        num = prod_to_node[self.src]

        cav_prod = np.ones(self.E + 1, dtype=float)
        for e in range(self.E):
            cav_prod[self.cav[e]] *= Theta[e]
        denom = cav_prod[:self.E]

        return num / (denom + 1e-12)

    def run(self, seed_list, tol: float = 1e-3, max_steps=None):
        seeds = np.zeros(self.N, dtype=float)
        seeds[seed_list] = 1.0
        S = 1.0 - seeds
        I = seeds.copy()
        R = np.zeros(self.N, dtype=float)

        S_src = S[self.src]
        Phi = 1.0 - S_src
        Theta = np.ones(self.E, dtype=float)

        results = []
        results.append({
            'iteration': 0,
            'susceptible': {n: S[n] for n in range(self.N)},
            'infected': {n: I[n] for n in range(self.N)},
            'recovered': {n: R[n] for n in range(self.N)},
        })

        t = 0
        while True:
            t += 1
            Theta = Theta - self.w * Phi
            S_ij = (1.0 - seeds[self.src]) * self.mulmul(Theta)
            Phi = (1.0 - self.w) * (1.0 - self.gamma_nodes[self.src]) * Phi - (S_ij - S_src)
            S_src = S_ij.copy()

            prod_to_node = np.ones(self.N, dtype=float)
            for e in range(self.E):
                prod_to_node[self.dst[e]] *= Theta[e]
            S = (1.0 - seeds) * prod_to_node

            R = R + self.gamma_nodes * I
            I = 1.0 - S - R

            results.append({
                'iteration': t,
                'susceptible': {n: S[n] for n in range(self.N)},
                'infected': {n: I[n] for n in range(self.N)},
                'recovered': {n: R[n] for n in range(self.N)},
            })

            if I.max() <= tol or (max_steps is not None and t >= max_steps):
                break

        return results


def dmp_prob(G: nx.DiGraph,
             beta: float,
             gamma: float,
             initial_infected_nodes,
             tol: float = 1e-3,
             max_steps=None):
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    W = A * beta
    gammas = np.full(N, gamma, dtype=float)

    model = DMP_SIR(W, gammas)
    return model.run(initial_infected_nodes, tol=tol, max_steps=max_steps)
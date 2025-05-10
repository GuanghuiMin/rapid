import os
import pandas as pd
import networkx as nx
import random
import warnings
import numpy as np
from tqdm import tqdm
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.sparse import isspmatrix, csr_matrix
from scipy.sparse.linalg import eigs

file_title_mapping=dataset_mapping = {
    "soc-sign-bitcoinalpha.csv.gz": "Bitcoin-Alpha",
    "soc-sign-bitcoinotc.csv.gz": "Bitcoin-OTC",
    "p2p-Gnutella08.txt.gz": "p2p-Gnutella",
    "p2p-Gnutella31.txt.gz": "p2p-Gnutella",
    "soc-Epinions1.txt.gz": "soc-Epinions",
    "soc-Slashdot0902.txt.gz": "soc-Slashdot",
    "email-Enron.txt.gz": "email-Enron",
    "email-EuAll.txt.gz": "email-EuAll",
    "soc-pokec-relationships.txt.gz": "soc-Pokec"
}

def save_args_to_csv(args, output_dir):
    args_dict = vars(args)
    args_df = pd.DataFrame(args_dict.items(), columns=["Argument", "Value"])
    args_df.to_csv(os.path.join(output_dir, "args_settings.csv"), index=False)
    print(f"Arguments saved to: {os.path.join(output_dir, 'args_settings.csv')}")


def calculate_centralities_approx(
        graph,
        alpha=0.85,
        max_iter=100,
        tol=1e-6,
        k_for_approx=100,
        seed=42,
        n_jobs=-1,
        methods=None
):
    """
    Compute approximate centralities for the given graph using the specified methods,
    and measure the runtime of each centrality computation.

    Parameters:
      graph: a NetworkX graph.
      alpha: damping factor for PageRank.
      max_iter: maximum iterations for iterative methods.
      tol: tolerance for convergence.
      k_for_approx: sample size for approximations (Betweenness/Closeness).
      seed: random seed used for sampling.
      n_jobs: number of threads to run in parallel (if n_jobs > 0).
      methods: list of centrality methods to compute
               (default: ["Degree", "Eigenvector", "PageRank", "Betweenness", "Closeness"]).

    Returns:
      A tuple (results, times) where:
         results: a dict mapping centrality name to its computed centrality dictionary.
         times: a dict mapping centrality name to its computation runtime in seconds.
    """
    if methods is None:
        methods = ["Degree", "Eigenvector", "PageRank", "Betweenness", "Closeness"]
    elif isinstance(methods, str):
        methods = [methods]

    results = {}
    times = {}
    n = graph.number_of_nodes()
    adj_matrix = nx.adjacency_matrix(graph)

    centrality_functions = {
        "Degree": lambda: {node: deg / float(n - 1)
                           for node, deg in graph.degree()},
        "Eigenvector": lambda: approximate_eigenvector_centrality(adj_matrix, max_iter=max_iter, tol=tol),
        "PageRank": lambda: approximate_pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol),
        "Betweenness": lambda: approximate_betweenness(graph, k=min(k_for_approx, n), seed=seed),
        "Closeness": lambda: approximate_closeness_centrality(graph, k=min(k_for_approx, n), seed=seed)
    }

    # Only compute the specified methods.
    selected_functions = {name: func for name, func in centrality_functions.items() if name in methods}

    def run_and_time(name, func):
        start = time.time()
        result = func()
        end = time.time()
        return name, result, end - start

    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        futures = {executor.submit(run_and_time, name, func): name for name, func in selected_functions.items()}
        for future in futures:
            name, result, runtime = future.result()
            if result is not None:
                results[name] = result
                times[name] = runtime
                print(f"{name} computed in {runtime:.2f} seconds")
    return results, times


# ---------------------------
# Approximate centrality helper functions
# ---------------------------

def approximate_eigenvector_centrality(adj_matrix, max_iter=100, tol=1e-6):
    n = adj_matrix.shape[0]
    x = np.ones(n) / np.sqrt(n)
    for _ in tqdm(range(max_iter), desc="Eigenvector iteration"):
        x_next = adj_matrix.dot(x)
        norm = np.linalg.norm(x_next)
        if norm == 0:
            x = np.zeros(n)
            break
        x_next = x_next / norm
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
    return dict(enumerate(x))


def approximate_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6):
    n = len(graph)
    x = np.ones(n) / n
    adj_matrix = nx.adjacency_matrix(graph)
    out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    out_degrees[out_degrees == 0] = 1
    transition_matrix = adj_matrix.multiply(1 / out_degrees[:, np.newaxis])
    for _ in tqdm(range(max_iter), desc="PageRank iteration"):
        x_next = alpha * transition_matrix.dot(x) + (1 - alpha) * np.ones(n) / n
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
    return dict(enumerate(x))


def approximate_betweenness(graph, k=100, seed=None):
    if seed is not None:
        random.seed(seed)
    betweenness = dict.fromkeys(graph, 0.0)
    nodes = list(graph.nodes())
    sampled_nodes = random.sample(nodes, min(k, len(nodes)))
    for s in tqdm(sampled_nodes, desc="Betweenness sampling"):
        S = []
        P = {}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        D = {}
        Q = [s]
        D[s] = 0
        while Q:
            v = Q.pop(0)
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]
            for w in graph.neighbors(v):
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:
                    sigma[w] += sigmav
                    if w not in P:
                        P[w] = []
                    P[w].append(v)
        delta = dict.fromkeys(nodes, 0)
        while S:
            w = S.pop()
            if w in P:
                for v in P[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    scale = len(sampled_nodes) * (len(nodes) - 1)
    for v in betweenness:
        betweenness[v] /= scale
    return betweenness


def approximate_closeness_centrality(graph, k=100, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(graph)
    nodes = list(graph.nodes())
    sample_sources = random.sample(nodes, min(k, n))
    dist_matrix = {}
    for s in tqdm(sample_sources, desc="Closeness BFS"):
        distances = nx.single_source_shortest_path_length(graph, s)
        dist_matrix[s] = distances
    closeness = {}
    for v in nodes:
        total_dist = 0
        count = 0
        for s in sample_sources:
            d = dist_matrix[s].get(v, n)
            total_dist += d
            count += 1
        if total_dist > 0:
            closeness[v] = (count - 1) / total_dist
        else:
            closeness[v] = 0.0
    return closeness

def calculate_centralities(
        graph,
        alpha=0.85,
        max_iter=100,
        tol=1e-6,
        k_for_approx=100,  # ignored, kept for API compatibility
        seed=42,           # ignored, kept for API compatibility
        n_jobs=-1,
        methods=None
):
    """
    Compute exact centralities for the given graph using NetworkX built-in methods,
    and measure the runtime of each centrality computation.

    Parameters:
      graph: a NetworkX graph.
      alpha: damping factor for PageRank.
      max_iter: maximum iterations for iterative methods.
      tol: tolerance for convergence.
      k_for_approx: ignored for exact methods (kept for API compatibility).
      seed: ignored for exact methods (kept for API compatibility).
      n_jobs: number of threads to run in parallel (if n_jobs > 0).
      methods: list of centrality methods to compute
               (default: ["Degree", "Eigenvector", "PageRank", "Betweenness", "Closeness"]).

    Returns:
      A tuple (results, times) where:
         results: a dict mapping centrality name to its computed centrality dictionary.
         times: a dict mapping centrality name to its computation runtime in seconds.
    """
    if methods is None:
        methods = ["Degree", "Eigenvector", "PageRank", "Betweenness", "Closeness"]
    elif isinstance(methods, str):
        methods = [methods]

    results = {}
    times = {}

    centrality_functions = {
        "Degree": lambda: {node: deg / float(graph.number_of_nodes() - 1) for node, deg in graph.degree()},
        "Eigenvector": lambda: nx.eigenvector_centrality(graph, max_iter=max_iter, tol=tol),
        "PageRank": lambda: nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol),
        "Betweenness": lambda: nx.betweenness_centrality(graph),
        "Closeness": lambda: nx.closeness_centrality(graph)
    }

    selected_functions = {name: func for name, func in centrality_functions.items() if name in methods}

    def run_and_time(name, func):
        start = time.time()
        result = func()
        end = time.time()
        return name, result, end - start

    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        futures = {executor.submit(run_and_time, name, func): name for name, func in selected_functions.items()}
        for future in futures:
            name, result, runtime = future.result()
            if result is not None:
                results[name] = result
                times[name] = runtime
                print(f"{name} computed in {runtime:.2f} seconds")
    return results, times

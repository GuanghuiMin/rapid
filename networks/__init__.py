from logging import raiseExceptions

import networkx as nx
import matplotlib.pyplot as plt
import gzip
import numpy as np
import os
import pandas as pd
import tarfile

def generate_powerlaw_graph(n, avg_degree, exponent=2.5, seed=None, max_degree=None):
    if seed is not None:
        np.random.seed(seed)

    raw_degrees = np.random.pareto(exponent - 1, n) + 1
    raw_degrees = raw_degrees / np.mean(raw_degrees)
    degrees = np.floor(raw_degrees * avg_degree).astype(int)
    degrees = np.array([max(1, d) for d in degrees])

    if max_degree is not None:
        degrees = np.minimum(degrees, max_degree)

    current_avg = np.mean(degrees)
    scale = avg_degree / current_avg
    degrees = np.floor(degrees * scale).astype(int)
    degrees = np.array([max(1, d) for d in degrees])

    if np.sum(degrees) % 2 != 0:
        degrees[0] += 1

    G = nx.configuration_model(degrees, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def generate_graph(graph_type, nodes=1000, avg_degree=10, rewiring_prob=0.1, seed=10, file_path=None, directed="symmetric"):
    if graph_type == "ba":
        m = max(1, avg_degree // 2)
        graph = nx.barabasi_albert_graph(nodes, m, seed=seed)

    elif graph_type == "er":
        p = avg_degree / (nodes - 1)
        graph = nx.erdos_renyi_graph(nodes, p=p, seed=seed)

    elif graph_type == "powerlaw":
        graph = generate_powerlaw_graph(nodes, avg_degree, seed=seed)

    elif graph_type == "smallworld":
        k = max(2, avg_degree // 2 * 2)  # must be even
        graph = nx.watts_strogatz_graph(nodes, k=k, p=rewiring_prob, seed=seed)

    elif graph_type == "grid":
        side = int(np.sqrt(nodes))
        graph = nx.grid_2d_graph(side, side)
        graph = nx.convert_node_labels_to_integers(graph)

    elif graph_type == "tree":
        graph = nx.random_tree(nodes, seed=seed)

    elif graph_type == "barbell":
        graph = nx.barbell_graph(nodes // 2, max(1, nodes // 10))

    elif graph_type == "complete":
        graph = nx.complete_graph(nodes)

    elif graph_type == "sbm":
        num_blocks = 3
        sizes = [nodes // num_blocks] * num_blocks
        p_in = avg_degree / nodes
        p = [[p_in if i == j else 0.01 * p_in for j in range(num_blocks)] for i in range(num_blocks)]
        graph = nx.stochastic_block_model(sizes, p, seed=seed)

    else:
        raise ValueError("Invalid graph type or missing parameters.")

    # Convert to directed graph if specified
    if directed == "symmetric":
        graph = nx.DiGraph(graph)  # All undirected edges → both directions
    elif directed == "asymmetric":
        graph = nx.DiGraph((u, v) for u, v in graph.edges() if np.random.rand() < 0.5)
        graph.add_nodes_from(range(nodes))
    elif directed is None:
        pass  # leave as is
    else:
        raise ValueError("directed must be one of 'symmetric', 'asymmetric', or None")

    # Report statistics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_deg = num_edges / num_nodes
    print(f"[GRAPH] Type={graph_type}, Directed={directed}, Nodes={num_nodes}, Edges={num_edges}, Avg Degree={avg_deg:.2f}")

    return graph


def load_real_data(data_dir, file_name, deezer_subgraph='HR'):
    """
    Load graph data from various formats, including gemsec_deezer_dataset.tar.gz.

    Args:
        data_dir: Directory path.
        file_name: File name to load.
        deezer_subgraph: 'HR', 'HU', or 'RO' (used only when reading gemsec Deezer).

    Returns:
        G: networkx Graph.
    """
    file_path = os.path.join(data_dir, file_name)
    print(f"Loading dataset from {data_dir}/{file_name}...")

    # Handle Deezer tar.gz package
    if file_name == 'gemsec_deezer_dataset.tar.gz':
        edge_file_name = f"{deezer_subgraph}_edges.csv"
        with tarfile.open(file_path, 'r:gz') as tar:
            members = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}
            if edge_file_name not in members:
                raise ValueError(f"Subgraph file '{edge_file_name}' not found in the archive.")

            print(f"Extracting subgraph '{edge_file_name}'...")
            f = tar.extractfile(members[edge_file_name])
            df = pd.read_csv(f, header=None, names=['source', 'target'])

            def clean_node_id(x):
                if isinstance(x, str) and x.startswith('node_'):
                    return int(x.replace('node_', ''))
                else:
                    return int(x)

            G = nx.DiGraph()
            for _, row in df.iterrows():
                u = clean_node_id(row['source'])
                v = clean_node_id(row['target'])
                G.add_edge(u, v)

        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")
        print(f"Subgraph '{deezer_subgraph}' loaded with nodes:", G.number_of_nodes(),
              "edges:", G.number_of_edges(),
              "avg degree:", round(G.number_of_edges() / G.number_of_nodes(), 2))
        return G

    # Handle TSV files
    if file_name.endswith(".tsv") or file_name.endswith(".tsv.gz"):
        if file_name.endswith(".gz"):
            df = pd.read_csv(file_path, sep="\t", compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
        print("TSV file loaded. Data preview:")
        print(df.head())

        G = nx.DiGraph()
        if 'RID' in df.columns and 'SOURCES' in df.columns:
            for rid in df['RID'].unique():
                try:
                    G.add_node(int(rid))
                except Exception:
                    G.add_node(rid)
            for idx, row in df.iterrows():
                rid = row['RID']
                src = row['SOURCES']
                if pd.notna(src) and str(src).strip() != "":
                    try:
                        src_int = int(src)
                        rid_int = int(rid)
                        G.add_edge(src_int, rid_int)
                    except Exception as e:
                        print(e)

        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")
        print("Dataset loaded from TSV with nodes:", G.number_of_nodes(),
              "edges:", G.number_of_edges(),
              "avg degree:", round(G.number_of_edges() / G.number_of_nodes(), 2))
        return G

    # Handle other .gz or .txt files
    G = nx.DiGraph()
    with gzip.open(file_path, 'rt') as f:
        if file_name == "com-friendster.top5000.cmty.txt.gz":
            for line in f:
                nodes = list(map(int, line.strip().split()))
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        G.add_edge(nodes[i], nodes[j])
                        G.add_edge(nodes[j], nodes[i])
        elif file_name == "gplus_combined.txt.gz":
            for line in f:
                if line.startswith('#'):
                    continue
                source, target = map(int, line.strip().split())
                G.add_edge(source, target)
        elif file_name in ["soc-sign-bitcoinalpha.csv.gz", "soc-sign-bitcoinotc.csv.gz"]:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                source, target = int(parts[0]), int(parts[1])
                G.add_edge(source, target)
        else:
            for line in f:
                if line.startswith('#'):
                    continue
                source, target = map(int, line.strip().split())
                G.add_edge(source, target)

    G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")
    print("Dataset loaded from text/gzip file with nodes:", G.number_of_nodes(),
          "edges:", G.number_of_edges(),
          "avg degree:", round(G.number_of_edges() / G.number_of_nodes(), 2))
    return G


def visualize_graph(graph, title="Directed Graph", nodes_to_draw=500):
    subgraph = graph.subgraph(list(graph.nodes)[:nodes_to_draw])
    plt.figure(figsize=(8, 8))
    nx.draw(subgraph, node_size=10, edge_color='gray', with_labels=False)
    plt.title(f"{title} ({nodes_to_draw} Nodes Subgraph)")
    plt.show()


if __name__ == '__main__':
    graph3 = generate_graph("smallworld", nodes=2000, avg_degree=4, seed=10)
    visualize_graph(graph3, title="Smallworld Directed Graph")

    data_dir = "./data"
    file_name = "hiv-Trans.tsv"
    real_graph = load_real_data(data_dir, file_name)
    visualize_graph(real_graph, title="AIDS Transmission Network", nodes_to_draw=200)
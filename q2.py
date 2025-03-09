import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Function to generate and analyze Erdős-Rényi networks
def generate_er_network(n, p, runs=30):
    degrees = []
    clustering_coeffs = []
    path_lengths = []

    for _ in range(runs):
        G = nx.erdos_renyi_graph(n, p)

        # Average degree
        avg_degree = sum(dict(G.degree()).values()) / n
        degrees.append(avg_degree)

        # Average clustering coefficient
        clustering_coeff = nx.average_clustering(G)
        clustering_coeffs.append(clustering_coeff)

        # Average path length (only if the graph is connected)
        if nx.is_connected(G):
            path_length = nx.average_shortest_path_length(G)
            path_lengths.append(path_length)
        else:
            path_lengths.append(float('inf'))

    avg_degree = np.mean(degrees)
    avg_clustering = np.mean(clustering_coeffs)
    avg_path_length = np.mean([x for x in path_lengths if x != float('inf')])

    # Degree distribution
    degree_sequence = [d for n, d in G.degree()]
    degree_counts = np.bincount(degree_sequence)
    degrees = np.arange(len(degree_counts))

    plt.figure(figsize=(8, 5))
    plt.bar(degrees, degree_counts / sum(degree_counts), width=0.8, color='steelblue')
    plt.title(f"Degree Distribution (n={n}, p={p})")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

    return avg_degree, avg_clustering, avg_path_length

# Test configurations
configs = [
    (7 * 10**6, 10**-2),
    (1000, 0.01),
    (5000, 0.002)
]

results = {}

for n, p in configs:
    print(f"Running configuration: n={n}, p={p}")
    avg_degree, avg_clustering, avg_path_length = generate_er_network(n, p)
    results[(n, p)] = (avg_degree, avg_clustering, avg_path_length)

# Report results
for (n, p), (avg_degree, avg_clustering, avg_path_length) in results.items():
    print(f"\nConfiguration: n={n}, p={p}")
    print(f"Average Degree: {avg_degree:.4f} (Expected: {n * p:.4f})")
    print(f"Average Clustering Coefficient: {avg_clustering:.4f} (Expected: {p:.4f})")
    if avg_path_length:
        expected_path_length = np.log(n) / (n * p)
        print(f"Average Path Length: {avg_path_length:.4f} (Expected: {expected_path_length:.4f})")

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Values of p (rewiring probability)
p_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Parameters for the Watts-Strogatz graph
n = 100  # Number of nodes
k = 4    # Each node connected to k nearest neighbors

# Initialize lists to store CC and APL values
clustering_coeffs = []
path_lengths = []

# Generate networks and calculate properties
for p in p_values:
    G = nx.watts_strogatz_graph(n, k, p)
    clustering_coeffs.append(nx.average_clustering(G))
    try:
        path_lengths.append(nx.average_shortest_path_length(G))
    except nx.NetworkXError:
        # Handle disconnected graphs when p = 1
        path_lengths.append(np.nan)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(p_values, clustering_coeffs, label='Clustering Coefficient (CC)', marker='o')
plt.plot(p_values, path_lengths, label='Average Path Length (APL)', marker='s')
plt.xlabel('Proportion of Rewired Edges (p)')
plt.ylabel('Value')
plt.title('Change in CC and APL vs Proportion of Rewired Edges')
plt.legend()
plt.grid(True)
plt.show()

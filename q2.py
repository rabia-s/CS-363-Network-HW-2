import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to generate an E-R network and calculate properties
def erdos_renyi_network(n, p, repetitions=30):
    avg_degrees = []
    avg_clustering_coeffs = []
    avg_path_lengths = []
    degree_distributions = []
    
    for _ in range(repetitions):
        # Generate the E-R graph
        G = nx.erdos_renyi_graph(n, p)
        
        # Degree-related metrics
        degrees = [deg for _, deg in G.degree()]
        avg_degrees.append(np.mean(degrees))
        degree_distributions.extend(degrees)
        
        # Calculate properties if the graph is connected
        if nx.is_connected(G):
            avg_path_lengths.append(nx.average_shortest_path_length(G))
        else:
            avg_path_lengths.append(float('inf'))  # Mark graphs that aren't connected
            
        avg_clustering_coeffs.append(nx.average_clustering(G))
    
    # Aggregate results
    return {
        "avg_degree": np.mean(avg_degrees),
        "avg_clustering_coeff": np.mean(avg_clustering_coeffs),
        "avg_path_length": np.mean([x for x in avg_path_lengths if x != float('inf')]),  # Exclude infinite cases
        "degree_distribution": degree_distributions
    }

# Plot function for degree distribution
def plot_degree_distribution(degree_distributions, n, p):
    sns.histplot(degree_distributions, kde=True, bins=30, color="blue", alpha=0.7)
    plt.title(f"Degree Distribution\n(n={n}, p={p})")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

# Input configurations (n, p)
configs = [
    (100, 0.1),
    (1000, 0.01),       # Small network
    # (10000, 0.005),     # Medium network
    # (7 * 10**6, 0.01),  # Large network
]

# Run experiments for each configuration
results = []
for n, p in configs:
    result = erdos_renyi_network(n, p)
    results.append((n, p, result))
    
    # Print results
    print(f"Configuration (n={n}, p={p}):")
    print(f" - Average degree: {result['avg_degree']}")
    print(f" - Average clustering coefficient: {result['avg_clustering_coeff']}")
    print(f" - Average path length: {result['avg_path_length']}")
    
    # Make plots
    plot_degree_distribution(result['degree_distribution'], n, p)

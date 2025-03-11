import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

def create_ring_lattice(n, k):
    """
    Create a ring lattice with n nodes, each connected to k nearest neighbors.
    Parameters:
    n (int): Number of nodes
    k (int): Number of nearest neighbors to connect to (must be even)
    
    Returns:
    G (networkx.Graph): Ring lattice graph
    """
    if k % 2 != 0:
        raise ValueError("k must be even")
    
    G = nx.Graph()
    # Add nodes
    for i in range(n):
        G.add_node(i)
    
    # Add edges
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n)
            G.add_edge(i, (i - j) % n)
    
    return G

def rewire_edges(G, p):
    """
    Rewire edges of a graph G with probability p using the Watts-Strogatz model.
    
    Parameters:
    G (networkx.Graph): Original graph
    p (float): Probability of rewiring an edge
    
    Returns:
    G_new (networkx.Graph): Graph with rewired edges
    """
    G_new = G.copy()
    nodes = list(G_new.nodes())
    n = len(nodes)
    
    # For each node
    for u in nodes:
        # For each of its neighbors (considering only one direction to avoid double rewiring)
        neighbors = list(G_new.neighbors(u))
        for v in neighbors:
            if u < v:  # Consider each edge only once
                # Rewire with probability p
                if np.random.random() < p:
                    # Remove the edge
                    G_new.remove_edge(u, v)
                    
                    # Find a new target node different from u and not already connected to u
                    possible_targets = [node for node in nodes if node != u and not G_new.has_edge(u, node)]
                    
                    if possible_targets:
                        # Select a random target
                        w = np.random.choice(possible_targets)
                        # Add the new edge
                        G_new.add_edge(u, w)
                    else:
                        # If no valid target is found, restore the original edge
                        G_new.add_edge(u, v)
    
    return G_new

def calculate_metrics(G):
    """
    Calculate global clustering coefficient and average path length for a graph.
    
    Parameters:
    G (networkx.Graph): The graph
    
    Returns:
    cc (float): Global clustering coefficient
    apl (float): Average path length
    """
    # Calculate global clustering coefficient
    cc = nx.average_clustering(G)
    
    # Calculate average path length (only if the graph is connected)
    try:
        apl = nx.average_shortest_path_length(G)
    except nx.NetworkXError:  # Graph is not connected
        # Use the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_graph = G.subgraph(largest_cc)
        apl = nx.average_shortest_path_length(largest_cc_graph)
    
    return cc, apl

def run_small_world_experiment(n=100, k=4, p_values=None):
    """
    Run the small-world network experiment for various rewiring probabilities.
    
    Parameters:
    n (int): Number of nodes
    k (int): Number of nearest neighbors (must be even)
    p_values (list): List of rewiring probabilities to test
    
    Returns:
    results (dict): Dictionary containing the results
    """
    if p_values is None:
        p_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    results = {
        'p_values': p_values,
        'clustering_coefficients': [],
        'average_path_lengths': []
    }
    
    # Create initial ring lattice
    print(f"Creating ring lattice with {n} nodes and {k} neighbors...")
    G_initial = create_ring_lattice(n, k)
    
    # Calculate metrics for different rewiring probabilities
    for p in p_values:
        print(f"Processing p = {p}...")
        start_time = time.time()
        
        # Rewire the network
        G_rewired = rewire_edges(G_initial, p)
        
        # Calculate metrics
        cc, apl = calculate_metrics(G_rewired)
        
        # Store results
        results['clustering_coefficients'].append(cc)
        results['average_path_lengths'].append(apl)
        
        print(f"  Clustering Coefficient: {cc:.4f}")
        print(f"  Average Path Length: {apl:.4f}")
        print(f"  Time: {time.time() - start_time:.2f} seconds")
    
    return results

def normalize_metric(values):
    """Normalize metrics to [0,1] range for plotting"""
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for val in values]

def plot_results(results):
    """
    Plot the results of the small-world experiment.
    
    Parameters:
    results (dict): Dictionary containing the results
    """
    p_values = results['p_values']
    cc_values = results['clustering_coefficients']
    apl_values = results['average_path_lengths']
    
    # Normalize values for better visualization
    cc_normalized = normalize_metric(cc_values)
    apl_normalized = normalize_metric(apl_values)
    
    plt.figure(figsize=(10, 6))
    
    # Plot original values
    plt.subplot(1, 2, 1)
    plt.plot(p_values, cc_values, 'o-', color='blue', label='Clustering Coefficient')
    plt.plot(p_values, apl_values, 'o-', color='red', label='Average Path Length')
    plt.xlabel('Rewiring Probability (p)')
    plt.ylabel('Metric Value')
    plt.title('Small-World Network Metrics')
    plt.grid(True)
    plt.legend()
    
    # Plot normalized values
    plt.subplot(1, 2, 2)
    plt.plot(p_values, cc_normalized, 'o-', color='blue', label='Clustering Coefficient (normalized)')
    plt.plot(p_values, apl_normalized, 'o-', color='red', label='Average Path Length (normalized)')
    plt.xlabel('Rewiring Probability (p)')
    plt.ylabel('Normalized Value')
    plt.title('Normalized Small-World Network Metrics')
    plt.grid(True)
    plt.legend()
    
    # Highlight the small-world region (approximate)
    small_world_start = 0.01
    small_world_end = 0.1
    if small_world_start in p_values and small_world_end in p_values:
        idx_start = p_values.index(small_world_start)
        idx_end = p_values.index(small_world_end)
        plt.axvspan(small_world_start, small_world_end, alpha=0.2, color='green')
        plt.text((small_world_start + small_world_end)/2, 0.5, 'Small-world\nregion',
                 horizontalalignment='center', color='green')
    
    plt.tight_layout()
    plt.savefig('small_world_network_metrics.png', dpi=300)
    plt.show()

def visualize_networks(n=20, k=4, p_values=None):
    """
    Visualize networks with different rewiring probabilities.
    
    Parameters:
    n (int): Number of nodes (smaller for better visualization)
    k (int): Number of nearest neighbors
    p_values (list): List of rewiring probabilities to visualize
    """
    if p_values is None:
        p_values = [0, 0.1, 0.5, 1.0]
    
    plt.figure(figsize=(15, 4))
    
    # Create initial ring lattice
    G_initial = create_ring_lattice(n, k)
    
    for i, p in enumerate(p_values):
        plt.subplot(1, len(p_values), i + 1)
        
        # Rewire the network
        G_rewired = rewire_edges(G_initial, p)
        
        # Calculate metrics
        cc, apl = calculate_metrics(G_rewired)
        
        # Position nodes in a circle
        pos = nx.circular_layout(G_rewired)
        
        # Draw the network
        nx.draw(G_rewired, pos, with_labels=True, node_color='skyblue', 
                node_size=500, font_size=10, font_weight='bold', 
                edge_color='gray', width=1.0)
        
        plt.title(f'p = {p}\nCC = {cc:.3f}, APL = {apl:.3f}')
    
    plt.tight_layout()
    plt.savefig('small_world_network_visualization.png', dpi=300)
    plt.show()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n = 100  # Number of nodes
    k = 4    # Number of neighbors (must be even)
    p_values = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # Rewiring probabilities
    
    # Run experiment
    print(f"Running small-world experiment with {len(p_values)} p-values...")
    results = run_small_world_experiment(n, k, p_values)
    
    # Plot results
    print("Plotting results...")
    plot_results(results)
    
    # Visualize networks with fewer nodes for clarity
    print("Visualizing example networks...")
    visualize_networks(n=20, k=4, p_values=[0, 0.1, 0.5, 1.0])
    
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
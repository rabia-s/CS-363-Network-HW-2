import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings

# Suppress warnings about graph connectivity
warnings.filterwarnings("ignore", message="Graph is not connected.")

def generate_er_network(n, p):
    """Generate an Erdős–Rényi random graph G(n, p)"""
    return nx.erdos_renyi_graph(n, p)

def measure_network_properties(G):
    """Measure various properties of the network"""
    # Average degree
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    
    # Average clustering coefficient (local)
    avg_clustering = nx.average_clustering(G)
    
    # Degree distribution
    degree_counts = {}
    for d in degrees:
        if d in degree_counts:
            degree_counts[d] += 1
        else:
            degree_counts[d] = 1
    
    # Convert to probability distribution
    degree_dist = {k: v/len(degrees) for k, v in degree_counts.items()}
    
    try:
        # For average path length, use largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc).copy()
            avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            avg_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        # Graph is not connected
        avg_path_length = float('inf')
    
    return {
        'avg_degree': avg_degree,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path_length,
        'degree_dist': degree_dist
    }

def theoretical_predictions(n, p):
    """Calculate the theoretical predictions for an ER network with parameters n and p"""
    # Average degree
    theo_avg_degree = p * (n - 1)
    
    # Average clustering coefficient
    theo_avg_clustering = p
    
    # Average path length (approximate formula, valid for connected ER graphs)
    # ln(n) / ln(np) is a common approximation
    if p > np.log(n) / n:  # Assuming connected graph
        theo_avg_path_length = np.log(n) / np.log(np.maximum(1.01, n * p))
    else:
        theo_avg_path_length = float('inf')  # Not guaranteed to be connected
        
    return {
        'theo_avg_degree': theo_avg_degree,
        'theo_avg_clustering': theo_avg_clustering,
        'theo_avg_path_length': theo_avg_path_length
    }

def run_experiments(configurations, num_runs=30):
    """Run experiments for multiple configurations and average the results"""
    results = {}
    
    for config_name, (n, p) in configurations.items():
        print(f"\nRunning configuration {config_name}: n={n}, p={p}")
        config_results = {
            'avg_degree': [],
            'avg_clustering': [],
            'avg_path_length': [],
            'degree_dists': []
        }
        
        # Get theoretical predictions
        theo = theoretical_predictions(n, p)
        
        # Run multiple experiments
        for i in tqdm(range(num_runs), desc=f"Running {num_runs} experiments"):
            start_time = time.time()
            
            # Generate network
            G = generate_er_network(n, p)
            
            # Measure properties
            props = measure_network_properties(G)
            
            # Store results
            config_results['avg_degree'].append(props['avg_degree'])
            config_results['avg_clustering'].append(props['avg_clustering'])
            config_results['avg_path_length'].append(props['avg_path_length'])
            config_results['degree_dists'].append(props['degree_dist'])
            
            end_time = time.time()
            if i == 0:
                print(f"  Sample run completed in {end_time - start_time:.2f} seconds")
        
        # Average the results
        avg_results = {
            'n': n,
            'p': p,
            'avg_degree': np.mean(config_results['avg_degree']),
            'avg_clustering': np.mean(config_results['avg_clustering']),
            'avg_path_length': np.mean([pl for pl in config_results['avg_path_length'] if pl != float('inf')]),
            'degree_dists': config_results['degree_dists'],
            'theo_avg_degree': theo['theo_avg_degree'],
            'theo_avg_clustering': theo['theo_avg_clustering'],
            'theo_avg_path_length': theo['theo_avg_path_length']
        }
        
        results[config_name] = avg_results
        
        # Print results for this configuration
        print(f"\nResults for configuration {config_name}:")
        print(f"  Average degree: {avg_results['avg_degree']:.4f} (Theoretical: {theo['theo_avg_degree']:.4f})")
        print(f"  Average clustering: {avg_results['avg_clustering']:.4f} (Theoretical: {theo['theo_avg_clustering']:.4f})")
        print(f"  Average path length: {avg_results['avg_path_length']:.4f} (Theoretical: {theo['theo_avg_path_length']:.4f})")
    
    return results

def plot_degree_distributions(results):
    """Plot the degree distributions for each configuration"""
    fig, axs = plt.subplots(len(results), 1, figsize=(10, 5 * len(results)))
    if len(results) == 1:
        axs = [axs]
    
    for i, (config_name, result) in enumerate(results.items()):
        # Average the degree distributions
        avg_dist = {}
        for dist in result['degree_dists']:
            for k, v in dist.items():
                if k in avg_dist:
                    avg_dist[k] += v
                else:
                    avg_dist[k] = v
        
        avg_dist = {k: v / len(result['degree_dists']) for k, v in avg_dist.items()}
        
        # Sort by degree
        degrees = sorted(avg_dist.keys())
        probs = [avg_dist[k] for k in degrees]
        
        # Plot
        axs[i].bar(degrees, probs, alpha=0.7)
        axs[i].axvline(x=result['avg_degree'], color='r', linestyle='--', 
                      label=f"Measured Avg: {result['avg_degree']:.2f}")
        axs[i].axvline(x=result['theo_avg_degree'], color='g', linestyle='--', 
                      label=f"Theoretical Avg: {result['theo_avg_degree']:.2f}")
        
        # Poisson distribution for comparison
        if max(degrees) < 100:  # Only plot for reasonable degree ranges
            lambda_param = result['theo_avg_degree']
            x = np.arange(0, max(degrees) + 1)
            # Use scipy.special.factorial for factorial calculation
            from scipy import special
            poisson_pmf = np.exp(-lambda_param) * np.power(lambda_param, x) / special.factorial(x)
            axs[i].plot(x, poisson_pmf, 'k-', label='Poisson PMF')
        
        axs[i].set_title(f"Degree Distribution for {config_name} (n={result['n']}, p={result['p']})")
        axs[i].set_xlabel("Degree")
        axs[i].set_ylabel("Probability")
        axs[i].legend()
        
        # Set reasonable x-axis limits
        mean_degree = result['avg_degree']
        axs[i].set_xlim(max(0, mean_degree - 5 * np.sqrt(mean_degree)), 
                       mean_degree + 5 * np.sqrt(mean_degree))
    
    plt.tight_layout()
    plt.savefig('degree_distributions.png')
    plt.show()

def print_comparison_table(results):
    """Print a table comparing measured and theoretical values"""
    print("\n" + "=" * 80)
    print(f"{'Configuration':<15} | {'Property':<20} | {'Measured':<15} | {'Theoretical':<15} | {'Diff %':<10}")
    print("=" * 80)
    
    for config_name, result in results.items():
        properties = [
            ('Average Degree', result['avg_degree'], result['theo_avg_degree']),
            ('Clustering Coeff', result['avg_clustering'], result['theo_avg_clustering']),
            ('Avg Path Length', result['avg_path_length'], result['theo_avg_path_length'])
        ]
        
        for i, (prop_name, measured, theoretical) in enumerate(properties):
            if theoretical == float('inf'):
                diff_percent = "N/A"
            else:
                diff_percent = f"{100 * abs(measured - theoretical) / max(1e-10, theoretical):.2f}%"
            
            print(f"{config_name if i == 0 else '':<15} | {prop_name:<20} | {measured:<15.4f} | {theoretical:<15.4f} | {diff_percent:<10}")
        
        print("-" * 80)

def main():
    # Define simplified configurations (n, p)
    configurations = {
        'Small Dense': (100, 0.1),        # Small, dense network
        'Medium Sparse': (500, 0.05),     # Medium, moderately sparse network
        'Large Sparse': (1000, 0.01)      # Larger, sparse network
    }
    
    # Reduced number of runs to speed up execution
    num_runs = 10
    
    # Run experiments
    results = run_experiments(configurations, num_runs)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Plot degree distributions
    plot_degree_distributions(results)

if __name__ == "__main__":
    main()
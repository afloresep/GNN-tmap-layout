import pandas as pd
import numpy as np
import tmap as tm
from mhfp.encoder import MHFPEncoder
import igraph as ig
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import math
import os

# Load and preprocess data
df = pd.read_csv('dataset.csv')
df.set_index(df.index, inplace=True)  # Ensure the index is set correctly

# Create MHFP fingerprints
perm = 512
enc = MHFPEncoder(perm)
fingerprints = [tm.VectorUint(enc.encode(s)) for s in df['canonical_smiles'].unique()]

# Create LSH Forest
lsh_forest = tm.LSHForest(d=128, l=8, store=True, file_backed=False, weighted=False)
lsh_forest.batch_add(fingerprints)
lsh_forest.index()

# Get the KNN graph
from_nodes = tm.VectorUint()
to_nodes = tm.VectorUint()
weights = tm.VectorFloat()
lsh_forest.get_knn_graph(from_nodes, to_nodes, weights, k=3, kc=3)

# Convert to numpy arrays
knng_from = np.array(from_nodes)
knng_to = np.array(to_nodes)
weights = np.array(weights)

def create_graph_and_detect_communities(from_nodes, to_nodes, weights):
    edges = list(zip(from_nodes, to_nodes))
    unique_nodes = set(from_nodes) | set(to_nodes)
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    g = ig.Graph()
    g.add_vertices(len(unique_nodes))
    g.add_edges([(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edges])
    g.es['weight'] = weights

    # Perform Leiden community detection
    communities = g.community_leiden(
        objective_function='CPM',
        weights='weight',
        resolution_parameter=0.02,
        beta=0.01,
        n_iterations=500
    )

    return g, communities, reverse_mapping

def plot_graph(g, filename, communities=None):
    layout = g.layout_fruchterman_reingold()

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot edges
    edge_xs, edge_ys = [], []
    for edge in g.es:
        x1, y1 = layout[edge.source]
        x2, y2 = layout[edge.target]
        edge_xs.extend([x1, x2, None])
        edge_ys.extend([y1, y2, None])
    ax.plot(edge_xs, edge_ys, color='lightgray', alpha=0.5, zorder=1)
    
    # Plot nodes
    if communities:
        num_communities = len(set(communities.membership))
        color_map = plt.cm.get_cmap('tab20')
        colors = [color_map(i / num_communities) for i in communities.membership]
    else:
        colors = 'skyblue'
    
    ax.scatter([coord[0] for coord in layout], [coord[1] for coord in layout], 
               c=colors, s=20, zorder=2)
    
    ax.axis('off')
    # plt.tight_layout()
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.close()

def plot_community_molecules(community, df, reverse_mapping, filename, mols_per_row=5):
    # Get the original node identifiers for the community
    original_nodes = [reverse_mapping.get(node, node) for node in community]
    
    # Filter the DataFrame to get only the molecules in this community
    community_smiles = df[df.index.isin(original_nodes)]['canonical_smiles']
    
    print(f"Number of molecules in community: {len(community_smiles)}")
    
    if len(community_smiles) == 0:
        print(f"Warning: No molecules found for community. Check DataFrame index and reverse_mapping.")
        return
    
    # Convert SMILES to RDKit molecules
    mols = [Chem.MolFromSmiles(smi) for smi in community_smiles]
    
    # Generate 2D coordinates for each molecule
    for mol in mols:
        AllChem.Compute2DCoords(mol)
    
    # Calculate the grid size
    n_mols = len(mols)
    n_rows = math.ceil(n_mols / mols_per_row)
    
    # Create a matplotlib figure
    fig, axes = plt.subplots(n_rows, mols_per_row, figsize=(4*mols_per_row, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 or mols_per_row > 1 else [axes]
    
    for idx, (mol, ax) in enumerate(zip(mols, axes)):
        img = Draw.MolToImage(mol, size=(300, 300))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Node: {original_nodes[idx]}", fontsize=8)
    
    # Remove any unused subplots
    for idx in range(len(mols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    # Ensure the directory exists

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    
    # # Save the figure
    # try:
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')
    #     print(f"Community visualization saved as '{filename}'")
    # except Exception as e:
    #     print(f"Error saving image: {e}")
    
    # plt.close(fig)

# Create graph and detect communities
graph, communities, reverse_mapping = create_graph_and_detect_communities(knng_from, knng_to, weights)

# Plot original graph
plot_graph(graph, "original_graph.png")

# Plot graph with communities
plot_graph(graph, "graph_with_communities.png", communities)

print("Original graph saved as 'original_graph.png'")
print("Graph with communities saved as 'graph_with_communities.png'")

# Print community information
print(f"Number of communities: {len(communities)}")
print(f"Modularity: {communities.modularity}")

# Extract and display nodes for each community
for idx, community in enumerate(communities):
    original_nodes = [reverse_mapping[node] for node in community]
    print(f"\nCommunity {idx + 1}:")
    print(f"Number of nodes: {len(community)}")
    print(f"Nodes: {original_nodes}")

    # Plot molecules for each community
    plot_community_molecules(community, df, reverse_mapping, f"community_{idx+1}_molecules.png")

# Save the graph with community information
graph.vs['community'] = communities.membership
graph.write_graphml("graph_with_communities.graphml")

print("\nGraph with communities saved as 'graph_with_communities.graphml'")
print("Finished plotting all communities.")



# Find community 5
community_5 = None
for idx, community in enumerate(communities):
    if idx == 3:  # Python uses 0-based indexing, so community 5 is at index 4
        community_5 = community
        break

if community_5 is not None:
    print(f"Plotting community 5:")
    original_nodes = [reverse_mapping[node] for node in community_5]
    print(f"Number of nodes: {len(community_5)}")
    print(f"Nodes: {original_nodes}")

    # Plot molecules for community 5
    plot_community_molecules(community_5, df, reverse_mapping, "community_5_molecules.png")
else:
    print("Community 5 not found. The graph might have fewer communities.")

# Optionally, you can also plot the subgraph for community 5
if community_5 is not None:
    subgraph = graph.subgraph(community_5)
    plot_graph(subgraph, "community_5_graph.png")
    print("Subgraph for community 5 saved as 'community_5_graph.png'")

    
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx

# 1. Create a random graph with 100 nodes
num_nodes = 100
edge_prob = 0.045  # Connection probability
num_classes = 4

torch.manual_seed(42)
np.random.seed(42)

# Generate node features and random category/class labels for records
x = torch.randn((num_nodes, 16))
y = torch.randint(0, num_classes, (num_nodes,))

# Generate random edges between the 100 nodes
edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob, directed=False)
pyg_data = Data(x=x, edge_index=edge_index, y=y)

# 2. Convert PyG Data to NetworkX Graph for plotting
G = to_networkx(pyg_data, to_undirected=True)

# 3. Graph Layout & Visual Attributes
plt.figure(figsize=(14, 10), facecolor="#1a1a2e")
ax = plt.gca()
ax.set_facecolor("#1a1a2e")

# Force-directed spring layout
pos = nx.spring_layout(G, seed=42, k=0.25, iterations=50)

# Calculate node degrees (connections)
degrees = dict(G.degree())
node_sizes = [150 + degrees[i] * 50 for i in G.nodes()]

# Color map for the 4 classes
colors = ["#FF5722", "#00BCD4", "#4CAF50", "#E91E63"]
node_colors = [colors[y[i].item()] for i in G.nodes()]

# 4. Draw Edges
nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.35,
    edge_color="#a0a0c0",
    width=1.2,
    ax=ax,
)

# 5. Draw Nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="white",
    linewidths=1.2,
    ax=ax,
)

# 6. Label hub nodes (top 15 highest-connected records)
hub_nodes = sorted(degrees, key=degrees.get, reverse=True)[:15]
labels = {node: f"R_{node}" for node in hub_nodes}
nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=9,
    font_color="#ffffff",
    font_weight="bold",
    ax=ax,
)

# 7. Legend and Title
for i, color in enumerate(colors):
    ax.scatter([], [], c=color, label=f"Record Class {i}", s=100, edgecolors="white")

legend = ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="#16213e",
    edgecolor="white",
    fontsize=10,
    labelcolor="white",
)

plt.title(
    f"Visual Representation of 100 Nodes & Connected Edges\n"
    f"Total Records: {num_nodes} | Total Edges: {G.number_of_edges()} | Avg Degree: {2 * G.number_of_edges() / num_nodes:.2f}",
    color="white",
    fontsize=15,
    fontweight="bold",
    pad=15,
)

plt.axis("off")
plt.tight_layout()

# Save image file
output_path = "graph_visualization.png"
plt.savefig(output_path, dpi=300, facecolor="#1a1a2e", bbox_inches="tight")
print(f"[SUCCESS] Graph visualization saved to: {output_path}")

# Show interactive visual window
plt.show()

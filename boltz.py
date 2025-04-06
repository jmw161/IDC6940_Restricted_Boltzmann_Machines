import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize graph
num_nodes = 3
G = nx.complete_graph(num_nodes)

# Assign random weights to edges
weights = {edge: np.random.uniform(-1, 1) for edge in G.edges}
nx.set_edge_attributes(G, weights, 'weight')

# Initialize random states for nodes (0 or 1)
states = [np.random.choice([0, 1]) for node in G.nodes]
state_scalars = []

def states_to_scalar(states):
    """Convert list of states to a scalar value."""
    return sum(state * (2 ** i) for i, state in enumerate(states))

def stochastic_update(states, weights):
    """Perform stochastic update of node states."""
    for node in G.nodes:
        # Calculate the weighted sum of neighbors' states
        weighted_sum = sum(weights[(min(node, neighbor), max(node, neighbor))] * states[neighbor] 
                          for neighbor in G.neighbors(node))
        # Update state based on sigmoid probability
        prob = 1 / (1 + np.exp(-weighted_sum))
        states[node] = np.random.choice([0, 1], p=[1-prob, prob])

def plot_graph(G, states, ax, frame):
    """Plot the graph with node states."""
    ax.clear()
    pos = nx.circular_layout(G)
    blue = '#AAAAFF' if frame % 2 == 0 else '#CCCCFF'
    white = '#FFFFFF' if frame % 2 == 0 else '#DDDDDD'
    node_colors = [white if states[node] == 0 else blue for node in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edgecolors='black', node_size=800, ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f'{v:.2f}' for k, v in labels.items()}, ax=ax)

def plot_pmf(state_scalars, ax):
    """Plot the PMF of state scalar values."""
    ax.clear()
    values, counts = np.unique(state_scalars, return_counts=True)
    pmf = counts / counts.sum()
    ax.bar(values, pmf, color='gray', edgecolor='black')
    ax.set_title("PMF of States")
    ax.set_xlabel("State Scalar")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(2 ** len(states)))

fig, ax = plt.subplots()
frames = 10

def update(frame):
    print(frame, states, states_to_scalar(states))
    state_scalars.append(states_to_scalar(states))
    plot_graph(G, states, ax, frame)
    stochastic_update(states, weights)

def init():
    plot_graph(G, states, ax, 0)

ani = FuncAnimation(fig, update, frames=frames, repeat=False, interval=1000, init_func=init)
ani.save('boltz_animation.mp4', writer='ffmpeg')
# plt.show()


# Create animation with two subplots
fig, (ax_graph, ax_pmf) = plt.subplots(1, 2, figsize=(10, 5))
frames = 1000  # Number of iterations

def update(frame):
    print(frame, states, states_to_scalar(states))
    state_scalars.append(states_to_scalar(states))
    plot_graph(G, states, ax_graph, frame)
    plot_pmf(state_scalars, ax_pmf)
    stochastic_update(states, weights)

def init():
    plot_graph(G, states, ax_graph, 0)
    plot_pmf(state_scalars, ax_pmf)

ani = FuncAnimation(fig, update, frames=frames, repeat=False, interval=10, init_func=init)
ani.save('boltz_animation_with_pmf.mp4', writer='ffmpeg')
# plt.show()
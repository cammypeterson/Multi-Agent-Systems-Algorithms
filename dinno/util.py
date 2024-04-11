import networkx as nx
import random


def create_graph(num_nodes, graph_type, target_connectivity=1.0, max_iterations=1000):

    if graph_type == "random":
        G = nx.path_graph(num_nodes)
        current_connectivity = nx.algebraic_connectivity(G)
        if target_connectivity is not None:
            # Generated with ChatGPT
            iteration = 0
            while abs(current_connectivity - target_connectivity) > 0.001 and iteration < max_iterations:
                if current_connectivity < target_connectivity:
                    a, b = sorted(random.sample(range(num_nodes), 2))
                    if not G.has_edge(a, b):
                        G.add_edge(a, b)
                else:
                    edges = list(G.edges())
                    if len(edges) > 1:
                        G.remove_edge(*random.choice(edges))
                current_connectivity = nx.algebraic_connectivity(G)
                iteration += 1
    elif graph_type == "cycle":
        G = nx.cycle_graph(num_nodes)
        current_connectivity = nx.algebraic_connectivity(G)
    elif graph_type == "complete":
        G = nx.complete_graph(num_nodes)
        current_connectivity = nx.algebraic_connectivity(G)

    return G, current_connectivity
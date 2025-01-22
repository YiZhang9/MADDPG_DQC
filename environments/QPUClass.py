
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy




class QPUClass():
    
    def __init__(self):
        self.pos = None #init - the positions of the nodes
        self.G = self.create_DQC_graph()  #here we have two guadelupe QPUs connected
        self.numNodes = len(self.G.nodes)
        self.numEdges = len(self.G.edges)
        self.numEdgesClassic = len(self.G.edges) - 1 # WARNING: only for this specific DQC architecture
        self.numEdgesQuantum = 1 # WARNING: only for this specific DQC architecture
        

    

    # Function to create a Guadelupe-like graph
    def create_graph(self):
        G = nx.Graph()
        # Add nodes
        nodes = [i for i in range(16)]  # We have 16 nodes
        G.add_nodes_from(nodes)
        
        # Add edges 
        edges = [
            (1, 2), (2, 3), (3, 5), (5, 8),
            (8, 11), (11, 14), (14, 13), (13, 12),
            (12, 10), (10, 7), (7, 4), (4, 1),
            # Corners
            (6, 7), (0, 1), (8, 9), (12, 15)
        ]
        G.add_weighted_edges_from((u, v, 0) for u, v in edges)
        for edge in G.edges():
            G.edges[edge]['label'] = "simple"
            G.edges[edge]['mask_tele_qubit'] = True   #initialization of the masks, you can initially perform any action
            G.edges[edge]['mask_swap'] = True
        return G
    
    # Function to create two Guadelupe-like graphs connected with a quantum link
    def create_DQC_graph(self):
            # Create 
        G1 = self.create_graph()
        G2 = self.create_graph()
        # Union the two graphs
        G = nx.disjoint_union(G1, G2)

        for node in G.nodes:
            G.nodes[node]['weight'] = 0  # Set your initial weight value in the nodes! (cooldown)

        # Add edge between node 0 of both graphs (In the union graph, node 0 of second graph is 16)
        G.add_edge(0, 16, weight=0, label="quantum", tele_qubit=False)
        G.edges[(0,16)]['mask_generate'] = True
        numNodes = len(G.nodes)
        numEdges = len(G.edges)

        # The code below just prints the coupling graph 
        # Positions for nodes in a circular pattern
        circle_nodes = [1,2,3,5,8,11,14,13,12,10,7,4]
        angle = np.linspace(0, 2 * np.pi, len(circle_nodes) + 1)[:-1]
        pos1 = {node: (np.cos(a), np.sin(a)) for node, a in zip(circle_nodes, angle)}
        # Positions for the corner nodes
        corner_nodes = [0, 6, 9, 15]
        # Choose angles of corner nodes to be close to their nearest node
        corner_angles = [angle[circle_nodes.index(n)] for n in [1, 7, 8, 12]]
        # Position corner nodes closer to the circle (radius 1.5 instead of 2)
        pos1.update({node: (1.5 * np.cos(a), 1.5 * np.sin(a)) for node, a in zip(corner_nodes, corner_angles)})
        # The same positions for the second graph, shifted to the right and rotated 180 degrees
        rotation = np.pi
        shift = 4  # Increase shift from 3 to 4
        pos2 = {node + 16: (shift + np.cos(a + rotation), np.sin(a + rotation)) for node, a in zip(circle_nodes, angle)}
        pos2.update({node + 16: (shift + 1.5 * np.cos(a + rotation), 1.5 * np.sin(a + rotation)) for node, a in zip(corner_nodes, corner_angles)})
        # Combine the positions
        pos_G = {**pos1, **pos2}
        # Plot the graph
        fig, ax = plt.subplots()
        edges = G.edges()
        colors = ['red' if e in [(0, 16), (16, 0)] else 'black' for e in edges]
        nx.draw(G, pos_G, with_labels=True, ax=ax, edge_color=colors, node_color='lightblue', node_size=300)
        #plt.show()
        self.pos = pos_G
        return G

    
  
    

    
















        
        










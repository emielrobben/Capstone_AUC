import networkx as nx
import matplotlib.pyplot as plt
import random

class RBN:
    def __init__(self, K, N, flip_probability):
        self.K = K  # Number of inputs per node
        self.N = N  # Number of nodes in the network
        self.flip_probability = flip_probability
        self.G = nx.DiGraph()
        self.initialization()
        self.generate_logic_tables()
        
    def initialization(self):
        expected_degrees = [self.K for _ in range(self.N)]
        # Create a random graph with the specified expected degrees
        self.G = nx.expected_degree_graph(expected_degrees, selfloops=False)
        self.G = nx.DiGraph(self.G)
        for u, v in list(self.G.edges()):
            # Randomly reverse some edges with 50% probability
            if random.random() < 0.5:
                self.G.remove_edge(u, v)
                self.G.add_edge(v, u)
        # Initialize the node states with random boolean values      
        for i in range(self.N):
            self.G.nodes[i]["state"] = random.choice([True, False])
        
    def generate_logic_tables(self):
        for i in self.G.nodes:
            inputs = list(self.G.predecessors(i))
            # Generate random truth tables for each node
            truth_table = [random.choice([True, False]) for i in range(2**len(inputs))]
            self.G.nodes[i]["inputs"] = inputs
            self.G.nodes[i]["truth_table"] = truth_table
            
    def bin_to_dec(self, bin_list):
        dec = 0
        # Convert binary list to decimal number
        for i in range(len(bin_list)):
            if bin_list[i]:
                dec = dec + 2**(len(bin_list) - 1 - i)
        return dec
    
    def probabilistic_flip(self, value):
        # Flip the value with the given probability
        if random.random() < self.flip_probability:
            return not value
        return value
            
    def step(self, show_plot=False):
        new_state = []
        for node in self.G.nodes:
            # Get the inputs and truth table from the node
            inputs = self.G.nodes[node]["inputs"]
            truth_table = self.G.nodes[node]["truth_table"]
            # Get the state of the input nodes
            input_vals = [self.G.nodes[input]["state"] for input in inputs]
            # Convert boolean list to decimal index
            index = self.bin_to_dec(input_vals)
            # Get the output from the truth table
            output = truth_table[index]
            # Apply probabilistic flip
            output = self.probabilistic_flip(output)
            new_state.append(output)

        # Update node states with new_state
        for i, state in enumerate(new_state):
            self.G.nodes[i]["state"] = state

    def plot_states(self):
        # Draw the graph with updated states
        color_map = {True: 'blue', False: 'red'}
        colors = [color_map[self.G.nodes[node]["state"]] for node in self.G.nodes()]
        nx.draw(self.G, with_labels=True, node_color=colors, font_weight='bold')
        plt.show()

# Create an instance of the RBN class with 5 inputs per node, 40 nodes, and flip probability of 0.4
network = RBN(4, 10, 0.4)
# Execute 10 steps and display the network states
for i in range(10):
    print([network.G.nodes[i]["state"] for i in range(network.N)])
    network.step()
    network.plot_states()
           

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.sparse import coo_matrix


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
        self.G = nx.expected_degree_graph(expected_degrees, selfloops=False)
        self.G = nx.DiGraph(self.G)
        for u, v in list(self.G.edges()):
            if random.random() < 0.5:
                self.G.remove_edge(u, v)
                self.G.add_edge(v, u)
        for i in range(self.N):
            self.G.nodes[i]["state"] = random.choice([True, False])

    def generate_logic_tables(self):
        for i in self.G.nodes:
            inputs = list(self.G.predecessors(i))
            truth_table = [random.choices([True, False], weights=[self.flip_probability, 1 - self.flip_probability])[0]
                           for _ in range(2 ** len(inputs))]
            self.G.nodes[i]["inputs"] = inputs
            self.G.nodes[i]["truth_table"] = truth_table

    def bin_to_dec(self, bin_list):
        dec = 0
        # Convert binary list to decimal number
        for i in range(len(bin_list)):
            if bin_list[i]:
                dec = dec + 2 ** (len(bin_list) - 1 - i)
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

    def _generate_transitions(self):
        for current_state in range(2 ** self.N):
            for updated_node in range(self.N):
                for updated_state in [0, 1]:
                    inputs = self.G.nodes[updated_node]["inputs"]
                    input_vals = [(current_state >> i) & 1 for i in inputs]
                    index = self.bin_to_dec(input_vals)
                    output = self.G.nodes[updated_node]["truth_table"][index]
                    if output == updated_state:
                        next_state = current_state ^ (
                                    (current_state >> updated_node) & 1 ^ updated_state) << updated_node
                        probability = (1 - self.flip_probability) if output else self.flip_probability
                        yield current_state, next_state, probability

    def create_initial_vector_and_sparse_matrix(self):
        # Step 1: Create initial probability vector
        num_states = 2 ** self.N
        init_prob_vector = np.full(num_states, 1 / num_states)

        # Step 2: Create sparse matrix in COO format
        row = []
        col = []
        data = []
        # Use the _generate_transitions method to create the matrix
        for current_state, next_state, probability in self._generate_transitions():
            row.append(current_state)
            col.append(next_state)
            data.append(probability)

        # Create the sparse matrix using the COO format
        transition_matrix = coo_matrix((data, (row, col)), shape=(num_states, num_states))

        return init_prob_vector, transition_matrix

    def find_stationary_distribution(self, transition_matrix, initial_vector, num_iterations=100):
        # Create a copy of the initial probability vector to avoid modifying the original vector
        current_vector = initial_vector.copy()

        # Perform the power iteration method for the given number of iterations
        for _ in range(num_iterations):
            # Multiply the current vector by the transition matrix
            current_vector = transition_matrix.dot(current_vector)

        # Normalize the resulting vector to ensure it is a probability distribution
        stationary_distribution = current_vector / np.sum(current_vector)

        # Return the stationary distribution
        return stationary_distribution


# Create an instance of the RBN class with 5 inputs per node, 40 nodes, and flip probability of 0.4
k_av = 4
N = 20
flip_prob = 0.0
network = RBN(k_av, N, flip_prob)

# Call the function to create the initial probability vector and sparse matrix
initial_vector, sparse_matrix = network.create_initial_vector_and_sparse_matrix()

# Call the new function to find the stationary distribution (pmf)
pmf = network.find_stationary_distribution(sparse_matrix, initial_vector)

attractor_states = []
for state in range(2 ** N):
    if pmf[state] > 0.001:
        attractor_states.append(state)

# Print the probability of each attractor state
for state in attractor_states:
    binary_state = format(state, f'0{N}b')  # Convert state to binary string with N digits
    probability = pmf[state]
    print(f"Attractor state {binary_state}: probability = {probability}")

    #
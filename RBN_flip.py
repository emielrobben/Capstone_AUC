import networkx as nx
import matplotlib.pyplot as plt
import random
import collections


class RBN:
    def __init__(self, K, N, flip_probability):
        self.K = K  # Number of inputs per node
        self.N = N  # Number of nodes in the network
        self.flip_probability = flip_probability
        self.G = nx.DiGraph()
        self.initialization()
        self.generate_logic_tables()
        self.state_history = []  # To store the history of the states

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
            truth_table = [random.choice([True, False]) for i in range(2 ** len(inputs))]
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
            # Update the state history
            self.state_history.append(tuple(map(int, new_state)))

        # Update node states with new_state
        for i, state in enumerate(new_state):
            self.G.nodes[i]["state"] = state

    def plot_states(self):
        # Draw the graph with updated states
        color_map = {True: 'blue', False: 'red'}
        colors = [color_map[self.G.nodes[node]["state"]] for node in self.G.nodes()]
        nx.draw(self.G, with_labels=True, node_color=colors, font_weight='bold')
        plt.show()

    def find_cycle_start(self, state_history, given_states):
        for i, state in enumerate(state_history):
            if state == given_states:
                cycle_start = i
                break
        else:
            return None

        cycle = []
        for state in state_history[cycle_start:]:
            if state in cycle:
                break
            cycle.append(state)

        return cycle_start + len(cycle)

    def conditional_pmf(self, given_states):
        # Convert given_states to a tuple of integers
        given_states = tuple(map(int, given_states))

        # Find the index where the network starts to jump between attractor states
        cycle_start = self.find_cycle_start(self.state_history, given_states)

        if cycle_start is None:
            return {}  # If the given_states do not appear in the state_history, return an empty dictionary

        # Extract the attractor states from the state_history, starting from the cycle_start index
        attractor_states = self.state_history[cycle_start:]

        # Count the occurrences of each state in the attractor_states
        state_counts = collections.Counter(attractor_states)

        # Calculate the total number of occurrences
        total_occurrences = sum(state_counts.values())

        # Calculate the probability mass function by dividing the count of each state by the total number of occurrences
        pmf = {state: count / total_occurrences for state, count in state_counts.items()}

        return pmf


# Create an instance of the RBN class with 5 inputs per node, 40 nodes, and flip probability of 0.4
network = RBN(3, 5, 0)
# Execute 10 steps and display the network states
for i in range(30):
    print([network.G.nodes[i]["state"] for i in range(network.N)])
    network.step()
    network.plot_states()

initial_state = [network.G.nodes[i]["state"] for i in range(network.N)]
pmf = network.conditional_pmf(initial_state)
print("Probability mass function for attractor states given the initial state:", pmf)
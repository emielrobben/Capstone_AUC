import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.sparse import coo_matrix
from scipy import linalg


class RBN:
    def __init__(self, K, N, r):
        self.K = K  # Number of inputs per node
        self.N = N  # Number of nodes in the network
        self.r = r  # Flip probability
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
            truth_table = [random.choices([True, False], weights=[self.r, 1 - self.r])[0] for _ in
                           range(2 ** len(inputs))]
            self.G.nodes[i]["inputs"] = inputs
            self.G.nodes[i]["truth_table"] = truth_table

    def bin_to_dec(self, bin_list):
        dec = 0
        for i in range(len(bin_list)):
            if bin_list[i]:
                dec = dec + 2 ** (len(bin_list) - 1 - i)
        return dec

    def dec_to_bin(self, dec):
        bin_list = [False] * self.N
        binary_repr = bin(dec)[2:]

        for i in range(len(binary_repr)):
            bin_list[self.N - len(binary_repr) + i] = bool(int(binary_repr[i]))

        return bin_list

    def bin_to_bin_str(self, bin_list):
        bin_str = ''.join(['1' if b else '0' for b in bin_list])
        return bin_str

    def step(self, show_plot=False):
        new_state = []
        for node in self.G.nodes:
            inputs = self.G.nodes[node]["inputs"]
            truth_table = self.G.nodes[node]["truth_table"]
            input_vals = [self.G.nodes[input]["state"] for input in inputs]
            index = self.bin_to_dec(input_vals)
            output = truth_table[index]
            new_state.append(output)

        for i, state in enumerate(new_state):
            self.G.nodes[i]["state"] = state

    def plot_states(self):
        color_map = {True: 'blue', False: 'red'}
        colors = [color_map[self.G.nodes[node]["state"]] for node in self.G.nodes()]
        nx.draw(self.G, with_labels=True, node_color=colors, font_weight='bold')
        plt.show()

    def create_initial_vector_and_sparse_matrix(self):
        num_states = 2 ** (self.N)
        initial_vector = np.ones(num_states) / num_states

        row_indices = np.array([], dtype=int)
        col_indices = np.array([], dtype=int)
        data = np.array([])
        transition_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_states, num_states))

        # for the first state: I initialize my network in that state
        # (I need a dec to bin for that, given the number of nodes I have, and then make a step.
        # I record where I am then using dec to bin. Lets experiment with n=5.

        # just try it once. state ito F,T

        for k in range(num_states):
            Network_state = self.dec_to_bin(k)
            after_step = [False] * self.N
            # now manipulating the nodes. The network is already initialized
            for i in range(self.N):
                self.G.nodes[i]["state"] = Network_state[i]
                # print("before",Network_state[i])
            self.step()
            for i in range(self.N):
                after_step[i] = self.G.nodes[i]["state"]
                # print("äfter", after_step[i])

            # order row/column: https://brilliant.org/wiki/markov-chains/#markov-chain
            # the row indice is the before step, which is simply the index of the initial vector. although it feels weird
            # I changed it now and it works now.
            row_indices = np.append(row_indices, self.bin_to_dec(after_step))
            col_indices = np.append(col_indices, k)
            # print("bin to dec",k, self.bin_to_dec(after_step) )
            data = np.append(data, 1)
            # there is only one data point per column/row. so after this we can go to the next one. but it does that automatically.
            # Create a transition matrix after all steps
        transition_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_states, num_states))
        # print(transition_matrix)
        # I do this for all states. I need to be careful:
        # to be a bit more efficient: I just need to take my "before step" initialization, and only change the nodes that need to changes.

        return initial_vector, transition_matrix

    def find_stationary_distribution(self, initial_vector, transition_matrix, tolerance, num_iterations=100):
        probability_vector = initial_vector
        # now just iterating many times. Let's first just create something that works:

        for i in range(num_iterations):
            # Multiply the current vector by the transition matrix
            probability_vector_i = transition_matrix * probability_vector
            if np.linalg.norm(probability_vector_i - probability_vector) < tolerance:
                probability_vector = probability_vector_i

                return probability_vector

            probability_vector = probability_vector_i

        return probability_vector


# Create an instance of the RBN class with 4 inputs per node, 10 nodes, and r=0.6
N = 10
network = RBN(4, N, 0.6)

# Call the function to create the initial probability vector and sparse matrix
initial_vector, sparse_matrix = network.create_initial_vector_and_sparse_matrix()

# Call the function to find the stationary distribution (pmf) using power iteration
pmf = network.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
# pmf_eig=(network.find_steady_state_eig(sparse_matrix)/(2**N))
p_sum = 0
for i in range(2 ** N):
    if pmf[i] > 0.0001:
        p_sum = p_sum + pmf[i]
        print("p(", network.bin_to_bin_str(network.dec_to_bin(i)), ") :", pmf[i])
print("Probability sum:", p_sum)
# print("Probability mass function (pmf):", pmf)

# in case I want to compare my first method with my eigenvalue method:

# for i in range(2**N):
#    if (pmf_eig[i])> 0.000001:
#        p_sum= p_sum + pmf_eig[i]
#        print("p(",network.bin_to_bin_str(network.dec_to_bin(i)),") :" ,pmf_eig[i])
# print("Probability sum:", p_sum)
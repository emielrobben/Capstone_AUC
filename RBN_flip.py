


import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.sparse import coo_matrix
from scipy import linalg
import math


class RBN:
    def __init__(self, K, N, r):
        self.K = K  # Number of inputs per node
        self.N = N  # Number of nodes in the network
        self.r = r  # Flip probability
        self.G = nx.DiGraph()
        self.initialization()
        self.generate_logic_tables(self.r)

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

    def generate_logic_tables(self, r):
        for i in self.G.nodes:
            inputs = list(self.G.predecessors(i))
            truth_table = [random.choices([True, False], weights=[r, 1 - r])[0] for _ in range(2 ** len(inputs))]
            self.G.nodes[i]["inputs"] = inputs
            self.G.nodes[i]["truth_table"] = truth_table

    # this function is used in Fisher
    def modify_logic_tables(self, increment_prob):
        for i in self.G.nodes:
            truth_table = self.G.nodes[i]["truth_table"]
            for j in range(len(truth_table)):
                if random.random() < increment_prob:
                    truth_table[j] = not truth_table[j]
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

    # Put the Fisher Characteristica here
    # steps:
    # 1. see how to implement the pmf
    # pmf for different r? How does one do that? Just run the RBN different times? that should shange nothing in the code for Markov itself
    # important to run enough times
    # plotting? Is the easy part, already implemented

    def filter_pmf_and_compare(self, pmf, pmf_prev, threshold, num_states):
        filtered_pmf = np.zeros(num_states)
        result = np.zeros(num_states)

        for i in range(num_states):
            log_pmf = 0 if pmf[i] == 0 else math.log(pmf[i])
            log_pmf_prev = 0 if pmf_prev[i] == 0 else math.log(pmf_prev[i])
            log_diff = abs(log_pmf - log_pmf_prev)

            if log_diff >= threshold:
                filtered_pmf[i] = pmf[i]
                result[i] = log_diff

        return result, filtered_pmf

    def Compute_Fisher(self, d_r, num_T):
        """


        Parameters
        ----------
        d_r : float
            the increments of r
        r : float
            the flip probability
        k : TYPE
            DESCRIPTION.
        N : integer
            number of nodes in th network

        Returns
        -------
        F_array : array with the values for different r for F.

        """
        N = self.N
        num_states = 2 ** N
        F = 0
        pmf_stack = np.empty((num_states, 0))
        # pmf_array = np.empty((int(1/d_r) + 1, num_states))
        F_array = np.zeros(int(1 / d_r) + 1)
        last_pmf = np.zeros(num_states)
        F_array[0] = 0
        threshold = 0.001
        # at the beginning, you initialize the network. After this you will never initialize the network again: you will only make small changes to it.
        for r in np.arange(0, 1 + d_r, d_r):
            t = 0

            # for a number of times, a transition matrix is created and pmf calculated.
            # how to determine if I need to do more rounds? Lets just say I will keep it static now.

            for i in range(num_T):
                self.generate_logic_tables(r)
                initial_vector, sparse_matrix = self.create_initial_vector_and_sparse_matrix()
                pmf = self.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
                # add the previous pmf tp the last one (vector addition)
                combined_pmf = pmf + last_pmf
            average_pmf = combined_pmf / num_T
            # for every r, there should be stored an array.
            pmf_stack = np.column_stack((pmf_stack, average_pmf))
            # now, for every column in the pmf_stack, it will be compared to the previous one
        num_columns = pmf_stack.shape[1]
        for column_index in range(1, num_columns):
            column_1 = pmf_stack[:, column_index]
            column_0 = pmf_stack[:, column_index - 1]
            # crucial step: we first had an array with 2 indices. Now we extract one specific column from it.
            result, filtered_pmf = self.filter_pmf_and_compare(column_1, column_0, threshold, num_states)

            for j in range(len(result)):
                result_value = result[j]
                filtered_pmf_value = filtered_pmf[j]

                F += (filtered_pmf_value * ((result_value / d_r) ** 2))
            F_array[t + 1] = F
            F = 0
            t = t + 1

            # then for the fisher calculations,

        return F_array


# Create an instance of the RBN class with 4 inputs per node, 10 nodes, and r=0.6
N = 10
network = RBN(4, N, 0.6)

F_array = network.Compute_Fisher(0.1, 30)
x_values = np.linspace(0, 1, len(F_array))
# how does this work again: can we just call the outout of the
# Plot F_array against the equally spaced values
plt.plot(x_values, F_array, marker='o', linestyle='-')
plt.xlabel('x values')
plt.ylabel('F_array values')
plt.title('F_array values plotted on equal distance between 0 and 1')
plt.grid(True)
plt.show()
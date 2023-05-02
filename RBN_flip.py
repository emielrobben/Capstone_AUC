
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

    def filter_pmf_and_compare(self, pmf, threshold, pmf_prev, n):
        filtered_pmf = []
        result = []

        for idx, value in enumerate(pmf):
            if value > threshold:
                state = self.dec_to_bin(idx)
                filtered_pmf.append((state, value))

        for item in filtered_pmf:
            state, prob = item
            state_dec = self.bin_to_dec(state)

            if state_dec in [x[0] for x in pmf_prev]:
                prev_prob = [x[1] for x in pmf_prev if x[0] == state_dec][0]
                abs_diff = abs(math.log(prob) - math.log(prev_prob))
                result.append((state, abs_diff))
            else:
                result.append((state, math.log(prob)))

        return result, filtered_pmf

    def Fisher(self, d_r, r, k, N):
        average_p = 0
        # at the beginning, you initialize the network. After this you will never initialize the network again: you will only make small changes to it.
        network = self.RBN(k, N, r)
        for r in np.arange(0, 1 + d_r, d_r, iterations_for_average):
            # we want an average: so modify the logic tables in a different way.all compare it to the last one (which is only one array).
            # To go to the next d_r, take which state?
            # or do we take one network, and do we shift the activity from 0 to 1?
            # changing the logic tables of the network:
            self.modify_logic_tables(d_r)
            # the Fisher array is the Fisher values for all values of r
            F_array = [0] * (np.arange(0, 1 + d_r, d_r, iterations_for_average))

            for i in range(iterations_for_average):
                last_pmf = pmf
                old_average_p = average_p
                initial_vector, sparse_matrix = network.create_initial_vector_and_sparse_matrix()
                # Call the function to find the stationary distribution (pmf) using power iteration
                pmf = network.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
                # a problem: with calling RBN I would get into an infinite loop?
                # for calculating every r, I want to have the connections already unitialized, but the logic tables changed?
                # or I can make it a bit easier: just reinitialize it every time: the activity should be higher for a general r anyways.

                # for loop: we have the pmf now, we need to identify the differences with the previous pmf
                # we look at the difference per state, and sum them
                # first we go through the pmf, and identify the attractors
                F = 0
                result, filtered_pmf = self.filter_pmf_and_compare(pmf, threshold=0.001, pmf_prev, n=2 ** N)
                for i in range(len(result)):
                    _, result_value = result[i]
                    _, filtered_pmf_value = filtered_pmf[i]

                    F += (filtered_pmf_value * ((result_value / d_r) ** 2))
                    F_array[x] = F

                # Now, the only thing we have to do, is to do this for many, and then average. then, it should be put in the master array

                # now we have done that, we can calculate the Fisher information of that particular state.

                # lets sum the value of a specific state for multiple runs and then average.
                # where do we average? we can average, given an initialized network, and see the average of the change.
                # if we initialize the network over and over again, the changes might still work.

        return F_array


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
Fish = network.Fisher(0.05)
x_values = np.linspace(0, 1, len(Fisher))

# Plot F_array against the equally spaced values
plt.plot(x_values, Fisher, marker='o', linestyle='-')
plt.xlabel('x values')
plt.ylabel('F_array values')
plt.title('F_array values plotted on equal distance between 0 and 1')
plt.grid(True)
plt.show()
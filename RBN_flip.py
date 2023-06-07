import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.sparse import coo_matrix
from scipy import linalg
import math
import cProfile
import re
from functools import partial
import multiprocessing
from multiprocessing import Pool
import sys
import copy
import seaborn as sns


# Importing necessary modules for network creation, plotting, randomization,
# array handling, sparse matrix operations, mathematical functions,
# profiling, regular expressions, functional programming utilities,
# multiprocessing and system-specific parameters.

# Function that computes the average probability mass function (pmf) given certain parameters.
def compute_average_pmf(args):
    # Unpacking the input parameters for the function
    rbn_instance, r, num_T = args
    # Get the instance of the RBN
    network = rbn_instance
    # Initialize a zeroed pmf array of size equal to 2^N (total possible states)
    combined_pmf = np.zeros(2 ** network.N)
    for _ in range(num_T):
        # Generate the logic tables for the network based on flip probability r
        network.generate_logic_tables(r)
        # Create the initial vector and the sparse transition matrix for the network
        initial_vector, sparse_matrix = network.create_initial_vector_and_sparse_matrix()
        # Compute the stationary distribution (long-term behavior) of the network
        pmf = network.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
        # Combine the computed pmf into the cumulative one
        combined_pmf += pmf
    # Compute the average pmf over all iterations
    average_pmf = combined_pmf / num_T
    return average_pmf
class RBN:
    # The __init__ method initializes the RBN with a given number of inputs per node (K),
    # a number of nodes (N), and a flip probability (r).
    def __init__(self, K, N, r):
        self.K = K  # Number of inputs per node
        self.N = N  # Number of nodes in the network
        self.r = r  # Flip probability
        self.p = 0.5
        self.G = nx.DiGraph()
        self.initialization()
        self.generate_logic_tables(self.r)


    # This method sets up the initial state of the RBN, setting the degrees of nodes and edges,
    # and assigning initial states to the nodes.
    def initialization(self):
        #Accounting for the fact that the function deletes half of the nodes
        self.K = 2 * (self.K)
        expected_degrees = [self.K for _ in range(self.N)]
        self.G = nx.expected_degree_graph(expected_degrees, selfloops=False)
        self.G = nx.DiGraph(self.G)
        for u, v in list(self.G.edges()):
            if random.random() < 0.5:
                self.G.remove_edge(u, v)
                self.G.add_edge(v, u)
        for i in range(self.N):
            self.G.nodes[i]["state"] = random.choice([True, False])
    #
    def initialization_exp(self):
        # Create a Watts-Strogatz small-world network
        self.G = nx.watts_strogatz_graph(self.N, self.K, self.p)

        # Convert to a directed graph and randomize edge direction
        self.G = nx.DiGraph(self.G)
        for u, v in list(self.G.edges()):
            if random.random() < 0.5:
                self.G.remove_edge(u, v)
                self.G.add_edge(v, u)

        # Assign a random state to each node
        for i in range(self.N):
            self.G.nodes[i]["state"] = random.choice([True, False])

    def expand_network(self, N_diff):
        new_node_index = self.N  # Index for new nodes starts from the existing number of nodes
        for _ in range(N_diff):
            # Create new nodes and randomly connect them to old nodes
            self.G.add_node(new_node_index, state=random.choice([True, False]))
            old_node = random.choice(range(self.N))  # Randomly select an old node
            self.G.add_edge(new_node_index, old_node)

            # Randomly connect new nodes to each other but not to themselves
            for other_new_node_index in range(self.N, new_node_index):
                if random.random() < 0.5:  # 50% chance to create an edge
                    self.G.add_edge(new_node_index, other_new_node_index)
                if random.random() >= 0.5:  # 50% chance to create an edge
                    self.G.add_edge(other_new_node_index, new_node_index)
            new_node_index += 1

        self.N += N_diff  # Update total number of nodes

        # Generate and modify logic tables for all nodes, as this has changed with the addition of new edges
        self.generate_logic_tables(self.r)

    # This method generates logic tables for all nodes in the network.
    def generate_logic_tables(self, r):
        for i in self.G.nodes:
            inputs = list(self.G.predecessors(i))
            truth_table = [random.random() < r for _ in range(2 ** len(inputs))]
            self.G.nodes[i]["inputs"] = inputs
            self.G.nodes[i]["truth_table"] = truth_table

    def generate_logic_tables_random(self):
        # Iterate over all nodes
        for i in self.G.nodes:
            # Compute a random r between 0 and 1 for each node
            r = random.random()

            # Get all predecessor nodes
            inputs = list(self.G.predecessors(i))

            # Generate truth table with entries based on the random r
            truth_table = [random.random() < r for _ in range(2 ** len(inputs))]

            # Assign the inputs and truth table to the node
            self.G.nodes[i]["inputs"] = inputs
            self.G.nodes[i]["truth_table"] = truth_table

    # def generate_logic_tables_max_diff(self, original_r):
    #     # Compute the new r to maximize difference
    #     if original_r == 0.5:
    #         new_r = random.choice([0, 1])  # if original_r is 0.5, randomly choose 0 or 1
    #     else:
    #         new_r = 1 - original_r
    #
    #     # Iterate over all nodes
    #     for i in self.G.nodes:
    #         # Get all predecessor nodes
    #         inputs = list(self.G.predecessors(i))
    #
    #         # Generate truth table with entries based on the new r
    #         truth_table = [random.random() < new_r for _ in range(2 ** len(inputs))]
    #
    #         # Assign the inputs and truth table to the node
    #         self.G.nodes[i]["inputs"] = inputs
    #         self.G.nodes[i]["truth_table"] = truth_table

    def modify_logic_tables(self, increment_prob):
        for i in self.G.nodes:
            truth_table = self.G.nodes[i]["truth_table"]
            if random.random() < increment_prob:
                if random.random() < 0.2:  # 50% chance to do bitflip or swap
                    # Bitflip
                    index_to_flip = random.randint(0, len(truth_table) - 1)
                    truth_table[index_to_flip] = not truth_table[index_to_flip]
                else:
                    # Swap
                    # Get the indexes of zeros and ones
                    zeros = [j for j, bit in enumerate(truth_table) if bit == 0]
                    ones = [j for j, bit in enumerate(truth_table) if bit == 1]

                    # Calculate the swap proportion
                    swap_count = int(len(truth_table) * 0.5 * increment_prob)  # Change this value to set swap proportion

                    # Select indexes to swap. If there aren't enough elements, reduce the swap count
                    zeros_to_swap = random.sample(zeros, min(swap_count, len(zeros)))
                    ones_to_swap = random.sample(ones, min(swap_count, len(ones)))

                    # Swap zeros and ones
                    for zero_index, one_index in zip(zeros_to_swap, ones_to_swap):
                        truth_table[zero_index], truth_table[one_index] = truth_table[one_index], truth_table[
                            zero_index]

            self.G.nodes[i]["truth_table"] = truth_table



    # Helper functions to convert from binary to decimal and vice versa.
    def bin_to_dec(self, bin_list):
        return np.dot(bin_list, 2 ** np.arange(len(bin_list))[::-1])

    def dec_to_bin(self, dec):
        bin_list = [False] * self.N
        binary_repr = bin(dec)[2:]

        for i in range(len(binary_repr)):
            bin_list[self.N - len(binary_repr) + i] = bool(int(binary_repr[i]))

        return bin_list
    # not used at the moment, but can be handy
    def bin_to_bin_str(self, bin_list):
        bin_str = ''.join(['1' if b else '0' for b in bin_list])
        return bin_str

    # Generates the next state of the network.
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
    # Plotting the graph
    def plot_states(self):
        color_map = {True: 'blue', False: 'red'}
        colors = [color_map[self.G.nodes[node]["state"]] for node in self.G.nodes()]
        nx.draw(self.G, with_labels=True, node_color=colors, font_weight='bold')
        plt.show()

    # Creates the initial state vector and the transition matrix.
    def create_initial_vector_and_sparse_matrix(self):
        #The initial state consists of a vector with uniform probability
        num_states = 2 ** (self.N)
        initial_vector = np.ones(num_states) / num_states

        row_indices = np.zeros(num_states, dtype=int)
        col_indices = np.zeros(num_states, dtype=int)

        after_step = np.zeros(self.N)
        for k in range(num_states):
            Network_state = self.dec_to_bin(k)
            # now manipulate the nodes. The network is already initialized
            for i in range(self.N):
                self.G.nodes[i]["state"] = Network_state[i]
            self.step()
            for i in range(self.N):
                after_step[i] = self.G.nodes[i]["state"]

            row_indices[k] = self.bin_to_dec(after_step)
            col_indices[k] = k

        data = np.ones(num_states)
        transition_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_states, num_states))

        return initial_vector, transition_matrix

    #
    # Finds the stationary distribution of the network given an initial state and a transition matrix.
    def find_stationary_distribution(self, initial_vector, transition_matrix, tolerance, num_iterations=200):
        probability_vector = initial_vector

        for i in range(num_iterations):
            probability_vector_i = transition_matrix * probability_vector
            if np.allclose(probability_vector_i, probability_vector, atol=tolerance):
                probability_vector = probability_vector_i
                return probability_vector

            probability_vector = probability_vector_i

        return probability_vector

    # This method finds the stationary distribution of the network using the power iteration method.
    def filter_pmf_and_compare(self, pmf, pmf_prev, threshold):
        num_states = len(pmf)
        log_pmf = np.where(pmf == 0, 0, np.log(pmf))
        log_pmf_prev = np.where(pmf_prev == 0, 0, np.log(pmf_prev))
        log_diff = np.abs(log_pmf - log_pmf_prev)
        diff = abs(pmf-pmf_prev)

        filtered_pmf = np.where(log_diff >= threshold, pmf, 0)
        result = np.where(log_diff >= threshold, log_diff, 0)

        return result, filtered_pmf, diff

    # This method computes the Fisher information measure for the network. The Fisher information is used to
    # quantify the sensitivity of the distribution of network states to changes in the network parameter r.
    # Begin the definition of the compute_Fisher method
    def compute_Fisher(self, d_r, num_T, threshold, n_processes=None):
        # Initialize some parameters:
        # self.N: the number of nodes in the network
        # num_states: the number of possible states of the network (since each node can be in one of two states)
        # F: this will hold the Fisher information, start with zero
        # diff_cum: this will hold the cumulative difference, start with zero
        N = self.N
        num_states = 2 ** N
        F = 0
        diff_cum = 0

        # Initialize diff_array and F_array with zeros, their length is determined by the range of r values from 0 to 1 inclusive with step size d_r
        diff_array = np.zeros(len(np.arange(0, 1 + d_r, d_r)))
        F_array = np.zeros(len(np.arange(0, 1 + d_r, d_r)))
        # r_values: generate the set of r values from 0 to 1 inclusive with step size d_r
        r_values = np.arange(0, 1 + d_r, d_r)

        # Use parallel processing to speed up the calculation of the average pmf for each r value.
        # The computation of the pmf is a computationally expensive operation and hence we use multiprocessing
        with Pool(processes=n_processes) as pool:
            pmf_stack = np.array(pool.map(compute_average_pmf, [(self, r, num_T) for r in r_values]))

        # Transpose pmf_stack for easier column-wise processing
        pmf_stack = pmf_stack.T

        # Get the number of columns in the transposed pmf_stack, this should be the same as the number of r values
        num_columns = pmf_stack.shape[1]
        assert num_columns == int(1 / d_r) + 1, f'{num_columns=}'

        # Initialize a counter for the main loop
        t = 0

        # Process each pair of adjacent columns from the second column
        for column_index in range(1, num_columns):
            # Get the current and previous columns
            column_1 = pmf_stack[:, column_index]
            column_0 = pmf_stack[:, column_index - 1]

            # Call the filter_pmf_and_compare method on the current and previous columns, getting three results
            result, filtered_pmf, diff = self.filter_pmf_and_compare(column_1, column_0, threshold)

            # Iterate over the length of the result, accumulate the Fisher information and the cumulative difference
            for j in range(len(result)):
                result_value = result[j]
                filtered_pmf_value = filtered_pmf[j]
                diff_value = diff[j]

                # Calculate and add up the Fisher information
                F += (filtered_pmf_value * ((result_value / d_r) ** 2))
                # Add up the differences
                diff_cum += diff_value

            # Update diff_array and F_array at the (t+1) position
            diff_array[t + 1] = diff_cum
            F_array[t + 1] = F

            # Reset F and diff_cum for the next iteration
            F = 0
            diff_cum = 0

            # Increase the counter
            t = t + 1

        # Set the last value of F_array to zero
        F_array[-1] = 0

        # Return the arrays of Fisher information and cumulative differences
        return F_array, diff_array
# This function plots Fisher information and difference in PMF as a function of x values.
def Fisher_plot(d_r, num_T, threshold, num_processes, network):
    # Calculate Fisher information array
    F_array, diff_array = network.compute_Fisher(d_r, num_T, threshold, num_processes)

    # Plot F_array against the equally spaced values
    x_values = np.linspace(0, 1, len(F_array))
    plt.plot(x_values, F_array, marker='o', linestyle='-')
    plt.xlabel('r values')
    plt.ylabel('F_array values')
    plt.title('Fisher information plotted for different r values')
    plt.grid(True)


    plt.show()
    x_values = np.linspace(0, 1, len(diff_array))
    plt.plot(x_values, diff_array, marker='o', linestyle='-')
    plt.xlabel('r values')
    plt.ylabel('diff_array values')
    plt.title('Values of the difference in pmf plotted for different r values')
    plt.grid(True)
    # plt.savefig(r'C:\Users\emiel\OneDrive\Bureaublad\Capstone_g\figure.png')
    plt.show()
# This function creates a bar plot showing how much the PMF changes for a certain r + dr.
# not used in the final product
def one_barplot(network, r, num_T, d_r):
    # a bar plot with how much the pmf changes for a certain r +dr
    args = (network, r, num_T)
    categories = ['0.1+d_r', '0.5+d_r']
    num_compare = 2
    values = np.zeros(num_compare)
    r = 0.1
    average_pmf = compute_average_pmf(args)
    r = 0.1 + d_r
    average1_pmf = compute_average_pmf(args)
    values[0] = np.linalg.norm(average1_pmf - average_pmf)
    r = 0.5
    average_pmf = compute_average_pmf(args)
    r = 0.5 + d_r
    average1_pmf = compute_average_pmf(args)
    values[1] = np.linalg.norm(average1_pmf - average_pmf)

    plt.bar(categories, values, color=['red', 'blue'])
    plt.xlabel('Values of r')
    plt.ylabel('Differences')
    plt.title('The change of pmf per r')
    plt.show()

# This function shows the convergence of error over a number of iterations.
# Begin the definition of the convergence method
def convergence(num_T, N, network, r):
    resvec = np.zeros(num_T)
    combined_pmf = np.zeros(2 ** N)
    for i in range(num_T):
        network.generate_logic_tables(r)
        initial_vector, sparse_matrix = network.create_initial_vector_and_sparse_matrix()
        pmf = network.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
        combined_pmf_prev = combined_pmf.copy() # store previous combined_pmf
        combined_pmf = (i*(combined_pmf_prev) + pmf) / (i+1)
        resvec[i] = np.linalg.norm(combined_pmf - combined_pmf_prev)
    iteration = np.linspace(0, num_T - 1, num_T)
    plt.plot(iteration, resvec, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error convergence')
    plt.grid(True)
    plt.show()

# This function plots PMF for two scenarios side by side for visual comparison.
def pmf_barplot(network, r, num_T, d_r):
 # the upgraded bar plot with 2**N  times two bars

    args = (network, r, num_T)
    r = 0.1
    average_pmf = compute_average_pmf(args)
    r = 0.1 + d_r
    average1_pmf = compute_average_pmf(args)
    num_states = len(average_pmf)
    X_axis = np.arange(num_states)

    bar_width = 0.4
    plt.bar(X_axis - bar_width / 2, average_pmf, bar_width, label='0.1')
    plt.bar(X_axis + bar_width / 2, average1_pmf, bar_width, label='0.1 + d_r')

    plt.xticks(X_axis, X_axis)  # Use system states as x-axis labels
    plt.xlabel('System States (Decimal Value)')
    plt.ylabel('PMF')
    plt.title('PMF values for 0.1 and 0.1 + dr')
    plt.legend()
    plt.show()

    #The same, but for 0.5

    args = (network, r, num_T)
    r = 0.5
    average_pmf = compute_average_pmf(args)
    r = 0.5 + d_r
    average1_pmf = compute_average_pmf(args)

    num_states = len(average_pmf)
    X_axis = np.arange(num_states)

    bar_width = 0.4
    plt.bar(X_axis - bar_width / 2, average_pmf, bar_width, label='0.5')
    plt.bar(X_axis + bar_width / 2, average1_pmf, bar_width, label='0.5 + d_r')

    plt.xticks(X_axis, X_axis)  # Use system states as x-axis labels
    plt.xlabel('System States (Decimal Value)')
    plt.ylabel('PMF')
    plt.title('PMF values for 0.5 and 0.5 + dr')
    plt.legend()
    plt.show()
# This function sorts the pmf arrays and makes a line plot for visual comparison.
# Defining a function called sorted_pmf_plot that accepts four parameters:
# network - a network object/model
# r - a variable which might be related to radius or rate in the context
# num_T - could be the total number of transitions/time units for the process
# d_r - a delta (small change) in the variable 'r'
def sorted_pmf_plot(network, r, num_T, d_r):
    # Packing network, r, num_T into a tuple to be passed into compute_average_pmf function
    args = (network, r, num_T)
    # Setting r to a specific value
    r = 0.1
    # Computing average PMF (probability mass function) values with given parameters
    average_pmf = compute_average_pmf(args)
    # Sorting the computed PMF values
    average_pmf.sort()
    # Setting r to a new value by adding d_r to it
    r = 0.1 + d_r
    # Again computing and sorting average PMF values for the new value of r
    average1_pmf = compute_average_pmf(args)
    average1_pmf.sort()
    # Determining the total number of states for the system
    num_states = len(average_pmf)
    # Creating an array for the X axis to match the number of states
    X_axis = np.arange(num_states)
    # Plotting both sets of average PMFs
    plt.plot(X_axis, average_pmf, label='r=0.1', linestyle='-', marker='o')
    plt.plot(X_axis, average1_pmf, label='r=0.1 + d_r', linestyle='-', marker='o')
    # Filling the area under the curves of both plots with a light color to highlight the area difference
    plt.fill_between(X_axis, average_pmf, alpha=0.2)
    plt.fill_between(X_axis, average1_pmf, alpha=0.2)
    # Setting labels and title for the plot
    plt.xlabel('System States (Decimal Value)')
    plt.ylabel('PMF')
    plt.title('Sorted PMF values for 0.1 and 0.1 + dr')
    # Adding a legend for the plot
    plt.legend()
    # Displaying the plot
    plt.show()
    # Replicating the same process for r = 0.5
    args = (network, r, num_T)
    r = 0.5
    average_pmf = compute_average_pmf(args)
    average_pmf.sort()
    r = 0.5 + d_r
    average1_pmf = compute_average_pmf(args)
    average1_pmf.sort()
    num_states = len(average_pmf)
    X_axis = np.arange(num_states)
    plt.plot(X_axis, average_pmf, label='r=0.5', linestyle='-', marker='o')
    plt.plot(X_axis, average1_pmf, label='r=0.5 + d_r', linestyle='-', marker='o')

    plt.fill_between(X_axis, average_pmf, alpha=0.2)
    plt.fill_between(X_axis, average1_pmf, alpha=0.2)

    plt.xlabel('System States (Decimal Value)')
    plt.ylabel('PMF')
    plt.title('Sorted PMF values for 0.5 and 0.5 + dr')
    plt.legend()
    plt.show()

def print_pmf(network, N):
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

def hellinger_distance(pmf_environment, pmf_agent):
    # Ensure the PMFs sum to 1
    pmf_environment = pmf_environment / np.sum(pmf_environment)
    pmf_agent = pmf_agent / np.sum(pmf_agent)
    # Calculate the Hellinger distance
    return np.sqrt(np.sum((np.sqrt(pmf_environment) - np.sqrt(pmf_agent))**2)) / np.sqrt(2)

def create_agent_and_environment(K, N_agent, N_environment, r, d_r, num_T, threshold, num_processes):
    # Create the agent and the environment
    agent = RBN(K, N_agent, r)
    environment = copy.deepcopy(agent)
    N_diff = abs(N_environment - N_agent)
    environment.expand_network(N_diff)
    environment.generate_logic_tables_random()

    #If we want to know the r for which the Fisher information is maximal:
    # F_array, diff_array = agent.compute_Fisher(d_r, num_T, threshold, num_processes)
    # max_I = np.argmax(F_array) * d_r
    #print("r at maximum Fisher information", max_I)

    # Find the stationary distribution for the agent
    initial_vector, sparse_matrix = agent.create_initial_vector_and_sparse_matrix()
    pmf_agent = agent.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
    # Find the stationary distribution for the environment
    initial_vector, sparse_matrix = environment.create_initial_vector_and_sparse_matrix()
    pmf_environment = environment.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
    # Reshape the environment's pmf to have a separate dimension for each node
    A = pmf_environment.reshape([2] * N_environment)
    # Marginalize out the dimensions the agent does not have access to
    A = A.sum(axis=tuple(range(N_agent, N_environment)))
    A = A.reshape(-1)

    # Assert that the sum of all probabilities in the marginalized pmf is 1 (within numerical error)
    assert np.isclose(A.sum(), 1, atol=1e-8), "The PMF does not sum to 1"
    pmf_environment = A

    return pmf_environment, pmf_agent, agent


def mutation(agent, pmf_environment, hellinger_distance_array, change_count, steps_to_zero, mutation_rate,
             start_for_rate, maxiter, j, steps_to_zero_array):
    # Create a copy of the agent and apply mutation
    agent_new = copy.deepcopy(agent)
    agent_new.modify_logic_tables(mutation_rate)

    # Calculate PMFs for the old and new agents
    initial_vector, sparse_matrix = agent.create_initial_vector_and_sparse_matrix()
    pmf_agent_old = agent.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
    initial_vector, sparse_matrix = agent_new.create_initial_vector_and_sparse_matrix()
    pmf_agent = agent_new.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)

    # Calculate the Hellinger distance and store it
    last_rate = hellinger_distance(pmf_environment, pmf_agent_old)
    hellinger_distance_array[j] = last_rate

    # Update the agent if the Hellinger distance decreases
    if hellinger_distance(pmf_environment, pmf_agent) < last_rate:
        change_count += 1
        steps_to_zero += 1
        last_rate = hellinger_distance(pmf_environment, pmf_agent)
        if last_rate <= 0.00000:
            steps_to_zero_array = np.append(steps_to_zero_array, steps_to_zero)
            agent = agent_new

    # Calculate the rate measure and update the iteration_to_zero
    rate_measure = (last_rate - start_for_rate) / maxiter
    iteration_to_zero = j if last_rate <= 0.00000 else 300

    return agent, change_count, steps_to_zero, last_rate, iteration_to_zero, rate_measure, hellinger_distance_array, steps_to_zero_array


# Define function to calculate average change
def calculate_average_change(agent, pmf_environment, mutation_rate, maxiter, iteration_for_average):
    # Initialize counters and storage arrays
    average_count = 0
    steps_to_zero_array = np.zeros(0)
    av_it = 0
    av_rate_measure = 0
    av_distance = np.zeros(maxiter)
    # Create initial state vector and transition matrix
    initial_vector, sparse_matrix = agent.create_initial_vector_and_sparse_matrix()
    # Compute the stationary distribution of the agent
    pmf_agent = agent.find_stationary_distribution(initial_vector, sparse_matrix, tolerance=1e-8)
    # Set the initial agent pmf and agent state
    pmf_agent_invariant = pmf_agent
    agent_invariant = agent
    # Loop over iterations for averaging
    for k in range(iteration_for_average):
        # Initialize counters and storage variables for each iteration
        change_count = 0
        steps_to_zero = 0
        iteration_to_zero = 0
        rate_measure = 0
        # Reset the agent pmf and agent state to their initial values
        pmf_agent = pmf_agent_invariant
        agent = agent_invariant
        # Calculate initial distance between pmf_environment and pmf_agent
        start_for_rate = hellinger_distance(pmf_environment, pmf_agent)
        # Initialize storage for hellinger distances
        hellinger_distance_array = np.zeros(maxiter)
        # Run mutations over maxiter iterations
        for j in range(maxiter):
            agent, change_count, steps_to_zero, last_rate, iteration_to_zero, rate_measure, hellinger_distance_array, steps_to_zero_array = mutation(
                agent, pmf_environment, hellinger_distance_array, change_count, steps_to_zero, mutation_rate,
                start_for_rate, maxiter, j,
                steps_to_zero_array)
        # Accumulate statistics
        average_count += change_count
        av_it += iteration_to_zero
        av_rate_measure += rate_measure
        av_distance += hellinger_distance_array

    return average_count, av_it, av_rate_measure, steps_to_zero_array, av_distance



def calculate_average_change_multi_initial(K, N_agent, N_environment, r, d_r, num_T,threshold,num_processes, mutation_rate, maxiter, iteration_for_average):
    average_count = 0
    steps_to_zero_array = np.zeros(0)
    av_it = 0
    av_rate_measure = 0
    av_distance = np.zeros(maxiter)

    for k in range(iteration_for_average):
        change_count = 0
        steps_to_zero = 0
        iteration_to_zero = 0
        rate_measure = 0
        pmf_environment, pmf_agent, agent = create_agent_and_environment(K, N_agent, N_environment, r, d_r, num_T,
                                                                         threshold,
                                                                         num_processes)
        start_for_rate = hellinger_distance(pmf_environment, pmf_agent)
        hellinger_distance_array = np.zeros(maxiter)
        for j in range(maxiter):
            agent, change_count, steps_to_zero, last_rate, iteration_to_zero, rate_measure, hellinger_distance_array, steps_to_zero_array = mutation(
                agent, pmf_environment, hellinger_distance_array, change_count, steps_to_zero, mutation_rate,
                start_for_rate, maxiter, j,
                steps_to_zero_array)
        average_count += change_count
        av_it += iteration_to_zero
        av_rate_measure += rate_measure
        av_distance += hellinger_distance_array

    return average_count, av_it, av_rate_measure, steps_to_zero_array, av_distance
def calculate_decrease_hellinger_distance_mutation(K, r, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold,
                                          num_processes):
    change_array = np.zeros(int(1 / d_mutation))
    zero_array = np.zeros(int(1 / d_mutation))
    iteration_to_zero_array = np.zeros(int(1 / d_mutation))
    rate_array = np.zeros(int(1 / d_mutation))
    hellinger_distance_stack =  np.zeros((maxiter, int(1 / d_mutation)))

    for i in range(int(1 / d_mutation)):
        mutation_rate = i / 20
        pmf_environment, pmf_agent, agent = create_agent_and_environment(K, N_agent, N_environment, r, d_r, num_T, threshold,
                                                                         num_processes)
        average_count, av_it, av_rate_measure, steps_to_zero_array, av_distance = calculate_average_change(agent, pmf_environment,
                                                                                              mutation_rate, maxiter,
                                                                                              iteration_for_average)
        change_array[i] = average_count
        av_it /= iteration_for_average
        av_rate_measure /= iteration_for_average
        rate_array[i] = av_rate_measure
        av_distance /= iteration_for_average
        hellinger_distance_stack[:, i] = av_distance

        iteration_to_zero_array[i] = av_it
        if len(steps_to_zero_array) == 0:
            zero_array[i] = 30
        else:
            zero_array[i] = sum(steps_to_zero_array) / len(steps_to_zero_array)
    return change_array, zero_array, iteration_to_zero_array, rate_array, hellinger_distance_stack


def calculate_decrease_hellinger_distance_r_multi_av(K, r, iterations_convergence, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold,
                                          num_processes):
    change_array = np.zeros(int(1 / d_r)+1)
    zero_array = np.zeros(int(1 / d_r)+1)
    iteration_to_zero_array = np.zeros(int(1 / d_r)+1)
    rate_array = np.zeros(int(1 / d_r)+1)
    hellinger_distance_stack = np.zeros((maxiter, int(1 / d_r)+1))

    for i in range(int(1 / d_r)+1):
        r = i / (int(1 / d_r))
        average_count, av_it, av_rate_measure, steps_to_zero_array, av_distance = calculate_average_change_multi_initial(K, N_agent, N_environment, r, d_r, num_T,threshold,num_processes, mutation_rate, maxiter, iteration_for_average)
        av_it /= iteration_for_average
        av_rate_measure /= iteration_for_average
        rate_array[i] = av_rate_measure
        av_distance /= iteration_for_average


        iteration_to_zero_array[i] = av_it
        if len(steps_to_zero_array) == 0:
            zero_array[i] = 30
        else:
            zero_array[i] = sum(steps_to_zero_array) / len(steps_to_zero_array)
        hellinger_distance_stack[:, i] = av_distance

    return change_array, zero_array, iteration_to_zero_array, rate_array, hellinger_distance_stack


#This function exists to calculate the hellinger distance for different environments and agents, with the goal of finding
# the different convergence rates when starting at different initial hellinger distances.
def calculate_decrease_hellinger_distance_r_multi(K, r, iterations_convergence, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold,num_processes):
    change_array = np.zeros(iterations_convergence)
    zero_array = np.zeros(iterations_convergence)
    iteration_to_zero_array = np.zeros(iterations_convergence)
    rate_array = np.zeros(iterations_convergence)
    hellinger_distance_stack = np.zeros((maxiter, iterations_convergence))

    for i in range(iterations_convergence):
        pmf_environment, pmf_agent, agent = create_agent_and_environment(K, N_agent, N_environment, r, d_r, num_T, threshold,
                                                                  num_processes)

        average_count, av_it, av_rate_measure, steps_to_zero_array, av_distance = calculate_average_change(agent, pmf_environment,
                                                                                              mutation_rate, maxiter,
                                                                                              iteration_for_average)
        change_array[i] = average_count
        av_it /= iteration_for_average
        av_rate_measure /= iteration_for_average
        rate_array[i] = av_rate_measure
        av_distance /= iteration_for_average
        hellinger_distance_stack[:, i] = av_distance

        iteration_to_zero_array[i] = av_it
        if len(steps_to_zero_array) == 0:
            zero_array[i] = 30
        else:
            zero_array[i] = sum(steps_to_zero_array) / len(steps_to_zero_array)
    return change_array, zero_array, iteration_to_zero_array, rate_array, hellinger_distance_stack
def calculate_decrease_hellinger_distance_r(K, r, iterations_convergence, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold,num_processes):
    change_array = np.zeros(int(1 / d_r))
    zero_array = np.zeros(int(1 / d_r))
    iteration_to_zero_array = np.zeros(int(1 / d_r))
    rate_array = np.zeros(int(1 / d_r))
    hellinger_distance_stack = np.zeros((maxiter, int(1 / d_r)))

    for i in range(int(1 / d_r)):
        r = i / int(1 / d_r)
        pmf_environment, pmf_agent, agent = create_agent_and_environment(K, N_agent, N_environment, r, d_r, num_T, threshold,
                                                                  num_processes)

        average_count, av_it, av_rate_measure, steps_to_zero_array, av_distance = calculate_average_change(agent, pmf_environment,
                                                                                              mutation_rate, maxiter,
                                                                                              iteration_for_average)
        change_array[i] = average_count
        av_it /= iteration_for_average
        av_rate_measure /= iteration_for_average
        rate_array[i] = av_rate_measure
        av_distance /= iteration_for_average
        hellinger_distance_stack[:, i] = av_distance

        iteration_to_zero_array[i] = av_it
        if len(steps_to_zero_array) == 0:
            zero_array[i] = 30
        else:
            zero_array[i] = sum(steps_to_zero_array) / len(steps_to_zero_array)
    return change_array, zero_array, iteration_to_zero_array, rate_array, hellinger_distance_stack


def convergence_plots_r(K, r, iterations_convergence, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold, num_processes):
    change_array, zero_array, iteration_to_zero_array, rate_array, hellinger_distance_stack = calculate_decrease_hellinger_distance_r_multi_av(K, r, iterations_convergence, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold, num_processes)
    r_values = np.linspace(0, 1, hellinger_distance_stack.shape[1])

    for i in range(hellinger_distance_stack.shape[1]):
        steps = np.arange(hellinger_distance_stack.shape[0])
        plt.plot(steps, hellinger_distance_stack[:, i], label=f'r={r_values[i]:.2f}')

    plt.xlabel('Step')
    plt.ylabel('Hellinger Distance')
    plt.title('Hellinger Distance per Step for Different r')
    plt.legend(loc='best')  # shows the legend using the best location
    plt.show()


def convergence_plots_mutation(K, r, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold, num_processes):
    change_array, zero_array, iteration_to_zero_array, rate_array, hellinger_distance_stack = calculate_decrease_hellinger_distance_mutation(K, r, mutation_rate, N_agent, N_environment, d_mutation, maxiter, iteration_for_average, d_r, num_T, threshold, num_processes)
    mutation_rates = np.linspace(0, 1, hellinger_distance_stack.shape[1])

    for i in range(hellinger_distance_stack.shape[1]):
        steps = np.arange(hellinger_distance_stack.shape[0])
        plt.plot(steps, hellinger_distance_stack[:, i], label=f'mutation rate={mutation_rates[i]:.2f}')

    plt.xlabel('Step')
    plt.ylabel('Hellinger Distance')
    plt.title('Hellinger Distance per Step for Different Mutation Rate')
    plt.legend(loc='best')  # shows the legend using best location
    plt.show()


def calculate_decrease_Hellinger_per_r_and_mutation_rate(K, r, iterations_convergence,mutation_rate, N_agent, N_environment, maxiter, iteration_for_average,  d_r, d_mutation, num_T, threshold, num_processes):
    r_and_mutation_stack = np.empty((int(1 / d_r)+1, int(1/d_mutation)+1))
    for t in range(int(1/d_mutation)+1):
        mutation_rate = t / int(1 / d_mutation)
        change_array, zero_array, iteration_to_zero_array, rate_array, distance_per_r_stack = calculate_decrease_hellinger_distance_r_multi_av(K, r, iterations_convergence,mutation_rate, N_agent, N_environment, d_mutation,  maxiter, iteration_for_average, d_r, num_T, threshold, num_processes)

        r_and_mutation_stack[:, t] = rate_array
        print(r_and_mutation_stack)
    return r_and_mutation_stack


def heatmap_r_mutation(r_and_mutation_stack, d_mutation, d_r):
    # Create a list of r values and mutation rates
    r_values = np.around(np.linspace(0, 1, int(1 / d_r) + 1), decimals=2)
    mutation_rates = np.around(np.linspace(0, 1, int(1 / d_mutation) + 1), decimals=2)

    data = r_and_mutation_stack

    # Define the size of the figure
    plt.figure(figsize=(10, 8))

    # Create the heatmap using seaborn
    heat_map = sns.heatmap(data, fmt=".4f",cmap='viridis', linewidth=1, annot=True,
                           xticklabels=mutation_rates, yticklabels=r_values)

    plt.title("HeatMap of learning rate for different Mutation and r values")
    plt.xlabel("Mutation rate")
    plt.ylabel("r values")
    plt.show()

def plot_results_r(r_values, rate_array):
    plt.figure(figsize=(5, 4))
    plt.plot(r_values, rate_array, marker='o', linestyle='-')
    plt.xlabel('r values')
    plt.ylabel('rate values')
    plt.title('the rate of going to a Hellinger distance of 0')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_results_mutation(mutation_values, rate_array):
    plt.figure(figsize=(5, 4))

    plt.plot(mutation_values, rate_array, marker='o', linestyle='-')
    plt.xlabel('mutation rate')
    plt.ylabel('rate values')
    plt.title('the rate of going to a Hellinger distance of 0')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Set parameters for the RBN
    K = 2  #Connectivity
    N = 5
    N_agent = 3  # Number of nodes agent
    N_environment = 4  # Number of nodes environment
    r = 0.5  # the value of the flip probability
    mutation_rate = 0.5  # the value of the mutation rate
    threshold = 0  # threshold for when we say the Hellinger distance between an agent and a Environment is negligible
    d_r = 0.1  # values of the change of r, used for analysis
    d_mutation = 0.1  # values of the change of the mutation rate, used for analysis
    num_T = 20  # the number of pmfs that is used to create a pmf that is used to calculate the Fisher information
    maxiter = 50 # maximum number of iterations in the mutation process
    iteration_for_average = 40  #amount of iterations of initializations of agents and environments that is used for analysis.
    #p = 0.5
    iterations_convergence = 5  # Not used at the moment
    num_processes = 10  # Not used at the moment
    network = RBN(K,N,r)


    convergence_plots_r(K, r, iterations_convergence, mutation_rate, N_agent, N_environment, d_mutation,
                                 maxiter, iteration_for_average, d_r, num_T, threshold, num_processes)

    r_and_mutation_stack = calculate_decrease_Hellinger_per_r_and_mutation_rate(K, r, iterations_convergence,
                                                                                mutation_rate, N_agent,
                                                                                N_environment, maxiter,
                                                                                iteration_for_average, d_r,
                                                                                d_mutation, num_T, threshold,
                                                                                num_processes)
    heatmap_r_mutation(r_and_mutation_stack, d_mutation, d_r)


if __name__ == "__main__":
    main()
#

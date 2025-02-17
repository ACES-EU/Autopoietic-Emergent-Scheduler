import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from GTAadam_with_IDW import GTAdam
from disropt.functions import SquaredNorm, Variable, Exp, AffineForm
from disropt.utils.graph_constructor import binomial_random_graph,\
     metropolis_hastings
from disropt.problems import Problem
import time
from copy import deepcopy
from utilities import compute_surrogate, update_alpha_beta_yaml
import utilities_swarming
# %%
# Get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()
# %%
# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.3, seed=1)
W = metropolis_hastings(Adj)
# Reset local seed
np.random.seed(int(time.time())+10*local_rank)
# Create agents
agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
    in_weights=W[local_rank, :].tolist(),)
# %%
# n = number of variables
file_path = '/home/loriscannelli/Desktop/Learning/ACES_demo/Active/Swarming/config'
n = 2
x = Variable(n)
ub = 6.0
lb = 0.0
constr = [x <= ub, x >= lb]
# %%
# Generate random points for the surrogates
N_exp = 2*n  # Number of points in initial dictionary
N_test = 2  # Number of MonteCarlo repetitions
# Set algorithm parameters
# Number of experiments each agent does with its own IDW
number_outer_iterations_scaled = 2
# Total number of experiments
number_outer_iterations = number_outer_iterations_scaled * nproc
nr_init = 1  # Number of 'swarming' rounds (not used in the paper)
stepsize_list = [0.001]  # Stepsizes to test
num_iterations = 1000  # Number of ADAM iterations
for stepsize in stepsize_list:
    exploration_list = []
    exploitation_list = []
    scaling_list = []
    obj_func_list = []
    dict_list = []
    for i_test in range(N_test):
        scaling_single_test_list = []
        y_dictionary = np.zeros((N_exp, 1))
        x_dictionary = (ub-lb)*np.random.rand(N_exp, n)+lb
        for idx in range(N_exp):
            update_alpha_beta_yaml(file_path, x_dictionary[idx,:], local_rank)
            y_dictionary[idx, 0] = utilities_swarming.swarming(local_rank)
        scaling = nproc
        scaling_single_test_list.append(scaling)
        # Set the current dictionary and create a surrogate
        D = {'x': x_dictionary, 'y': y_dictionary}
        gpr = compute_surrogate(D)
        obj_func = 0
        for j in range(N_exp):
            obj_func += gpr.alpha_[j, 0] * Exp(
                SquaredNorm(AffineForm(x, np.identity(n),
                                       -D['x'][j, :].reshape(-1, 1)))
                / (-2 * gpr.kernel_.get_params()['k2__length_scale'] ** 2))
        obj_func *= gpr.kernel_.get_params()['k1__constant_value']
        # Assign the local cost function
        pb = Problem(obj_func, constr)
        agent.set_problem(pb)
        # Initialize some auxiliary object
        turn = False
        counter = 0
        exploration = np.zeros((n, number_outer_iterations))
        exploitation = np.zeros((n, number_outer_iterations))
        # Run the distributed method
        for i in range(number_outer_iterations):
            delta = 1*scaling  # IDW weight
            if local_rank == 0:
                print('Outer iteration {}'.format(i))
            if np.mod(i, nproc) == local_rank:
                turn = True
            # Try different initialization ('swarming')
            final_points = []
            final_global_costs = []
            for j in range(nr_init):
                # ADAM
                x0 = (ub-lb)*np.random.rand(n, 1)+lb
                adam = GTAdam(agent=agent, initial_condition=x0,
                              enable_log=True)
                adam_seq = adam.run(
                    iterations=num_iterations,
                    stepsize=stepsize,
                    dictionary=D,
                    local_rank=local_rank,
                    turn=turn,
                    delta=delta,
                    verbose=False)
                # Find global cost via consensus
                achieved_global_cost = \
                    adam.run_consensus_on_cost(iterations=100)
                final_points.append(deepcopy(adam.get_result().reshape(n,)))
                final_global_costs.append(achieved_global_cost.item())
            # Pick the solution leading to the lower cost
            best_solution_idx = np.argmin(final_global_costs)
            exploration[:, i] = final_points[best_solution_idx]
            # In a cyclic way each agent updates its dictionary and surrogate
            if turn:
                counter += 1
                x_dictionary = np.append(x_dictionary,
                                         exploration[:, i].reshape(-1, n),
                                         axis=0)
                update_alpha_beta_yaml(file_path, exploration[:, i].reshape(-1, n).flatten(), local_rank)
                y_dictionary = np.append(y_dictionary, np.array([[utilities_swarming.swarming(local_rank)]]), axis=0)
                D = {'x': x_dictionary, 'y': y_dictionary}
                gpr = compute_surrogate(D)
                obj_func = 0
                for j in range(N_exp + counter):
                    obj_func += gpr.alpha_[j, 0] * Exp(
                        SquaredNorm(AffineForm(x, np.identity(n),
                                               -D['x'][j, :].reshape(-1, 1)))
                        / (-2*gpr.kernel_.get_params()['k2__length_scale']**2))
                obj_func *= gpr.kernel_.get_params()['k1__constant_value']
                pb = Problem(obj_func, constr)
                agent.set_problem(pb)
                scaling_single_test_list.append(scaling)
                turn = False
            # Compute solutions without IDW, for evaluation purposes
            final_points = []
            final_global_costs = []
            for j in range(nr_init):
                x0 = (ub-lb)*np.random.rand(n, 1)+lb
                adam = GTAdam(agent=agent, initial_condition=x0,
                              enable_log=True)
                adam_seq = adam.run(
                    iterations=num_iterations,
                    stepsize=stepsize,
                    dictionary=D,
                    local_rank=local_rank,
                    turn=turn,
                    delta=delta,
                    verbose=True)
                achieved_global_cost = \
                    adam.run_consensus_on_cost(iterations=100)
                final_points.append(deepcopy(adam.get_result().reshape(n,)))
                final_global_costs.append(achieved_global_cost.item())
            best_solution_idx = np.argmin(final_global_costs)
            exploitation[:, i] = final_points[best_solution_idx]
        if local_rank == 0:
            print('End outer iteration {}'.format(i))
            print('Final optimized point: {}'.format(exploitation[:,i]))
        exploitation_list.append(exploitation)
        exploration_list.append(exploration)
        scaling_list.append(scaling_single_test_list)
        obj_func_list.append(obj_func)
        dict_list.append(D['x'])
    # Save data
    with open('./Data/swarming/agents_step_{}.npy'.format(stepsize), 'wb') \
         as f:
        np.save(f, nproc)
        np.save(f, n)
        np.save(f, number_outer_iterations_scaled)
        np.save(f, N_test)
        np.save(f, nr_init)
    np.save('./Data/swarming/agent_{}_sequence_step_{}.npy'.format(agent.id,
                                                             stepsize),
            exploration_list)
    np.save('./Data/swarming/agent_{}_exploitation_step_{}.npy'.format(agent.id,
                                                                 stepsize),
            exploitation_list)
    np.save('./Data/swarming/agent_{}_dictionaries_step_{}.npy'.format(agent.id,
                                                                 stepsize),
            dict_list)
    np.save('./Data/swarming/agent_{}_scaling_step_{}.npy'.format(agent.id,
                                                            stepsize),
            scaling_list)
    with open('./Data/swarming/agent_{}_function_step_{}.pkl'.format(agent.id,
                                                               stepsize),
              'wb') as output:
        pickle.dump(obj_func_list, output, pickle.HIGHEST_PROTOCOL)
    with open('./Data/swarming/constraints_step_{}.pkl'.format(stepsize),
              'wb') as output:
        pickle.dump(constr, output, pickle.HIGHEST_PROTOCOL)
from problems import * 
import examples
import tests
from typing import List
from plot_style import *


import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["figure.autolayout"] = True
main_colors = ["r", "b", "c", "g", "m", "k"]

P_TYPE_BASIC_SOLVABLE = 'basic_solvable'
P_TYPE_NON_SOLVABLE_1 = 'non_solvable_1'
P_TYPE_NON_SOLVABLE_2 = 'non_solvable_2'
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

class ExperimentResults:
    def __init__(self, costs: np.ndarray, grad_norms: np.ndarray, dim: int, runs: int, problem_type: str, opt_points):
        self.costs = costs
        self.mean_cost = np.mean(costs)
        self.grad_norms = grad_norms
        self.dim = dim
        self.runs = runs
        self.opt_points = opt_points
        self.problem_type = problem_type
        self.id = None
        self.eps_non_solvable = None
        ExperimentResults.set_id(self)
        
    def print_data(self):
        file_name = "full_results_" + self.id + ".csv"
        n = len(self.costs)
        mode = 'w'
        header = "run,last_cost,last_grad_norm\n"
        with open(file_name, mode=mode) as file:
            file.write(header)
        mode = 'a'
        print(len(self.grad_norms))
        for i in range(n):
            line = str(i) + "," + str("{:.4e}".format(self.costs[i])) + "," + str("{:.4e}".format(self.grad_norms[i])) + "\n"
            print(line)
            with open(file_name, mode=mode) as file:
                file.write(line)
        
    
    def set_id(self):
        id = "dim_" + str(self.dim) + "_runs_" + str(self.runs) + "_type_" + self.problem_type
        if self.eps_non_solvable is not None:
            id += "_eps_" + str(round(self.eps_non_solvable, 2))
        self.id = id
        
    def __str__(self):
        info = "[ExperimentResults object] : \n"
        info += "problem_type = " + self.problem_type
        if self.eps_non_solvable is not None:
            info += ",\t eps = " + str(round(self.eps_non_solvable,2))
        info += ",\t dim = " + str(self.dim) + ",\t runs = " + str(self.runs)
        info += ", mean final cost = " + str(self.mean_cost) + "\n"
        return info
    

def summary(experiment_results_array: List[ExperimentResults]):
    first_experiment = experiment_results_array[0]
    file_name = "summary_" + first_experiment.id + ".txt"
    i = 0
    for results in experiment_results_array:
        if i == 0:
            mode = 'w'
        else: 
            mode = 'a'
        with open(file_name, mode=mode) as file:
            file.write(results.__str__())
        i += 1
        
def collect_mean_costs(experiment_results_array: List[ExperimentResults]):
    mean_costs = []
    for results in experiment_results_array:
        mean_costs.append(results.mean_cost)
    return np.array(mean_costs)
            

def max_absolute_difference(arr):
    max_diff = -1
    size = len(arr)
    for i in range(size):
        for j in range(i + 1, size):
            diff = np.abs(arr[i] - arr[j])
            if diff > max_diff:
                max_diff = diff
    return max_diff


def single_run(problem_, verbosity = 0, line_at = None, A = None, B = None, C = None):
    optimizer = pymanopt.optimizers.TrustRegions(verbosity = verbosity)
    results = optimizer.run(problem_)
    data = results.full_results
    print("Final cost and point")
    print("Final cost")
    print(results.cost)
    print("Final point")
    print(results.point)
    cost_value_at_point = problem_.cost(results.point)
    print("Cost value at final point")
    print(cost_value_at_point)
    
    x = data[:, 0]
    x_axis = np.arange(len(x))
    y = data[:, 1]
    # Create the 2D plots
    # Cost vs. iteration
    plt.plot(x_axis, x, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    if line_at != None: 
        if line_at != 'traces':
            plt.axhline(y=line_at, color='r', linestyle='--', label='opt_guess')
        else:
            opt_value = as_traces(A, B, C, results.point)
            plt.axhline(y=opt_value, color = 'r', linestyle='--', label='opt_guess')
    # Show the plot
    plt.savefig('single_run_cost.png')
    plt.show()
    
    # Grad. norm vs. iteration
    plt.plot(x_axis, y, marker='o', linestyle='-', color = 'red')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Grad. norm')
    plt.grid(True)
    # Show the plot
    plt.savefig('single_run_grad_norm.png')
    plt.show()
    
def n_runs_2(problem_, runs = 10, runs_cost = None, runs_grad = None, verbosity = 0, line_at = None, adjust_scale = False, problem_type = P_TYPE_BASIC_SOLVABLE, verbose = 0, save = True, close = True):
    data_all_runs = []
    if runs_cost == None:
        runs_cost = runs
    if runs_grad == None:
        runs_grad = runs
    # Execute runs and gather data
    costs, grad_norms, opt_points = [], [], []
    for i in range(runs):
        optimizer = pymanopt.optimizers.TrustRegions(verbosity = verbosity)
        results = optimizer.run(problem_)
        costs.append(results.cost)
        grad_norms.append(results.gradient_norm)
        opt_points.append(results.point)
        data = results.full_results
        data_all_runs.append(data)
        
    # Generate final cost vs. run plot
    results = ExperimentResults(costs = costs, grad_norms = grad_norms, dim = problem_.dim, runs = runs, problem_type = problem_.problem_type, opt_points = opt_points)
    if problem_type ==  P_TYPE_NON_SOLVABLE_1:
        results.eps_non_solvable = problem_.eps_non_solvable
        results.set_id()
        
    x_axis = np.arange(runs)
    # Create the 2D plot
    # Final cost vs. run
    plt.plot(x_axis, costs, marker='o', linestyle='-', markersize = 2)
    plt.xlabel('Run')
    plt.ylabel('Final cost')
    # plt.yscale('log')
    cost_mean = np.mean(costs)
    max_cost_diff = max_absolute_difference(costs)
    diff_y = 1e1
    if adjust_scale:
        # Set the y-range (y-axis limits)
        y_min = cost_mean + diff_y * max_cost_diff
        y_max = cost_mean - diff_y * max_cost_diff
        plt.ylim(y_min, y_max)
        plt.gca().invert_yaxis()
    plt.grid(True)
    if save: 
        file_name = 'n_runs_cost'
        file_name += '_' + results.id
        plt.savefig(file_name + ".png")
    if verbose >= 1: 
        plt.show()
    if close:
        plt.close()

    # Colors for cost vs. iteration plot
    rgb1 = (0.0, 0.0, 0.0)
    rgb2 = (0.75, 0.75, 0.75)
    diff = (rgb2[0] - rgb1[0], rgb2[1] - rgb1[1], rgb2[2] - rgb1[2])
    colors = []
    r, b, g = 1.0, 1.0, 1.0
    for i in range(runs_cost):
        step = float(i) / float(runs_cost)
        color = (rgb1[0] + diff[0] * step * r, rgb1[1] + diff[1] * step * b, rgb1[2] + diff[2] * step * g, 1)
        colors.append(color)
        
    # Create cost vs. iterations plot
    for i in range(runs_cost):
        data = data_all_runs[i]
        y = data[:, 0]
        x_axis = np.arange(len(y))
        # Grad. norm vs. iteration
        plt.plot(x_axis, y, marker='.', linestyle='-', color = colors[i])
        #plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        if line_at != None: 
            plt.axhline(y=line_at, color='r', linestyle='--', label='opt_guess')
        
    # Show the plot
    file_name_cost = file_name + '_cost_vs_iteration.png'
    plt.savefig(file_name_cost)
    plt.show()
    
    # Colors for grad_norm vs. iteration plot
    rgb1 = (0.0, 0.0, 1.0)
    rgb2 = (0.75, 0.75, 1.0)
    diff = (rgb2[0] - rgb1[0], rgb2[1] - rgb1[1], rgb2[2] - rgb1[2])
    colors = []
    r, b, g = 1.0, 1.0, 1.0
    for i in range(runs_grad):
        step = float(i) / float(runs_grad)
        color = (rgb1[0] + diff[0] * step * r, rgb1[1] + diff[1] * step * b, rgb1[2] + diff[2] * step * g, 1)
        colors.append(color)
        
    # Create grad_norm vs. iteration plot
    for i in range(runs_grad):
        data = data_all_runs[i]
        y = data[:, 1]
        x_axis = np.arange(len(y))
        # Grad. norm vs. iteration
        plt.plot(x_axis, y, marker='.', linestyle='-', color = colors[i])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Grad. norm')
        plt.grid(True)
        
    # Show the plot
    file_name_grad = file_name + '_grad_vs_iteration.png'
    plt.savefig(file_name_grad)
    plt.show()
    

        

    

def n_runs(problem_, n = 100, adjust_scale = False, problem_type = P_TYPE_BASIC_SOLVABLE, verbose = 0, save = True, close = True):
    costs = []
    opt_points = []
    grad_norms = []
    for i in range(n):
        optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
        results = optimizer.run(problem_)
        costs.append(results.cost)
        grad_norms.append(results.gradient_norm)
        opt_points.append(results.point)

    costs = np.array(costs) # Saving costs from last iteration in each run
    grad_norms = np.array(grad_norms) # Saving grad norms from last iteration in each run
    # grad_norms = np.array(results.full_results[:, 1]) # Careful, we are saving grad norms from last run
    problem_dim = problem_.manifold._n
    if verbose >= 1: 
        print(costs)  
    
    max_cost_diff = max_absolute_difference(costs)
    cost_mean = np.mean(costs)
    

    results = ExperimentResults(costs = costs, grad_norms = grad_norms, dim = problem_dim, runs = n, problem_type = problem_type, opt_points = opt_points)
    if problem_type ==  P_TYPE_NON_SOLVABLE_1:
        results.eps_non_solvable = problem_.eps_non_solvable
        results.set_id()
    x_axis = np.arange(n)

    # Create the 2D plot
    # Final cost vs. run
    plt.plot(x_axis, costs, marker='o', linestyle='-', markersize = 2)
    plt.xlabel('Run')
    plt.ylabel('Final cost')
    # plt.yscale('log')
    if adjust_scale:
        # Set the y-range (y-axis limits)
        y_min = cost_mean + 1.5 * max_cost_diff
        y_max = cost_mean - 1.5 * max_cost_diff
        plt.ylim(y_min, y_max)
    plt.grid(True)
    if save: 
        file_name = 'n_runs_cost'
        file_name += '_' + results.id + '.png'
        plt.savefig(file_name)
    if verbose >= 1: 
        plt.show()
    if close:
        plt.close()
    return results
    
def exp_non_solvable_1(Ns = [10], ns = [100]):
    assert(len(Ns) == len(ns) and "Ns and ns must have the same length")
    n_exps = len(Ns)
    full_results = []
    problem_type = P_TYPE_NON_SOLVABLE_1
    for i in range(n_exps):
        print("Experiment # ", i + 1, " out of ", n_exps, " ...")
        N, n = Ns[i], ns[i]
        problem_ = prob_non_solvable_1(N = N)
        results = n_runs(problem_= problem_, n = n, problem_type = problem_type)
        full_results.append(results)
    summary(full_results)
    full_results[0].print_data()
    print("\n")
    
def exp_non_solvable_1_2(Ns = [10], ns = [10], eps_range = [0, 1], n_eps = 20, verbose = 0):
    assert(len(Ns) == len(ns) and "Ns and ns must have the same length")
    n_exps = len(Ns)
    full_results = []
    problem_type = P_TYPE_NON_SOLVABLE_1
    eps_values = np.linspace(eps_range[0], eps_range[1], n_eps)
    for i in range(n_exps):
        N, n = Ns[i], ns[i]
        print("Experiment # ", i + 1, " out of ", n_exps, " ...")
        print("epsilon values...")
        for eps in eps_values:
            print("eps = ", round(eps,2))
            save = (eps == eps_range[1])
            problem_ = prob_non_solvable_1(N = N, epsilon=eps)
            results = n_runs(problem_= problem_, n = n, problem_type = problem_type, save = save, verbose = verbose, close = False)
            full_results.append(results)
    summary(full_results)
    mean_costs = collect_mean_costs(full_results)
    x = eps_values
    y = mean_costs
    # Create the 2D plots
    # Cost vs. iteration
    plt.close()
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('epsilon value')
    plt.ylabel('mean cost')
    plt.grid(True)
    # Show the plot
    plot_name = 'mean_cost_vs_eps_' + full_results[1].id + '.png'
    plt.savefig(plot_name)
    if verbose >= 1: 
        plt.show()
    print("\n")
        
def exp_non_solvable_2(Ns = [10], ns = [100]):
    assert(len(Ns) == len(ns) and "Ns and ns must have the same length")
    n_exps = len(Ns)
    full_results = []
    problem_type = P_TYPE_NON_SOLVABLE_2
    for i in range(n_exps):
        print("Experiment # ", i + 1, " out of ", n_exps, " ...")
        N, n = Ns[i], ns[i]
        problem_ = prob_non_solvable_2(N = N)
        results = n_runs(problem_= problem_, n = n, problem_type = problem_type)
        full_results.append(results)
    summary(full_results)
    full_results[0].print_data()
    print("\n")
    
def exp_basic_solvable(Ns = [10], ns = [100]):
    assert(len(Ns) == len(ns) and "Ns and ns must have the same length")
    n_exps = len(Ns)
    full_results = []
    problem_type = P_TYPE_BASIC_SOLVABLE
    for i in range(n_exps):
        print("Experiment # ", i + 1, " out of ", n_exps, " ...")
        N, n = Ns[i], ns[i]
        problem_ = prob_basic_solvable(N = N)
        results = n_runs(problem_= problem_, n = n, problem_type = problem_type)
        full_results.append(results)
    summary(full_results)
    full_results[0].print_data()
    print("\n")
    
def compare_optimal_points(opt1, opt2, dist = 'fro', prints = False):
    q1, q2 = opt1[0], opt1[1]
    r1, r2 = opt2[0], opt2[1]
    if prints:
        print("Point 1:")
        print(q1)
        print(q2)    
        print("Point 2:")
        print(r1)
        print(r2)
    
    U1 = q1.T @ r1
    U2 = q2.T @ r2
    
    if prints:
        print("U1, U2:")
        print(U1)
        print(U2)

    return np.linalg.norm(U1 - U2, ord = dist)

def compare_optimal_points_2(opt1, opt2, dist = 'fro', prints = False):
    q1, q2 = opt1[0], opt1[1]
    r1, r2 = opt2[0], opt2[1]
    # if prints:
    #     print("Point 1:")
    #     print(q1)
    #     print(q2)    
    #     print("Point 2:")
    #     print(r1)
    #     print(r2)
    
    U1 = q2 @ q1.T
    U2 = r2 @ r1.T
    
    if prints:
        print("U1, U2:")
        print(U1)
        print(U2)

    return np.linalg.norm(U1 - U2, ord = dist)
    
    
    
# args = {}
# args['d'] = 3
# args['q'] = 0.7
# args['m'] = 40
# args['kappa1'] = 5
# args['kappa2'] = 0
# args['ERp'] = 0.75 # G non-complete with edge density 75%
# args['num_anchors'] = 5 # we have to fix one anchor to be the identity
# args['explicit_anchors'] = None
# problem = gen_problem_from_dict(args)


# def gen_problem_from_dict(main_args):
#     return generate_data(**{key: value for arg in main_args for key, value in main_args.items()})



    
def exp_compare_optimal_points(Ns = [10], ns = [10], problem_type = P_TYPE_NON_SOLVABLE_1):
    assert(len(Ns) == len(ns) and "Ns and ns must have the same length")
    n_exps = len(Ns)
    full_results = []
    for i in range(n_exps):
        print("Experiment # ", i + 1, " out of ", n_exps, " ...")
        dists = []
        N, n = Ns[i], ns[i]
        problem_ = prob_basic_solvable(N = N)
        results_1 = n_runs(problem_ = problem_, n = n, problem_type = problem_type)
        results_2 = n_runs(problem_ = problem_, n = n, problem_type = problem_type)
        for j in range(n):
            dist = compare_optimal_points(results_1.opt_points[j], results_2.opt_points[j])
            print("Run j = ", j + 1, ", |U_1 - U_2| = ", dist)
            dists.append(dist)
    print("\n")
    
    
def exp_compare_optimal_points_2(Ns = [10], ns = [10], problem_type = P_TYPE_NON_SOLVABLE_1):
    assert(len(Ns) == len(ns) and "Ns and ns must have the same length")
    n_exps = len(Ns)
    full_results = []
    for i in range(n_exps):
        print("Experiment # ", i + 1, " out of ", n_exps, " ...")
        dists = []
        N, n = Ns[i], ns[i]
        problem_ = prob_basic_solvable(N = N)
        results_1 = n_runs(problem_ = problem_, n = n, problem_type = problem_type)
        results_2 = n_runs(problem_ = problem_, n = n, problem_type = problem_type)
        for j in range(n):
            dist = compare_optimal_points_2(results_1.opt_points[j], results_2.opt_points[j], prints = True)
            print("Run j = ", j + 1, ", |U_1 - U_2| = ", dist)
            dists.append(dist)
    print("\n")
    
# This guess would be wrong, as it is not optimal in general for non-solvable instances
def optimal_guess(A, B, C):
    # Eigenvalues of A, B, C
    a, eigv_a = np.linalg.eig(A)
    b, eigv_b = np.linalg.eig(B)
    c, eigv_c = np.linalg.eig(C)
    # Eigenvalues of pairs A + B B + C, C + A
    ab, eigv_ab = np.linalg.eig(A + B)
    bc, eigv_bc = np.linalg.eig(B + C)
    ca, eigv_ca = np.linalg.eig(C + A)
    a_ord = np.sort(a) # in ascending order
    b_ord = np.sort(b)
    c_ord = np.sort(c)
    a_ord = a_ord[::-1] # reverse the array
    b_ord = b_ord[::-1]
    c_ord = c_ord[::-1]
    ab_ord, bc_ord, ca_ord = np.sort(ab), np.sort(bc), np.sort(ca)
    h = 0.5 * np.sum(c_ord * ab_ord + a_ord * bc_ord + b_ord * ca_ord)
    result = np.trace(A.T@A + B.T@B + C.T@C) + 2*h
    print(result)
    return result

def correct_order_transformation(original_eigs, new_eigs):
    dim = len(original_eigs)
    result = np.zeros((dim,dim))
    id = np.eye(dim)
    for i in range(dim):
        j = np.where(original_eigs == new_eigs[i])[0][0]
        result[:, i] = id[:,j]
    return result


def check_second_order_optimality(problem_data, opt_point):
    A, B, C = problem_data[0], problem_data[1], problem_data[2]

    a, eigv_a = np.linalg.eig(A)
    b, eigv_b = np.linalg.eig(B)
    c, eigv_c = np.linalg.eig(C)
    # Set diagonalized matrices for A, B, C
    A_eigs = np.diag(a)
    B_eigs = np.diag(b)
    C_eigs = np.diag(c)
    
    q1, q2 = opt_point[0], opt_point[1]
    A1 = q1.T @ A_eigs @ q1
    B2 = q2.T @ B_eigs @ q2
    C0 = C_eigs
    
    dim = A.shape[0]
    t = 1.0
    for j in range(10):
        Omega1 = skew_part(np.random.rand(dim, dim))
        Omega2 = skew_part(np.random.rand(dim, dim))
        comm_Omega1_C = Omega1 @ C0 - C0 @ Omega1
        comm_Omega2_C = Omega2 @ C0 - C0 @ Omega2
        
        brocketA = inner_product(Omega1 @ comm_Omega1_C,A1)
        brocketB = inner_product(Omega2 @ comm_Omega2_C, B2)
        AB_term = inner_product(t*(Omega1@Omega1 + Omega2@Omega2) - 2 * (Omega1 @ Omega2), A1 @ B2)
        Omega_diff_term = inner_product(Omega1 - Omega2, A1 @ (Omega1 - Omega2) @ B2)
        
        condition = brocketA + brocketB + AB_term + Omega_diff_term
        
        print("{:.6e}".format(condition))
    


# This guess would be wrong, as it is not optimal in general for non-solvable instances
def as_traces(A, B, C, opt_point):
    a, eigv_a = np.linalg.eig(A)
    b, eigv_b = np.linalg.eig(B)
    A_eigs = np.diag(a)
    B_eigs = np.diag(b)
    
    q1, q2 = opt_point[0], opt_point[1]
    A_mod1 = q1.T @ A_eigs @ q1
    B_mod1 = q2.T @ B_eigs @ q2
    M1 = A_mod1 + B_mod1
    M1_naive = A_eigs + B_eigs
    
    a_ord = np.sort(a)[::-1]
    A_eigs_ord = np.diag(a_ord)
    print("A_eigs_ord")
    print(A_eigs_ord)
    q_ord = correct_order_transformation(a, a_ord)
    print(q_ord)
    r1, r2 = (q2@q1.T) @ q_ord.T, q1.T @ q_ord.T
    B_mod2 = r1.T @ B_eigs @ r1
    C_mod2 = r2.T @ C @ r2
    M2 = B_mod2 + C_mod2
    M2_naive = B_eigs + C
    
    b_ord = np.sort(b)[::-1]
    B_eigs_ord = np.diag(b_ord)
    print("B_eigs_ord")
    print(B_eigs_ord)
    q_ord = correct_order_transformation(b, b_ord)
    print(q_ord)
    s1, s2 = q2.T @ q_ord.T, q1@q2.T @ q_ord.T
    C_mod3 = s1.T @ C @ s1
    A_mod3 = s2.T @ A_eigs @ s2
    M3 = C_mod3 + A_mod3
    M3_naive = C + A_eigs
    
    # Eigenvalues of A, B, C
    a, eigv_a = np.linalg.eig(A)
    b, eigv_b = np.linalg.eig(B)
    c, eigv_c = np.linalg.eig(C)
    # Eigenvalues of pairs A' + B', B'' + C'', C''' + A'''
    ab, eigv_ab = np.linalg.eig(M1)
    bc, eigv_bc = np.linalg.eig(M2)
    ca, eigv_ca = np.linalg.eig(M3)
    
    ab_naive, eigv_ab_naive = np.linalg.eig(M1_naive)
    bc_naive, eigv_bc_naive = np.linalg.eig(M2_naive)
    ca_naive, eigv_ca_naive = np.linalg.eig(M3_naive)
    
    a_ord, b_ord, c_ord = np.sort(a)[::-1], np.sort(b)[::-1], np.sort(c)[::-1]
    print("Eigenvalues of C", c)
    # This doesn't work either, and yields different results w.r.t. previous guess
    ab_ord, bc_ord, ca_ord = np.sort(ab), np.sort(bc), np.sort(ca)
    ab_naive_ord, bc_naive_ord, ca_naive_ord = np.sort(ab_naive), np.sort(bc_naive), np.sort(ca_naive)
    h_ord = 0.5 * np.sum(c_ord * ab_ord + a_ord * bc_ord + b_ord * ca_ord)
    h = 0.5 * np.sum(c_ord * np.diagonal(M1) + a_ord * np.diagonal(M2) + b_ord * np.diagonal(M3))
    h_naive = 0.5 * np.sum(c_ord * ab_naive_ord + a_ord * bc_naive_ord + b_ord * ca_naive_ord)
    h_raw = 0.5 * (np.trace(C.T @ M1) + np.trace(A_eigs_ord @ M2) + np.trace(B_eigs_ord@ M3))
    #print(a_ord)
    
    result = np.trace(A_eigs.T@A_eigs+ B_eigs.T@B_eigs + C.T@C)+ 2.*h
    result *= 0.5
    
    result_ord= np.trace(A_eigs.T@A_eigs+ B_eigs.T@B_eigs + C.T@C)+ 2.*h_ord
    result_ord *= 0.5
    
    result_naive = np.trace(A_eigs.T@A_eigs+ B_eigs.T@B_eigs + C.T@C)+ 2.*h_naive
    result_naive *= 0.5
    
    result_raw = np.trace(A_eigs.T@A_eigs+ B_eigs.T@B_eigs + C.T@C)+ 2.*h_raw
    result_raw *= 0.5
    
    F = A_mod1 + B_mod1 + C
    result_nonexpanded = 0.5*np.trace(F.T@F)
    print("Result naive: ", result_naive)
    print("Result ord: ", result_ord)
    print("Result original: ", result)
    print("Result raw: ", result_raw)
    print("Result nonexpanded: ", result_nonexpanded)
    
    i = 1
    j = 4
    
    delta_ij = c_ord[i] - c_ord[j]
    m_ij = M1[i,i]-M1[j,j]
    print(delta_ij)
    P = A_mod1 @ B_mod1
    p_ij = P[i,i] + P[j,j]
    ineq1 = -delta_ij * m_ij + 2*p_ij
    ab_terms_ij = A_mod1[i,i]*B_mod1[j,j] + A_mod1[j,j]*B_mod1[i,i] - 2*A_mod1[i,j]*B_mod1[i,j]
    ineq2 = -delta_ij * m_ij - 2*p_ij+4*ab_terms_ij
    ineq3 = -delta_ij * (A_mod1[i,i] - A_mod1[j,j]) + ab_terms_ij
    ineq4 = -delta_ij * (B_mod1[i,i] - B_mod1[j,j]) + ab_terms_ij
    print("1st inequality: ", ineq1)
    print("2nd inequality: ", ineq2)
    print("3rd inequality: ", ineq3)
    print("4th inequality: ", ineq4)
    
    # print("Eigenvalues M1: ")
    # print(ab)
    # print("Eigenvalues M1 ord: ")
    # print(ab_ord)
    
    # print("Eigenvalues M2: ")
    # print(bc)
    # print("Eigenvalues M2 ord: ")
    # print(bc_ord)
    
    # print("Eigenvalues M3: ")
    # print(ca)
    # print("Eigenvalues M3 ord: ")
    # print(ca_ord)
    
    
    print("M1 matrix:")
    print(M1)
    print("M2 matrix:")
    print(M2)
    print("M3 matrix:")
    print(M3)
    return result_ord
    
    
    
    
        
        
    
    
        
        

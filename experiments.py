from problems import * 
import examples
import tests
from typing import List
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'cm'
main_colors = ["r", "b", "c", "g", "m", "k"]

P_TYPE_BASIC_SOLVABLE = 'basic_solvable'
P_TYPE_NON_SOLVABLE_1 = 'non_solvable_1'
P_TYPE_NON_SOLVABLE_2 = 'non_solvable_2'


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

def single_run(problem_, verbosity = 0):
    optimizer = pymanopt.optimizers.TrustRegions(verbosity = verbosity)
    results = optimizer.run(problem_)
    data = results.full_results
    
    x = data[:, 0]
    x_axis = np.arange(len(x))
    y = data[:, 1]
    # Create the 2D plots
    # Cost vs. iteration
    plt.plot(x_axis, x, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
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

def n_runs(problem_, n = 100, adjust_scale = False, problem_type = P_TYPE_BASIC_SOLVABLE, verbose = 0, save = True, close = True):
    costs = []
    opt_points = []
    for i in range(n):
        optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
        results = optimizer.run(problem_)
        costs.append(results.cost)
        opt_points.append(results.point)

    costs = np.array(costs) # Saving costs from last iteration in each run
    grad_norms = np.array(results.full_results[:, 1]) # Careful, we are saving grad norms from last run
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
    print("\n")
    
def exp_non_solvable_1_2(Ns = [10], ns = [10], eps_range = [0, 1], n_eps = 10, verbose = 0):
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
    
    
    
        
        
    
    
        
        

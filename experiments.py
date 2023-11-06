from problems import * 
import examples
import tests
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'cm'
main_colors = ["r", "b", "c", "g", "m", "k"]


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

def n_runs(problem_, n = 100, suffix = ''):
    costs = []
    for i in range(n):
        optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
        results = optimizer.run(problem_)
        costs.append(results.cost)

    costs = np.array(costs)   
    
    max_cost_diff = max_absolute_difference(costs)
    cost_mean = np.mean(costs)

    x_axis = np.arange(n)

    # Create the 2D plot
    # Final cost vs. run
    plt.plot(x_axis, costs, marker='o', linestyle='-', markersize = 2)
    plt.xlabel('Run')
    plt.ylabel('Final cost')
    # plt.yscale('log')
    # Set the y-range (y-axis limits)
    y_min = cost_mean + 1.5 * max_cost_diff
    y_max = cost_mean - 1.5 * max_cost_diff
    plt.ylim(y_min, y_max)
    plt.grid(True)
    file_name = 'n_runs_cost'
    if suffix != '':
        file_name += '_' + suffix + '.png'
    plt.savefig(file_name)
    plt.show()
    
def exp_non_solvable_1(N = 10, n = 100):
    problem_ = prob_non_solvable_1(N = N)
    n_runs(problem_ = problem_, n = n, suffix = 'non_solvable_1_N_' + str(N) + '_n_' + str(n))
    
def exp_basic_solvable(N = 10, n = 100):
    problem_ = prob_basic_solvable(N = N)
    n_runs(problem_= problem_, n = n, suffix = 'basic_solvable_N_' + str(N) + '_n_' + str(n))
    
    
        
        

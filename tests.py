from examples import *
import problems

TEST_TOL = 1e-10

def basic_test(N = 3, check_grad = False, check_hess = False, tol = TEST_TOL, prints = False, verbosity = 0):
    res_cost_auto, res_cost_manual, exp_min, cost_min = basic_example(N = N, check_grad=check_grad, check_hess = check_hess, prints = prints, verbosity = verbosity, random = True)
    if np.abs(res_cost_auto - exp_min) >= tol:
        print("WARNING: Cost at minimum does not match expected minimum.")
        return False
    if np.abs(res_cost_auto - cost_min) >= tol:
        print("WARNING: Cost at minimum does not match cost at minimum point.")
        return False
    if np.abs(exp_min - cost_min) >= tol:
        print("WARNING: Expected minimum does not match cost at minimum point.")
        return False
    if np.abs(res_cost_manual - exp_min) >= tol:
        print("WARNING: Cost at minimum does not match expected minimum.")
        return False
    if np.abs(res_cost_manual - cost_min) >= tol:
        print("WARNING: Cost at minimum does not match cost at minimum point.")
        return False
    if np.abs(res_cost_auto - res_cost_manual) >= tol:
        print("WARNING: Cost values do not match btw automatic diff and manual gradient/hessian.")
        return False
    print("All cost values match. auto = ", res_cost_auto, " manual = ", res_cost_manual)
    return True

def n_basic_test(n = 10, sizes = [3, 6, 10]):
    for N in sizes:
        for i in range(n):
            if basic_test(N = N) == False:
                return False
            
def problem_vs_example(N = 6, tol = TEST_TOL):
    anp.random.seed(42)
     # Generate 'random' symmetric matrices of size N x N
    A = np.random.rand(N, N)
    A = .5 * (A + A.T)
    B = np.random.rand(N, N)
    B = .5 * (B + B.T)
    # Generate a C for which the problem can be 'solved'
    C = -(A + B)
    horn1 = problems.horn_problem(A, B, C)
    optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
    result1 = optimizer.run(horn1)
    result2 = basic_example(N = N, random = False, prints = False, verbosity = 0)
    res_cost_manual2 = result2[1] 
    if np.abs(res_cost_manual2 - result1.cost) >= tol:
        print("WARNING: cost values btw problems generated with problem module and examples module do not match.")
        return False
    print("All cost values match. problem.py = ", result1.cost, " example.py = ", res_cost_manual2)
    return True
    

    
        

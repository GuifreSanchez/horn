import tests
import problems as prob
from problems import norm2
from problems import basis_change, inner_product
import examples
import experiments
import numpy as np
import csv
from plot_style import *
import pymanopt.optimizers

def right_diagonal_ones(n):
    x = np.zeros((n, n))
    for i in range(n):
        x[i, n - i - 1] = 1
    print(x)
    return x

def horn_cost(point, problem_data):
    A, B, C = problem_data[0], problem_data[1], problem_data[2]
    a, eigv_a = np.linalg.eig(A)
    b, eigv_b = np.linalg.eig(B)
    c, eigv_c = np.linalg.eig(C)
    # Set diagonalized matrices for A, B, C
    A_eigs = np.diag(a)
    B_eigs = np.diag(b)
    C_eigs = np.diag(c) 
    # point is a np.ndarray of shape (2, N, N).
    Q1, Q2 = point[0], point[1]
    A_mod = basis_change(A_eigs, Q1)
    B_mod = basis_change(B_eigs, Q2)
    norm_A2 = inner_product(A, A)
    norm_B2 = inner_product(B, B)
    norm_C2 = inner_product(C, C)
    h = .5 * (norm_A2 + norm_B2 + norm_C2) + inner_product(A_mod, C_eigs) + inner_product(B_mod, C_eigs) + inner_product(A_mod, B_mod)
    return h


PERFORM_TESTS = False
RUN_EXAMPLES = False
RUN_EXPERIMENTS = False
CHECK_GRAD_HESS = False

if CHECK_GRAD_HESS:
    tests.check_grad_hess()

if PERFORM_TESTS:
    tests.problem_vs_example()    
    tests.n_basic_test()
    
if RUN_EXAMPLES:
    examples.ex_non_solvable_1(N = 6, epsilon = 0.75)
    
if RUN_EXPERIMENTS:
    # Ns = [5, 10, 15]
    # ns = [10, 10, 10]
    # experiments.exp_basic_solvable(Ns = Ns, ns = ns)
    # experiments.exp_non_solvable_1(Ns = Ns, ns = ns)
    # experiments.exp_non_solvable_2(Ns = Ns, ns = ns)
    
    # Ns = [5]
    # ns = [10]
    # experiments.exp_non_solvable_1_2(Ns = Ns, ns = ns)
    # Ns = [4, 10]
    # ns = [25, 25]
    # experiments.exp_compare_optimal_points(Ns = Ns, ns = ns)
    # np.random.seed()
    
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    # # Very important! The cost function is not computed directly with the generated matrices
    # # A, B, C, but with their corresponding diagonalizations!
    # for j in range(1):
    #     problem, A_, B_, C_ = prob.prob_abc(N = 5, type = 'non-solvable', seed = True)
    #     # Undiagonalized A, B. Ordered diagonalized C
    #     # opt_guess = experiments.optimal_guess(A, B, C)
    #     experiments.single_run(problem, verbosity=0,line_at = 'traces', A = A_, B = B_, C = C_) 
    # #experiments.exp_basic_solvable(Ns = Ns, ns = ns)
    
    Ns = [5]
    ns = [25]
    experiments.exp_basic_solvable(Ns = Ns, ns = ns)
    
    
# tests.check_grad_hess()

# print("Example of random solvable case")
# Ns = [10]
# ns = [25]
# N = Ns[0]
# #experiments.exp_basic_solvable(Ns = Ns, ns = ns)
# basic_prob = prob.prob_basic_solvable(N = N)
# experiments.n_runs_2(basic_prob, runs = 50, runs_cost = 15, runs_grad = 15, adjust_scale=True)

# print("Example of random non-solvable case")
# Ns = [10]
# ns = [25]
# N = Ns[0]
# #experiments.exp_basic_solvable(Ns = Ns, ns = ns)
# basic_prob = prob.prob_non_solvable_2(N= N)
# experiments.n_runs_2(basic_prob, runs = 50, runs_cost = 15, runs_grad = 15, adjust_scale=True)

# print("Example of random non-solvable case")
# Ns = [5]
# ns = [25]
# experiments.exp_non_solvable_2(Ns = Ns, ns = ns)

# print("Example of non-solvable (controlled) case")
# Ns = [5]
# ns = [25]
# experiments.exp_non_solvable_1(Ns = Ns, ns = ns)

# print("Example of non-solvable (controlled) case")
# Ns = [5]
# ns = [25]
# experiments.exp_non_solvable_1_2(Ns = Ns, ns = ns)

# # Check second-order optimality condition
# print("Second order optimality condition check")
# N = 10
# problem_ = prob.prob_basic_solvable(N = N)
# #problem_ = prob.prob_non_solvable_2(N = N)
# optimizer = pymanopt.optimizers.TrustRegions(verbosity = 1)
# results = optimizer.run(problem_)
# data = results.full_results
# x = data[:, 0]
# x_axis = np.arange(len(x))
# soc1, soc2, soc3, soc4 = data[:, 2], data[:, 3], data[:, 4], data[:, 5]
# # Create the 2D plots
# # Cost vs. iteration
# plt.plot(x_axis, soc1, marker='o', linestyle='-', markersize = 4,label =r'$\Omega_1 = \Omega_2 = \Omega_{12}$')
# plt.plot(x_axis, soc2, marker='o', linestyle='-', markersize = 4,label =r'$\Omega_1 = -\Omega_2 = \Omega_{34}$')
# plt.plot(x_axis, soc3, marker='o', linestyle='-', markersize = 4,label =r'$\Omega_1 = \Omega_{23},\ \Omega_2 = 0$')
# plt.plot(x_axis, soc4, marker='o', linestyle='-', markersize = 4,label =r'$\Omega_2 = \Omega_{14},\ \Omega_1 = 0$')
# plt.xlabel('Iteration')
# plt.ylabel('SOC')
# plt.grid(True)
# plt.legend(loc='upper left', fontsize = 'small')
# plt.savefig('omegas_non_solvable_def.png')
# plt.show()
# experiments.check_second_order_optimality(problem_.data, results.point)

# # Check optimal value guess
# print("Check optimal value guess")
# N = 5
# problem_ = prob.prob_basic_solvable(N = N)
# #problem_ = prob.prob_non_solvable_2(N = N)
# optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
# results = optimizer.run(problem_)
# cost = results.cost
# A, B, C = problem_.data[0], problem_.data[1], problem_.data[2]
# a, eigv_a = np.linalg.eig(A)
# b, eigv_b = np.linalg.eig(B)
# c, eigv_c = np.linalg.eig(C) 
# a = np.sort(a) # increasing order [::-1] for decreasing
# b = np.sort(b)
# c = np.sort(c)
# a, c = a[::-1], c[::-1]
# mu = np.sort(a + b)
# guess = np.sum(c * mu) + np.sum(a * b) + .5 * (norm2(A) + norm2(B) + norm2(C))
# q1, q2 = results.point[0], results.point[1]
# print(cost, guess, horn_cost(results.point, problem_.data))

# Check inequality for A1 B2
# N = 5
# for i in range(10):
#     A = np.random.rand(N, N)
#     B = np.random.rand(N, N)
#     A = .5 * (A + A.T)
#     B = .5 * (B + B.T)
#     a, _ = np.linalg.eig(A)
#     b, _ = np.linalg.eig(B)
#     a, b = np.sort(a)[::-1], np.sort(b)
#     bound_1 = np.trace(A) * np.trace(B) * 2./(2. + N)
#     minimum = np.sum(a * b)
#     print(bound_1, minimum)
    
# # Check new optimal value guess using permutations
# def special(a, b, N = 3, Delta = 1, random_ = False):
#     a_, b_ = [], []
#     if random_:
#         max_int = 20
#         for _ in range(N):
#             a_.append(np.random.randint(max_int))
#             b_.append(np.random.randint(max_int))
#         a = np.sort(np.array(a_))[::-1]
#         b = np.sort(np.array(b_))
#     # a = np.array([6, 5, 4, 3, 2])
#     # b = np.array([1, 3, 5, 7, 9])
#     c = np.array([0,0,0,0,1])
#     A = np.diag(a)
#     B = np.diag(b)
#     # problem_ = prob.horn_problem(A, B, C)
#     # optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
#     # results = optimizer.run(problem_)
#     # cost = results.cost
#     # Check Brockett works
#     # expected_cost = np.sum(a * b) + .5 * (norm2(A) + norm2(B))
#     # print(cost, expected_cost)
#     Delta = Delta
#     c = np.zeros(N)
#     c[0] = Delta
#     C = np.diag(c)
#     problem_ = prob.horn_problem(A, B, C)
#     optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
#     results = optimizer.run(problem_)
#     cost = results.cost
#     point = results.point
#     print("Real optimal cost", cost)
#     sum_ab = a + b
#     min_ab = np.sort(sum_ab)[0]
#     min_a_min_b = a[-1] + b[0] # the smallest possible combination
#     # print(sum_ab)
#     # print(min_ab)
#     import itertools
#     indices = [i for i in range(N)]
#     all_perm_indices = list(itertools.permutations(indices))
#     def permute(permutation, array):
#         result = [array[i] for i in permutation]
#         return result

#     all_perms_a = []
#     all_perms_b = []
#     for permutation in all_perm_indices:
#         all_perms_a.append(np.array(permute(permutation, a)))
#         all_perms_b.append(np.array(permute(permutation, b)))
        
#     norms_const = .5 * (norm2(A) + norm2(B) + norm2(C))
#     min_cost = 1e5
#     opt_perm_a = all_perm_indices[0]
#     opt_perm_b = all_perm_indices[0]
#     opt_a, opt_b = [], []
#     i = 0
#     j = 0
#     index_min_ab = -1
#     for perm_a in all_perms_a:
#         j = 0
#         for perm_b in all_perms_b:
#             current_index_min_ab = np.argmin(perm_a + perm_b)
#             current_min_ab_perm = (perm_a + perm_b)[index_min_ab]
#             current_cost = np.sum(perm_a * perm_b) + norms_const + Delta * current_min_ab_perm
#             if current_cost <= min_cost:
#                 min_cost = current_cost
#                 opt_a = perm_a
#                 opt_b = perm_b
#                 opt_perm_a = all_perm_indices[i]
#                 opt_perm_b = all_perm_indices[j]
#                 # print(opt_perm_b)
#                 # print("Hola", opt_b)
#                 # print(j)
#                 # print(perm_a + perm_b, current_min_ab_perm, current_index_min_ab)
#                 # print(optimal_perm_a)
#                 # print(optimal_perm_b)
#             j += 1
#         i += 1

#     expected_cost_4 = min_cost

#     prod_ab = 0
#     for i in range(N - 1):
#         prod_ab += a[i] * b[i]
        
#     expected_cost_5 = prod_ab + norms_const + Delta * (min_a_min_b) + min(a)*min(b)

#     expected_cost_2 = np.sum(a * b) + .5 * (norm2(A) + norm2(B) + norm2(C)) + Delta * min_ab
#     expected_cost_3 = np.sum(a * b) + .5 * (norm2(A) + norm2(B) + norm2(C)) + Delta * min_a_min_b
#     # print("Expected cost 1", expected_cost_2)
#     # print("Expected cost 2", expected_cost_3)
#     # print("Expected cost perms", expected_cost_4)
#     # print("Expected cost convex", expected_cost_5)
#     #print(point)
#     print("a, b, c used")
#     print(a)
#     print(b)
#     print(c)
    
#     def extract_permutation(x, perm_x):
#         permutation = []
#         x_list = list(x)
#         for el in perm_x:
#             permutation.append(x_list.index(el))
#         return permutation
        

#     # print(opt_perm_a)
#     # print(opt_perm_b)

#     # # opt_perm_a = extract_permutation(a, opt_a)
#     # # opt_perm_b = extract_permutation(b, opt_b)
#     # print(a, "->", opt_a, " :: ", opt_perm_a)
#     # print(b, "->", opt_b, " :: ", opt_perm_b)
#     # print(np.sum(opt_a*opt_b) + norms_const + Delta*np.sort(opt_a + opt_b)[0])

#     # print(opt_a + opt_b)
#     # print(np.argmin(opt_a + opt_b), np.sort(opt_a + opt_b)[0])

#     opt_Q1 = []
#     for i in opt_perm_a:
#         opt_Q1.append(np.eye(N)[i])
#     opt_Q1 = np.array(opt_Q1).T
#     #print(opt_Q1)

#     opt_Q2 = []
#     for i in opt_perm_b:
#         opt_Q2.append(np.eye(N)[i])
#     opt_Q2 = np.array(opt_Q2).T
#     #print(opt_Q2)

#     point = np.array([opt_Q1, opt_Q2])
#     A_mod = basis_change(A, opt_Q1)
#     B_mod = basis_change(B, opt_Q2)
#     #print(A_mod)
#     #print(B_mod)

#     q1, q2 = results.point
#     A1 = basis_change(A, q1)
#     B2 = basis_change(B, q2)
#     inn_test, inn_run, npsum = inner_product(A_mod, B_mod), inner_product(A1,B2), np.sum(opt_a*opt_b)
#     cost_test = problem_.cost(point)
#     cost_run = problem_.cost(results.point)
#     convex_a_b_run = (cost_run - norms_const - inn_run) / Delta
#     convex_a_b_test = (cost_test - norms_const - inn_test) / Delta
#     inn_abc_test = inner_product(A_mod + B_mod, C)
#     inn_abc_run = inner_product(A1 + B2, C)
#     # print("scalar products....")
#     # print(inn_test, inn_run, npsum)
#     # print(inn_abc_test, inn_abc_run)
#     # print("convex combinations....")
#     # print(convex_a_b_test, convex_a_b_run)
#     # print("full costs...")
#     # print(problem_.cost(point))
#     # print(problem_.cost(results.point))
#     C = problem_.data[2]

#     print(opt_a)
#     print(opt_b)
#     # print(results.point[0]@results.point[1].T)
#     # print(results.point[0])
#     # print(results.point[1])
#     print(opt_a + opt_b)
#     test2 = .25 * np.sum(a + b + c)**2
#     print(test2)
#     print(results.cost)
#     c_ord = np.sort(c)[::-1]
#     def extremal(eta1, eta2):
#         return .5 * ((eta1 + c_ord[0])**2 + (eta2 + c_ord[1])**2)
    
#     a_ord = np.sort(a)[::-1]
#     b_ord = np.sort(b)[::-1]
#     n = len(a_ord)
#     delta_a = a_ord[0] - a_ord[1]
#     delta_b = b_ord[0] - b_ord[1]
#     if delta_a <= delta_b:
#         a_ord = np.sort(b)[::-1]
#         b_ord = np.sort(a)[::-1]
#     s = np.sum(a) + np.sum(b)
#     delta_c = c_ord[0] - c_ord[1]
#     eta_min = .5 * (s - delta_c)
#     if eta_min < a_ord[1] + b_ord[1]:
#         eta_min = a_ord[1] + b_ord[1]
#     elif eta_min > a_ord[0] + b_ord[1]:
#         eta_min = a_ord[0] + b_ord[1]
#     eta1 = eta_min
#     eta2 = s - eta1
#     sol = .5*((eta1 + c_ord[0])**2 + (eta2 + c_ord[1])**2)
#     print("extremal", sol)
#     conjecture = test2 * n * .5
#     print("conjecture", conjecture)

# a = np.array([0, 3, 3])
# b = np.array([0, 1, 3])
# # Case doesn't work in 2 dim
# a = np.array([1, 0])
# b = np.array([0, 1])
# delta_a = np.random.randint(100)
# delta_b = delta_a + np.random.randint(10)
# a = np.array([1, delta_a - 1])
# b = np.array([1, delta_b - 1])
# special(a, b, N = 2, Delta = .5 * (delta_a + delta_b) , random_ = False)





# # Check if spectra of A1 + B2 is preserved at SOC points
# print("Check spec(A1 + B2) preserved at SOC points")
# N = 10
# #problem_ = prob.prob_non_solvable_2(N = N)
# #problem_ = prob.prob_basic_solvable(N = N, sort = True)

# a = np.array([5, 5, 4, 3, 1])
# b = np.array([-1, -2, -2, -2, -3])
# c = -(a + b)
# c = np.sort(c)[::-1]

# A = np.diag(a)
# B = np.diag(b)
# C = np.diag(c)

# problem_ = prob.horn_problem(A, B, C)

# problem_data = problem_.data
# #problem_ = prob.prob_non_solvable_2(N = N)
# costs = []
# points = []
# m_spectra = []
# n_runs = 4

# a_ord = np.sort(a)[::-1]
# b_ord = np.sort(b)[::-1]
# c_ord = np.sort(c)[::-1]

# a_inc = a_ord[::-1]
# b_inc = b_ord[::-1]
# c_inc = c_ord[::-1]

# a_ords = [a_ord, a_inc]
# b_ords = [b_ord, b_inc]
# c_ords = [c_ord, c_inc]
# print("Spectral guesses:")
# for alpha in a_ords:
#     for beta in b_ords:
#         for gamma in c_ords:
#             spectral_guess = .5 * (alpha + beta - gamma)
#             spectral_guess = np.sort(spectral_guess)
#             print(spectral_guess)



# A_eigs, B_eigs, C_eigs = A, B, C
# print("---------------------------------")
# print("Actual spectra of A1 + B2 at SOC points")
# for j in range(n_runs):
#     optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
#     results = optimizer.run(problem_)
#     point = results.point
#     cost = results.cost
#     costs.append(results.cost)
#     points.append(point)
#     q1, q2 = point[0], point[1]
#     M = basis_change(A_eigs, q1) + basis_change(B, q2)
#     m, eigv_m = np.linalg.eig(M)
#     m = np.sort(m)
#     m_c = m + c_ord
    
#     cost_2 = .5 * np.sum(np.square(m_c))
#     print(np.sort(m), "{:.4e}".format(cost), "{:.4e}".format(cost_2))
    
# spectral_guess = -1 * c_ord
# print("The guesses")
# print(spectral_guess)
# print(a)
# print(b)
# print(c)
    

# # Check if <A1,B2> is preserved at SOC points
# print("Check <A1,B2> preserved at SOC points")
# N = 10
# problem_ = prob.prob_non_solvable_2(N = N)
# problem_data = problem_.data
# #problem_ = prob.prob_non_solvable_2(N = N)
# costs = []
# points = []
# m_spectra = []
# n_runs = 10

# A, B, C = problem_data[0], problem_data[1], problem_data[2]
# a, eigv_a = np.linalg.eig(A)
# b, eigv_b = np.linalg.eig(B)
# c, eigv_c = np.linalg.eig(C)
# # Set diagonalized matrices for A, B, C
# A_eigs = np.diag(a)
# B_eigs = np.diag(b)
# C_eigs = np.diag(c)

# for j in range(n_runs):
#     optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
#     results = optimizer.run(problem_)
#     point = results.point
#     cost = results.cost
#     costs.append(results.cost)
#     points.append(point)
#     q1, q2 = point[0], point[1]
#     a1b2 = inner_product(basis_change(A_eigs, q1), basis_change(B_eigs, q2))
#     print(a1b2, cost)

# # Check if <A1 + B2,C> is preserved at SOC points and/or if it coincides with 
# # sum_i \lambda_i * \mu_i, where \lambda_1 >= ... >= \lambda_n are the ordered
# # eigenvalues of C and \mu_1 <= ... <= \mu_n the eigenvalues of A1 + B2
# print("Check <A1 + B2,C> and equivalence with sum lambda * mu")
# N = 10
# problem_ = prob.prob_non_solvable_2(N = N)
# problem_data = problem_.data
# #problem_ = prob.prob_non_solvable_2(N = N)
# costs = []
# points = []
# m_spectra = []
# n_runs = 10

# A, B, C = problem_data[0], problem_data[1], problem_data[2]
# a, eigv_a = np.linalg.eig(A)
# b, eigv_b = np.linalg.eig(B)
# c, eigv_c = np.linalg.eig(C)
# # Set diagonalized matrices for A, B, C
# A_eigs = np.diag(a)
# B_eigs = np.diag(b)
# C_eigs = np.diag(c)

# for j in range(n_runs):
#     optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
#     results = optimizer.run(problem_)
#     point = results.point
#     cost = results.cost
#     costs.append(results.cost)
#     points.append(point)
#     q1, q2 = point[0], point[1]
#     a1b2c = inner_product(basis_change(A_eigs, q1) + basis_change(B, q2), C_eigs)
#     print(a1b2, cost)
    
    
# # Check optimal value guess no. 2
# print("Check Brocket optimal value guess")
# N = 3
# costs = []
# guess1s = []
# guess2s = []
# n_probs = 300
# for j in range(n_probs):
#     problem_ = prob.prob_non_solvable_2(N = N)
#     problem_ = prob.prob_basic_solvable(N = N)
#     problem_data = problem_.data

#     A, B, C = problem_data[0], problem_data[1], problem_data[2]
#     a, eigv_a = np.linalg.eig(A)
#     b, eigv_b = np.linalg.eig(B)
#     c, eigv_c = np.linalg.eig(C)
#     # Set diagonalized matrices for A, B, C
#     A_eigs = np.diag(a)
#     B_eigs = np.diag(b)
#     C_eigs = np.diag(c)

#     optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
#     results = optimizer.run(problem_)
#     point = results.point
#     cost = results.cost

#     a_inc, b_inc, c_inc = np.sort(a), np.sort(b), np.sort(c)
#     a_dec, b_dec, c_dec = a_inc[::-1], b_inc[::-1], c_inc[::-1]

#     sum_ab = np.sum(a_inc * b_dec)
#     sum_bc = np.sum(b_inc * c_dec)
#     sum_ca = np.sum(c_inc * a_dec)

#     sum_ab_eq = np.sum(a_inc * b_inc)
#     sum_bc_eq = np.sum(b_inc * c_inc)
#     sum_ca_eq = np.sum(c_inc * a_inc)

#     op_order = sum_ab + sum_bc + sum_ca 
#     eq_order = sum_ab_eq + sum_bc_eq + sum_ca_eq

#     const_term = 0.5 * (norm2(A_eigs) + norm2(B_eigs) + norm2(C_eigs))
#     guess1 = op_order + const_term
#     guess2 = 1./3. * (eq_order + 2. * op_order) + const_term # this doesn't seem to even be a lower bound
#     costs.append(cost)
#     guess1s.append(guess1)
#     guess2s.append(guess2)
#     print(cost, guess1, guess2)
    
# plt.plot(costs, guess2s, marker='o', linestyle='', markersize = 2)
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.grid(True)

# Show the plot
# plt.savefig('single_run_cost.png')
# plt.show()

# # Compare optimal points experiments
# Ns = [5]
# ns = [10]
# experiments.exp_compare_optimal_points(Ns = Ns, ns = ns)
# experiments.exp_compare_optimal_points_2(Ns = Ns, ns = ns)

# # Commutators
# Ns = [5]
# ns = [10]
# N = Ns[0]
# problem_ = prob.prob_basic_solvable(N = N)
# n_runs = 10
# for j in range(n_runs):
#     optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
#     results = optimizer.run(problem_)
#     point = results.point
#     cost = results.cost
#     q1, q2 = point[0], point[1]
#     A, B, C = problem_.reduced_data[0], problem_.reduced_data[1], problem_.reduced_data[2]
#     A1 = basis_change(A, q1)
#     B2 = basis_change(B, q2)

#     comm_ab = A1@B2 - B2@A1
#     comm_bc = B2@C - C@B2
#     comm_ca = C@A1 - A1@C
#     print(comm_ab,"\n")

# Benign non-convexity n = 2
print("Benign non-convexity")
n = 2
a = [np.random.randint(100) for _ in range(n)]
b = [np.random.randint(100) for _ in range(n)]
c = [np.random.randint(100) for _ in range(n)]
a, b, c = np.random.rand(n), np.random.rand(n), np.random.rand(n)
A = np.diag(a)
B = np.diag(b)
C = np.diag(c)


problem_ = prob.horn_problem(A, B, C, sort = True)
A_eigs, B_eigs, C_eigs = problem_.reduced_data
#print(A_eigs,"\n", B_eigs, "\n", C_eigs)
alpha = np.sort(np.diagonal(A_eigs))[::-1]
beta = np.sort(np.diagonal(B_eigs))[::-1]
gamma = np.sort(np.diagonal(C_eigs))[::-1]
# print(alpha)
# print(beta)
# print(gamma)

Delta_alpha = alpha[0] - alpha[1]
Delta_beta = beta[0] - beta[1]
Delta_gamma = gamma[0] - gamma[1]
opt_guess = 0

if Delta_alpha <= Delta_beta:
    if Delta_gamma > Delta_alpha + Delta_beta:
        print("Delta_gamma > Delta_alpha + Delta_beta")
        opt_guess = .5 * ((alpha[1] + beta[1] + gamma[0])**2 + (alpha[0] + beta[0] + gamma[1])**2)
    elif Delta_gamma < Delta_beta - Delta_alpha:
        print("Delta_gamma > Delta_beta - Delta_alpha")
        opt_guess = .5 * ((alpha[0] + beta[1] + gamma[0])**2 + (alpha[1] + beta[0] + gamma[1])**2)
    else:
        print("Delta_gamma in between:", Delta_gamma, Delta_alpha + Delta_beta, Delta_beta - Delta_alpha)
        opt_guess = .25 * (np.sum(alpha) + np.sum(beta) + np.sum(gamma))**2
else:
    if Delta_gamma > Delta_alpha + Delta_beta:
        print("Delta_gamma > Delta_alpha + Delta_beta")
        opt_guess = .5 * ((alpha[1] + beta[1] + gamma[0])**2 + (alpha[0] + beta[0] + gamma[1])**2)
    elif Delta_gamma < Delta_alpha - Delta_beta:
        print("Delta_gamma > Delta_beta - Delta_alpha")
        opt_guess = .5 * ((alpha[1] + beta[0] + gamma[0])**2 + (alpha[0] + beta[1] + gamma[1])**2)
    else:
        print("Delta_gamma in between:", Delta_gamma, Delta_alpha + Delta_beta, Delta_beta - Delta_alpha)
        opt_guess = .25 * np.trace(A + B + C)**2

# Check eta_1, eta_2 do not depend on SOC point
n_runs = 10
print("eta_1, eta_2 for 10 runs...")
for _ in range(n_runs):
    optimizer = pymanopt.optimizers.TrustRegions(verbosity = 0)
    results = optimizer.run(problem_)
    q1, q2 = results.point[0], results.point[1]
    A1 = basis_change(A, q1)
    B2 = basis_change(B, q2)
    M = A1 + B2
    etas, _ = np.linalg.eig(M)
    print(etas)
    
# Check optimal cost coincides with given formulas
print("Final cost value vs. Theorem 2 given global minimum")
print("Final cost: ", results.cost)
print("Th. 2 results: ", opt_guess)




    


    




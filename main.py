import tests
from problems import *
import examples
import experiments
import csv

PERFORM_TESTS = False
RUN_EXAMPLES = False
RUN_EXPERIMENTS = True
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
    
    Ns = [4, 10]
    ns = [25, 25]
    experiments.exp_compare_optimal_points(Ns = Ns, ns = ns)
import tests
from problems import *
import examples
import experiments

PERFORM_TESTS = False
RUN_EXAMPLES = False
RUN_EXPERIMENTS = True
if PERFORM_TESTS:
    tests.problem_vs_example()    
    tests.n_basic_test()
    
if RUN_EXAMPLES:
    examples.ex_non_solvable_1(N = 6, epsilon = 0.75)
    
if RUN_EXPERIMENTS:
    experiments.exp_non_solvable_1(N = 10, n = 100)
    experiments.exp_basic_solvable(N = 10, n = 100)


    



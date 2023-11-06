from problems import *


def ex_non_solvable_1(N = 4, epsilon = 1):
    # Setting non-solvable problem
    A = np.eye(N)
    epsilon = 1
    B = np.random.rand(N, N)
    B = .5 * (B + B.T)
    b, eigv_b = np.linalg.eig(B)
    C = -((1 + epsilon) * np.eye(N) + np.diag(b))
    # Setting and running the problem
    problem = horn_problem(A, B, C)
    optimizer = pymanopt.optimizers.TrustRegions(verbosity = 1)
    result = optimizer.run(problem)
    opt_cost = result.cost
    exp_min = expected_minimum(A, B, C)
    print("opt_cost = ", opt_cost, ", exp_min = ", exp_min)
    return True
    

def basic_example(N = 3, check_grad = False, check_hess = False, prints = True, verbosity = 1, random = True):
    if random == False:
        anp.random.seed(42)
    manifold = pymanopt.manifolds.special_orthogonal_group.SpecialOrthogonalGroup(N, k = 2)
    # Generate 'random' symmetric matrices of size N x N
    A = np.random.rand(N, N)
    A = .5 * (A + A.T)
    B = np.random.rand(N, N)
    B = .5 * (B + B.T)
    # Generate a C for which the problem can be 'solved'
    C = -(A + B)
    # Get spectral decomposition for A, B, C
    a, eigv_a = np.linalg.eig(A)
    b, eigv_b = np.linalg.eig(B)
    c, eigv_c = np.linalg.eig(C)
    # Set diagonalized matrices for A, B, C
    A_eigs = np.diag(a)
    B_eigs = np.diag(b)
    C_eigs = np.diag(c)


    def inner_product(X, Y):
        return np.trace(X.T @ Y)

    def skew_part(matrix):
        result = matrix - matrix.T
        return 0.5 * result

    def basis_change(matrix, orth_matrix):
        return orth_matrix.T @ matrix @ orth_matrix

    @pymanopt.function.autograd(manifold)
    def cost(point):
        # point is a np.ndarray of shape (2, N, N).
        Q1, Q2 = point[0], point[1]
        A_mod = basis_change(A_eigs, Q1)
        B_mod = basis_change(B_eigs, Q2)
        h = inner_product(A_mod, C_eigs) + inner_product(B_mod, C_eigs) + inner_product(A_mod, B_mod)
        return h

    # Expected minimum value for the cost function under consideration, assuming the problem is solvable, 
    # i.e. exists C s.t. A + B + C = 0 and spec(A) = a, spec(B) = b, spec(C) = c, for the given eigenvalues. 
    def expected_minimum(A, B, C):
        result = -0.5*(inner_product(A, A) + inner_product(B, B) + inner_product(C, C))
        return result

    # Riemannian gradient of the cost function
    @pymanopt.function.numpy(manifold)
    def grad(point):
        Q1, Q2 = point[0], point[1]
        A_mod = basis_change(A_eigs, Q1)
        B_mod = basis_change(B_eigs, Q2)
        grad1 = 2 * skew_part(A_mod @ (C_eigs + B_mod))
        grad2 = 2 * skew_part(B_mod @ (C_eigs + A_mod))
        result = np.zeros_like(point)
        result[0] = grad1
        result[1] = grad2
        return result

    # Riemannian Hessian of the cost function
    @pymanopt.function.numpy(manifold)
    def hess(point, direction):
        Q1, Q2 = point[0], point[1]
        Q1_dot, Q2_dot = Q1 @ direction[0], Q2 @ direction[1]
        A_mod = basis_change(A_eigs, Q1)
        B_mod = basis_change(B_eigs, Q2)

        Dgradh1 = Q1_dot @ skew_part(A_mod @ (C_eigs + B_mod))
        temp1 = (Q1_dot.T @ A_eigs @ Q1 + Q1.T @ A_eigs @ Q1_dot) @ (C_eigs + B_mod)
        temp1 += (A_mod @ (Q2_dot.T @ B_eigs @ Q2 + Q2.T @ B_eigs @ Q2_dot))
        Dgradh1 += (Q1 @ skew_part(temp1))
        Dgradh1 = 2 * Dgradh1
        Hess1 = skew_part(Q1.T @ Dgradh1)

        Dgradh2 = Q2_dot @ skew_part(B_mod @ (C_eigs + A_mod))
        temp2 = (Q2_dot.T @ B_eigs @ Q2 + Q2.T @ B_eigs @ Q2_dot) @ (C_eigs + A_mod)
        temp2 += (B_mod @ (Q1_dot.T @ A_eigs @ Q1 + Q1.T @ A_eigs @ Q1_dot))
        Dgradh2 += (Q2 @ skew_part(temp2))
        Dgradh2 = 2 * Dgradh2
        Hess2 = skew_part(Q2.T @ Dgradh2)

        result = np.zeros_like(direction)
        result[0] = Hess1
        result[1] = Hess2

        return result

    # Optimize using automatic diff
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.TrustRegions(verbosity = verbosity)
    result = optimizer.run(problem)
    
    res_cost_auto = result.cost
    exp_min = expected_minimum(A_eigs, B_eigs, C_eigs)
    # Here we use that eigv_a @ diag(a) @ eigv_a.T = A 
    cost_min = cost([eigv_a.T @ eigv_c, eigv_b.T @ eigv_c])
    if prints:
        # The actual matrices Q1, Q2
        print("Optimization using automatic differentiation")
        print("--------------------------------------------")
        print("Optimal point")
        print(result.point)
        
        print("Cost values")
        print(res_cost_auto)
        
        print("Expected minimum")
        print(exp_min)
        
        print("Cost at minimum")
        print(cost_min)

        print("Custom gradient/hessian check...")
        if check_grad:
            problem_custom_gradient = pymanopt.Problem(manifold, cost = cost, riemannian_gradient = grad)
            check_gradient(problem_custom_gradient)
        if check_hess:
            problem_custom_hessian = pymanopt.Problem(manifold, cost = cost, riemannian_hessian = hess)
            check_hessian(problem_custom_hessian)
    
    problem_custom = pymanopt.Problem(manifold, cost = cost, riemannian_gradient = grad, riemannian_hessian = hess)
    optimizer = pymanopt.optimizers.TrustRegions(verbosity = verbosity)
    result = optimizer.run(problem_custom)
    
    res_cost_manual = result.cost
    if prints:
        print("Optimization using custom Riemannian gradient/hessian")
        print("-----------------------------------------------------")

        print("Optimal point")
        print(result.point)
        
        print("Cost values")
        print(res_cost_manual)
        
        print("Expected minimum")
        print(exp_min)
        
        print("Cost at minimum")
        print(cost_min)
    
    return res_cost_auto, res_cost_manual, exp_min, cost_min
    

    
# basic_example(N = 6)

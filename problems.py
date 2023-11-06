
import autograd.numpy as anp
import numpy as np
import pymanopt
import pymanopt.manifolds
from scipy.stats import special_ortho_group
import pymanopt.optimizers
from pymanopt.tools.diagnostics import check_gradient, check_hessian

def inner_product(X, Y):
    return np.trace(X.T @ Y)

def skew_part(matrix):
    result = matrix - matrix.T
    return 0.5 * result

def basis_change(matrix, orth_matrix):
    return orth_matrix.T @ matrix @ orth_matrix

def riem_grad(point, A_eigs, B_eigs, C_eigs):
    Q1, Q2 = point[0], point[1]
    A_mod = basis_change(A_eigs, Q1)
    B_mod = basis_change(B_eigs, Q2)
    grad1 = 2 * skew_part(A_mod @ (C_eigs + B_mod))
    grad2 = 2 * skew_part(B_mod @ (C_eigs + A_mod))
    result = np.zeros_like(point)
    result[0] = grad1
    result[1] = grad2
    return result

def riem_hess(point, direction, A_eigs, B_eigs, C_eigs):
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

# Expected minimum value for the cost function under consideration, assuming the problem is solvable, 
# i.e. exists C s.t. A + B + C = 0 and spec(A) = a, spec(B) = b, spec(C) = c, for the given eigenvalues. 
def expected_minimum(A, B, C):
    result = -0.5*(inner_product(A, A) + inner_product(B, B) + inner_product(C, C))
    return result

def horn_problem(A, B, C, auto = False):
    N = A.shape[0]
    manifold = pymanopt.manifolds.special_orthogonal_group.SpecialOrthogonalGroup(N, k = 2)
    # Get spectral decomposition for A, B, C
    a, eigv_a = np.linalg.eig(A)
    b, eigv_b = np.linalg.eig(B)
    c, eigv_c = np.linalg.eig(C)
    # Set diagonalized matrices for A, B, C
    A_eigs = np.diag(a)
    B_eigs = np.diag(b)
    C_eigs = np.diag(c)
    
    @pymanopt.function.autograd(manifold)
    def cost(point):
        # point is a np.ndarray of shape (2, N, N).
        Q1, Q2 = point[0], point[1]
        A_mod = basis_change(A_eigs, Q1)
        B_mod = basis_change(B_eigs, Q2)
        h = inner_product(A_mod, C_eigs) + inner_product(B_mod, C_eigs) + inner_product(A_mod, B_mod)
        return h
    
    # Riemannian gradient of the cost function
    @pymanopt.function.numpy(manifold)
    def grad(point):
        return riem_grad(point, A_eigs, B_eigs, C_eigs)

    # Riemannian Hessian of the cost function
    @pymanopt.function.numpy(manifold)
    def hess(point, direction):
        return riem_hess(point, direction, A_eigs, B_eigs, C_eigs)

    if auto:
        # Optimize using automatic diff
        result = pymanopt.Problem(manifold, cost)
    
    result = pymanopt.Problem(manifold, cost = cost, riemannian_gradient = grad, riemannian_hessian = hess)
    return result


def prob_non_solvable_1(N = 4, epsilon = 1):
    # Setting non-solvable problem
    A = np.eye(N)
    epsilon = 1
    B = np.random.rand(N, N)
    B = .5 * (B + B.T)
    b, eigv_b = np.linalg.eig(B)
    C = -((1 + epsilon) * np.eye(N) + np.diag(b))
    # Setting the problem
    problem = horn_problem(A, B, C)
    return problem

def prob_basic_solvable(N = 4):
    # Generate 'random' symmetric matrices of size N x N
    A = np.random.rand(N, N)
    A = .5 * (A + A.T)
    B = np.random.rand(N, N)
    B = .5 * (B + B.T)
    # Generate a C for which the problem can be 'solved'
    C = -(A + B)
    problem = horn_problem(A, B, C)
    return problem
    
    
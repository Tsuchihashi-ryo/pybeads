import numpy as np
from scipy.sparse import spdiags, dia_matrix, vstack
from numba import jit, float64, int64, boolean

@jit(float64[:,:](float64[:], int64, float64, float64, int64, float64, float64, float64, str), nopython=True, parallel=True)
def beads(y, d, fc, r, Nit, lam0, lam1, lam2, pen):
    """BEADS: Baseline estimation and denoising using sparsity"""
    
    EPS0 = 1e-6
    EPS1 = 1e-6
    
    if pen == 'L1_v1':
        @jit(float64(float64), nopython=True)
        def phi(xx):
            return np.sqrt(np.power(abs(xx), 2) + EPS1)
        
        @jit(float64(float64), nopython=True)
        def wfun(xx):
            return 1. / np.sqrt(np.power(abs(xx), 2) + EPS1)
        
    elif pen == 'L1_v2':
        @jit(float64(float64), nopython=True)
        def phi(xx):
            return abs(xx) - EPS1 * np.log(abs(xx) + EPS1)
        
        @jit(float64(float64), nopython=True)
        def wfun(xx):
            return 1. / (abs(xx) + EPS1)
    
    @jit(float64(float64[:]), nopython=True, parallel=True)
    def theta(xx):
        sum_pos = 0.0
        sum_neg = 0.0
        sum_small = 0.0
        for x in xx:
            if x > EPS0:
                sum_pos += x
            elif x < -EPS0:
                sum_neg += x
            else:
                sum_small += (1+r)/(4*EPS0) * x**2 + (1-r)/2 * x + EPS0*(1+r)/4
        return sum_pos - r * sum_neg + sum_small
    
    N = len(y)
    A, B = BAfilt(d, fc, N)
    H = lambda xx: B.dot(linv(A, xx))
    D1, D2 = make_diff_matrices(N)
    D = vstack([D1, D2])
    BTB = B.T.dot(B)
    
    w = np.vstack(([lam1 * np.ones(N-1), lam2 * np.ones(N-2)])).astype(np.float64)
    b = ((1-r) / 2 * np.ones(N)).astype(np.float64)
    d = BTB.dot(linv(A, y.astype(np.float64))) - lam0 * A.T.dot(b)
    
    gamma = np.ones(N, dtype=np.float64)
    x = y.astype(np.float64)
    cost = np.zeros(Nit)
    
    for i in range(Nit):
        diff = D.dot(x)
        wf = w * wfun(diff)
        Lmda = spdiags(wf.T, 0, 2*N-3, 2*N-3)
        
        k = np.abs(x) > EPS0
        gamma[~k] = ((1 + r) / 4) / EPS0
        gamma[k] = ((1 + r) / 4) / np.abs(x[k])
        Gamma = spdiags(gamma, 0, N, N)
        
        M = 2 * lam0 * Gamma + (D.T.dot(Lmda)).dot(D.T)
        x = A.dot(linv(BTB + A.T.dot(M.dot(A)), d))
        
        a = y - x
        cost[i] = 0.5 * np.sum(np.abs(H(a))**2) + lam0 * theta(x) + lam1 * np.sum(phi(np.diff(x))) + lam2 * np.sum(phi(np.diff(x, n=2)))
    
    f = y - x - H(y - x)
    
    return np.asarray(x).reshape(-1, 1), np.asarray(f).reshape(-1, 1), cost

@jit(nopython=True)
def linv(a, b):
    return np.linalg.spsolve_triangular(a.T, b, lower=True).reshape(-1, 1)

@jit(nopython=True)
def make_diff_matrices(N):
    e = np.ones(N-1, dtype=np.float64)
    D1 = spdiags([-e, e], [0, 1], N-1, N)
    D2 = spdiags([e, -2*e, e], range(3), N-2, N)
    D1[-1, -1], D2[-1, -1] = 1., 1.
    return D1, D2

@jit(nopython=True)  
def BAfilt(d, fc, N):
    b1 = np.array([1, -1], dtype=np.float64)
    for i in range(d):
        b1 = np.convolve(b1, [-1, 2, -1])
    
    b = np.convolve(b1, [-1, 1])
    
    omc = 2 * np.pi * fc
    t = np.power(((1 - np.cos(omc)) / (1 + np.cos(omc))), d)
    
    a = np.array([1], dtype=np.float64)
    for i in range(d):
        a = np.convolve(a, [1, 2, 1])
    
    a = b + t * a
    xa, xb = a, b
    dr = np.arange(-d, d+1)
    A = spdiags(xa, dr, N, N)
    B = spdiags(xb, dr, N, N)
    
    return A, B

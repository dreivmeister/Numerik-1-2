import numpy as np
from Numerik_1 import derivative, qr_decomposition_householder

# eindimensionale kondition
def absolute_kondition(f, x):
    return abs(derivative(f, x, mode='central'))

def relative_kondition(f, x):
    return absolute_kondition(f, x) * abs(x / f(x))

def zeilensummennorm(A):
    return np.max(np.sum(np.asarray(A),axis=1))

def spaltensummennorm(A):
    return np.max(np.sum(np.asarray(A),axis=1))

def frobeniusnorm(A):
    return np.sqrt(np.sum(np.abs(np.asarray(A))**2))

def spektralnorm(A):
    return np.sqrt(np.max(np.linalg.eigvals(np.dot(A.T,A))))

def matrix_cond(A):
    return zeilensummennorm(A) * zeilensummennorm(np.inv(A))

def fixpunktiteration(f, x0, maxiter=100, tol=1e-6):
    i = 1
    while i <= maxiter:
        xn = f(x0)
        if abs(xn - x0) < tol:
            return xn
        x0 = xn
        i += 1
    return xn

def newton_1d(f, x0, fp=None, maxiter=100, tol=1e-6):
    if fp == None:
        from autograd import grad
        fp = grad(f)
    i = 1
    while i <= maxiter:
        xn = x0 - f(x0)/fp(x0)
        if abs(f(xn)) < tol:
            return xn
        x0 = xn
        i += 1
    return xn

def newton_damped_1d(f, x0, fp=None, maxiter=100, tol=1e-6):
    if fp == None:
        from autograd import grad
        fp = grad(f)
    i = 1
    l = 1
    while i <= maxiter:
        xn = x0 - l*f(x0)/fp(x0)
        if abs(f(xn)) < tol:
            return xn
        x0 = xn
        i += 1
        l /= 2
    return xn

# import autograd.numpy as np
# from autograd import grad
def J(f, x, dx=1e-8):
    n = len(x)
    func = f(x)
    jac = np.zeros((n, n))
    for j in range(n):  # through columns to allow for vector addition
        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (f(x_plus) - func)/Dxj
    return jac
    
    
def newton_nd(f, x0, jac=None, maxiter=100, tol=1e-6):
    i = 1
    while i <= maxiter:
        jac = J(f, x0)
        xn = np.linalg.solve(jac, np.dot(jac, x0) - f(x0))
        if abs(f(xn)).all() < tol:
            return xn
        x0 = xn
        i += 1
    return xn


def f(x):
    return np.array([
        1 - x[0]**2 - x[1]**2,
        (x[0] - 2*x[1]) / (1 / (2 + x[1]))
    ])

# print(J(f, (1.,1.)))

# print(newton_nd(f, (1.,1.)))




def jacobi_verfahren_matrix(A, b, x0, maxiter=100, tol=1e-6):
    L = np.tril(A,k=-1)
    R = np.triu(A,k=1)
    D_1 = np.diag(1 / np.diag(A)) # D inverse
    M = -np.dot(D_1,L+R)
    c = np.dot(D_1,b)
    i = 1
    while i <= maxiter:
        xn = np.dot(M, x0) + c 
        if abs(xn - x0).all() < tol:
            return xn
        x0 = xn
        i += 1
    return xn

def jacobi_verfahren_index(A, b, x0, maxiter=100, tol=1e-6):
    n = len(x0)
    x = x0.copy()
    while np.linalg.norm(b - A @ x) > tol:
        x_new = np.zeros(n)
        for k in range(n):
            # old vals
            left = sum(A[k,:k] * x[:k]) 
            right = sum(A[k,k+1:] * x[k+1:]) 
            x_new[k] = (b[k] - left - right) / A[k,k]
        x = x_new.copy()
    return x_new

def JOR(A, b, x0, w, maxiter=100, tol=1e-6):
    n = len(x0)
    x = x0.copy()
    while np.linalg.norm(b - A @ x) > tol:
        x_new = np.zeros(n)
        for k in range(n):
            # old vals
            left = sum(A[k,:k] * x[:k]) 
            right = sum(A[k,k+1:] * x[k+1:])
             
            x_new[k] = w * (b[k] - left - right) / A[k,k] + (1 - w) * x[k]
        x = x_new.copy()
    return x_new

def gauss_seidel_verfahren_index(A, b, x0, maxiter=100, tol=1e-6):
    n = len(x0)
    x = x0.copy()
    while np.linalg.norm(b - A @ x) > tol:
        for k in range(n):
            left = sum(A[k,:k] * x[:k]) # new vals
            right = sum(A[k,k+1:] * x[k+1:]) # old vals
            x[k] = (b[k] - left - right) / A[k,k]
    return x

def SOR(A, b, x0, w, maxiter=100, tol=1e-6):
    n = len(x0)
    x = x0.copy()
    while np.linalg.norm(b - A @ x) > tol:
        for k in range(n):
            left = sum(A[k,:k] * x[:k]) # new vals
            right = sum(A[k,k+1:] * x[k+1:]) # old vals
            x_tmp = (b[k] - left - right) / A[k,k]
            x[k] = w * x_tmp + (1 - w) * x[k]
    return x
            


# print(jacobi_verfahren_matrix(np.array([[2,1],[5,7]]),np.array([11,13]),np.array([1,1])))
# print(jacobi_verfahren_index(np.array([[2,1],[5,7]]),np.array([11,13]),np.array([1,1])))
# print(gauss_seidel_verfahren_index(np.array([[2,1],[5,7]]),np.array([11,13]),np.array([0.,0.])))
    

def gauss_seidel_verfahren_matrix(A, b, x0, maxiter=100, tol=1e-6):
    DL = np.tril(A,k=0)
    R = np.triu(A,k=1) 
    DL_1 = np.linalg.inv(DL)
    
    M = -np.dot(DL_1,R)
    c = np.dot(DL_1,b)
    i = 1
    while i <= maxiter:
        xn = np.dot(M, x0) + c 
        if abs(xn - x0).all() < tol:
            return xn
        x0 = xn
        i += 1
    return xn


#print(gauss_seidel_verfahren(np.array([[2,1],[5,7]]),np.array([11,13]),np.array([1,1])))


#potenzmethode, potenzmethode für symmetrische
def potenzmethode_mises(A, x0, maxiter=100):
    for k in range(maxiter):
        xn = np.dot(A,x0)
        if k > maxiter-2:
            break
        xn = xn / np.linalg.norm(xn)
        x0 = xn
    return xn[0] / x0[0]

#print(potenzmethode_mises(np.array([[2,1,2],[-1,2,1],[1,2,4]]),np.array([1.,1.,1.])))

def inverse_iteration_mit_shift(A, s, x0, maxiter=100):
    n = len(A)
    A = A - s*np.identity(n)
    for k in range(maxiter):
        xn = np.linalg.solve(A,x0)
        if k > maxiter-2:
            break
        xn = xn / np.linalg.norm(xn)
        x0 = xn
    return s + (x0[0] / xn[0])

# print(inverse_iteration_mit_shift(np.array([[2,-0.1,0.4],[0.3,-1.,0.4],[0.2,-0.1,4]]), 2, np.array([1.,1.,1.])))
# print(inverse_iteration_mit_shift(np.array([[2,-0.1,0.4],[0.3,-1.,0.4],[0.2,-0.1,4]]), -1, np.array([1.,1.,1.])))
# print(inverse_iteration_mit_shift(np.array([[2,-0.1,0.4],[0.3,-1.,0.4],[0.2,-0.1,4]]), 4, np.array([1.,1.,1.])))
    
#inverse iteration mit shift
#qr verfahren
#lr verfahren

def qr_iteration(A, maxiter=20):
    for _ in range(maxiter):
        Q,R = qr_decomposition_householder(A)
        A = np.dot(R,Q)
    return np.diag(A)

A = np.array([[ 12, -51,   4],
              [  6, 167, -68],
              [ -4,  24, -41]])

print(qr_iteration(A))
import numpy as np

def horner_schema(f_c, x0):
    # Seite 15
    # f_c - list/np.ndarray - polynomial coefficients in form: [a_0, ..., a_n]
    # x0 - int/float - evaluation point f(x0)
    # returns: f_x - f(x0)
    n = len(f_c)
    
    f_x = f_c[-1]
    for i in range(n-1, -1, -1):
        f_x = f_x * x0 + f_c[i]
    return f_x

def erweitertes_horner_schema(f_c, x0):
    # Seite 16
    # f_c - list/np.ndarray - polynomial coefficients in form: [a_0, ..., a_n]
    # x0 - int/float - evaluation point f(x0)
    # returns: f_x - f(x0) and polynomial coefficients in form: [b_0, ..., b_n-1]
    n = len(f_c)
    
    b_c = [f_c[-1]]
    for i in range(n-1, -1, -1):
        b_c.append(b_c[-1] * x0 + f_c[i])
    f_x = b_c[-1] * x0 + f_c[-1]
    return f_x, b_c[::-1]
    
def doppelzeiliges_horner_schema(f_c, p, q):
    # Seite 17
    # f_c - list/np.ndarray - polynomial coefficients in form: [a_0, ..., a_n]
    # x0 - int/float - evaluation point f(x0)
    # returns: A and B and polynomial coefficients in form: [b_0, ..., b_n-1]
    n = len(f_c)
    
    c_c = [f_c[-1], f_c[-1] * p + f_c[-2]]
    for j in range(n-2, 1, -1):
        c_c.append(c_c[j] * q + c_c[j-1] * p + f_c[j])
    A = c_c[-2] * q + c_c[-1] * p + f_c[-2]
    B = c_c[-1] * q + f_c[-1]
    return A, B, c_c[::-1]

def newton_horner_schema(x_c, c_c, x0):
    # Seite 20
    # x_c - list/np.ndarray - nodes of interpolation polynomial: [x_0, ..., x_n-1]
    # c_c - list/np.ndarray - interpolation polynomial coefficients: [c_0, ..., c_n]
    # returns: y - f(x0) of interpolation polynomial f in newton form at x0
    n = len(c_c)
    
    y = c_c[-1]
    for j in range(n-1, -1, -1):
        y = y * (x0 - x_c[j]) + c_c[j]
    return y

def newton_schema(x, y):
    # Seite 22
    # x - [x_0, ..., x_n]
    # y - [y_0, ..., y_n]
    # returns: coefficients of newton interpolation polynomial: [d_0, ..., d_n]
    n = len(x)
    
    d = np.zeros((n,n))
    for i in range(n):
        d[i,0] = y[i]
    
    for j in range(1, n):
        for i in range(n-j):
            d[i,j] = (d[i,j-1] - d[i+1,j-1]) / (x[i] - x[i+j])
    return d[0,:]

#print(newton_schema([0,1,2,3],[-1,0,5,20]))
            

# TODO: hermite interpolation

# interpolation in monombasis mit vandermonde matrix
def interpolation_vandermonde(x, y):
	V = np.vander(x)
	return np.linalg.solve(V,y)

def generate_chebyshev_nodes(a, b, n):
    # interpolate function on interval [a,b] with order n
    # returns the nodes [x0, ..., xn] for interpolation (n+1)
    # Seite 32
    f1 = (b-a)/2
    f2 = (a+b)/2
    f3 = 2*n+2
    return [f1 * np.cos((2*j+1)/f3 * np.pi) + f2 for j in range(n+1)]


def linear_splines(x, y):
    # generates linear_splines
    n = len(x) # num nodes
    # deg = n-1 # deg of poly
    
    a = y[:-1]
    b = [(y[j+1] - y[j]) / (x[j+1] - x[j]) for j in range(n)] # first order divided difference
    
    return a, b

from scipy.sparse import diags
def gen_mat_cubic(h, mode):
    # replace scipy maybe
    n = len(h)
    if mode == 'natural':
        diagonals = [[2*(h[i]+h[i+1]) for i in range(n-1)], h[1:-1], h[1:-1]]
    elif mode == 'complete':
        diagonals = [[2*h[0]] + [2*(h[i]+h[i+1]) for i in range(n-1)] + [2 * h[-1]], h, h]
    elif mode == 'periodic':
        diagonals = [[2 * (h[-1] + h[0])] + [2*(h[i]+h[i+1]) for i in range(n-1)] + [2 * (h[-2] + h[-1])], h[:-1], h[:-1]]
    return diags(diagonals, [0, -1, 1]).toarray()
    
def gen_rhs_cubic(y, h, mode, yp=None):
    n = len(h)
    g = [(6*(y[j+1]-y[j]))/h[j] - (6*(y[j]-y[j-1]))/h[j-1] for j in range(1,n)]
    if mode == 'complete':
        g = [-6*yp[0] + (6*(y[1]-y[0]))/h[0]] + g + [6*yp[-1] + (6*(y[-1]-y[-2]))/h[-1]]
    elif mode == 'periodic':
        g = [(6*(y[1]-y[0]))/h[0] - (6*(y[-1]-y[-2]))/h[-1]] + g
    return np.asarray(g)
    
def cubic_splines(x, y, mode='natural', yp=None):
    # returns a, b, c, d the coeffs for cubic splines len(a) == n
    # n cubic splines with n+1 nodes of degree 3 (cubic)
    # modes can be ['natural','complete','periodic']
    n = len(x) # num nodes
    h = [x[i+1]-x[i] for i in range(n-1)] # len(h) = n-1
    print(h)
    
    rhs = gen_rhs_cubic(y, h, mode, yp)
    mat = gen_mat_cubic(h, mode)
    
    print(mat.shape)
    print(rhs.shape)
    
    s = np.linalg.solve(mat, rhs)
    
    a = y[:-1]
    b = [(y[j+1]-y[j])/h[j] - h[j]*(s[j+1]+2*s[j])/6 for j in range(n-1)]
    c = [s[j]/2 for j in range(n)]
    d = [(s[j+1]-s[j])/(6*h[j]) for j in range(n-2)]
    # len(a) = len(b) = len(c) = len(d)
    assert len(a) == len(b) == len(c) == len(d), 'wrong'
    
    return a, b, c, d



#https://stackoverflow.com/questions/31543775/how-to-perform-cubic-spline-interpolation-in-python
def compute_coeffs(moments, y, h):
    coeffs = []
    for j in range(len(y)-1):
        coeffs.append([
            y[j], # aj
            (y[j+1]-y[j])/h[j]-(h[j]/6)*(moments[j+1]+2*moments[j]), # bj
            moments[j]/2, # cj
            (moments[j+1]-moments[j])/(6*h[j]) # dj
            ]) 
    return coeffs
        
def compute_h(x):
    return [x[i+1]-x[i] for i in range(len(x)-1)]

def fill_matrix(h, v, mode):
    n = len(v) # 2<-n-1 3<-n 4<-number of points
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = v[i]
    for i in range(n-1):
        A[i+1,i] = h[i]
        A[i,i+1] = h[i]
    return A
    
    

def compute_v(h):
    return [2*(h[i-1]+h[i]) for i in range(1,len(h))]

def compute_rhs(y, h, mode, yp=None):
    return [(6*(y[j+1]-y[j]))/h[j-1] - (6*(y[j]-y[j-1]))/h[j] for j in range(1,len(y)-1)]


def thomas_algorithm(A,b):
    # TDMA
    # elim
    for i in range(1, A.shape[0]):
        m = A[i,i-1]/A[i-1,i-1]
        A[i,i] = A[i,i] - m*A[i-1,i]
        b[i] = b[i] - m*b[i-1]
    # back sub
    x = [b[-1]/A[-1,-1]]
    for i in range(A.shape[0]-2, -1, -1):
        x.append((b[i]-A[i,i+1]*x[-1])/A[i,i])
    return x

def cubic_spline_interpolation(x, y, yp=None, mode='natural'):
    h = compute_h(x)
    u = compute_rhs(y, h, mode, yp) # rhs
    
    v = compute_v(h)
    A = fill_matrix(h, v, mode)
    
    z = thomas_algorithm(A, u)
    z = [0] + z + [0]
    z = z[::-1]
    return compute_coeffs(z, y, h)



# n+1 - Anzahl der Knoten (4)
# n - Anzahl Intervalle/Anzahl Splines (3)
# natürlich: nxn Matrix
#print(cubic_spline_interpolation([0,1,2,3],[-1,0,5,20]))
# print(cubic_spline_interpolation([0,1,2,3],[-1,0,5,20], [1,1], mode='complete'))
# print(cubic_spline_interpolation([0,1,2,3],[-1,0,5,20], mode='periodic'))



def good_h(x, mp=2.22044604925e-16):
    # mp is machine precision of python float = np.finfo(float).eps
    #https://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon
    return np.abs(x) * mp ** (1/3)

def derivative(f, x, h=1e-6, mode='forward'):
    if mode == 'forward':
        return (f(x+h) - f(x)) / h
    elif mode == 'backward':
        return (f(x) - f(x-h)) / h
    elif mode == 'central':
        return (f(x+h) - f(x-h)) / (2*h)


def ai0re(f,x0,h):
    return (f(x0+h)-f(x0))/h
def aikre(h,a,i,k):
    m = h[i-k]/h[i]
    a = m*a[i][k-1]-a[i-1][k-1]
    return a/(m-1)
def richardson_extrapolation(f, x0, n, k):
    h = np.array([2**(-i) for i in range(n)])
    a = np.zeros((n,k))
    
    # erste spalte füllen
    for i in range(a.shape[0]):
        a[i,0] = ai0re(f,x0,h[i])
                
    # rest füllen
    last_val = (0,0)
    for k in range(1,a.shape[1]):
        for i in range(k,a.shape[0]):
            a[i,k] = aikre(h,a,i,k)
            if a[i,k] != float(0):
                last_val = (i,k)
    return a[last_val[0], last_val[1]]


from math import sin,cos
# def f(x):
#     return sin(x)
# x0 = 1
# n = 5
# k = 6
        
# print(richardson_extrapolation(f,x0,n,k))
# print(cos(x0))


def summierte_trapezregel(f, a, b, n):
    x = np.linspace(a,b,n)
    h = x[1]-x[0]
    return h/2 * (f(a) + 2 * sum(f(x[1:-1])) + f(b))

def summierte_simpson(f, a, b, n):
    x = np.linspace(a,b,n)
    h = x[1]-x[0]
    s = np.asarray([f((x[i]+x[i+1]) / 2) for i in range(len(x)-1)])
    return h/6 * (f(a) + 2 * sum(f(x[1:-1])) + 4 * sum(s) + f(b))

def summierte_mittelpunkt(f, a, b, n):
    x = np.linspace(a,b,n)
    h = x[1]-x[0]
    s = [f((x[i]+x[i+1]) / 2) for i in range(len(x)-1)]
    return h * sum(s)


def gauß_quadrature(f, a, b, n):
    if n == 1:
        x     = [0]
        alpha = [2]
    if n == 2:
        x     = [-0.57735026919,0.57735026919]
        alpha = [1,1]
    if n == 3:
        x     = [-0.774596669241,0,0.774596669241]
        alpha = [0.555555555556,0.888888888889,0.555555555556]
    # add n=4 and n=5
    if n == 4:
        x     = [-0.861136311594053,-0.339981043584856,0.339981043584856,0.861136311594053]
        alpha = [0.347854845137454,0.652145154862546,0.652145154862546,0.347854845137454]
    
    I = 0
    for i in range(n):
        I += f((b-a)/2 * x[i] + (a+b)/2)*alpha[i]
    I *= (b-a)/2
    
    return I


def ai0ri(f, h, a, b):
        s = 0
        i = a + h
        while i <= (b - h):
            s += f(i)
            i += h
        s += (1 / 2) * f(a)
        s += (1 / 2) * f(b)
        return h * s
def aikri(h, a, i, k):
    m = (h[i - k] / h[i])**2
    a = m * a[i][k - 1] - a[i - 1][k - 1]
    return a / (m - 1)
def romberg_integration(f,a,b,n,k):    
    h = np.array([2**(-i) for i in range(n)])
    t = np.zeros((n, k))

    # erste spalte füllen
    for i in range(t.shape[0]):
        t[i, 0] = ai0ri(f, h[i], a, b)

    # rest füllen
    last_val = (0, 0)
    for k in range(1, t.shape[1]):
        for i in range(k, t.shape[0]):
            t[i, k] = aikri(h, t, i, k)
            if t[i, k] != float(0):
                last_val = (i, k)
    return t[last_val]


# def f(x):
#     return np.sin(x)
# print(summierte_trapezregel(f, 0, 10, 100))
# print(summierte_simpson(f, 0, 10, 100))
# print(summierte_mittelpunkt(f, 0, 10, 100))
        
def back_substitution(R,b):
    # Seite 58
    # solves Rx = b with R being an upper triangular matrix
    n = len(b)
    x = np.zeros(n)
    x[-1] = b[-1] / R[-1,-1]
    for i in range(n-2,-1,-1):
        s = 0
        for j in range(i+1,n):
            s += R[i,j] * x[j]
        x[i] = (b[i] - s) / R[i,i]
    return x
    
def forward_sub(L,b):
    # Seite 59
    # solves Lx = b with L being an lower triangular matrix
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0] / L[0,0]
    for i in range(1, n):
        s = 0
        for j in range(i):
            s += L[i,j] * x[j]
        x[i] = (b[i] - s) / L[i,i]
    return x

def LUP_partial_pivoting(matrix):
    # lu with partial pivoting
    n, _ = matrix.shape
    U = matrix.copy()
    PF = np.identity(n)
    LF = np.zeros((n,n))
    for k in range(0, n-1):
        index = np.argmax(abs(U[k:,k])) # find abs max of curr col
        index = index + k
        
        # pivoting
        if index != k:
            P = np.identity(n)
            P[[index,k],k:n] = P[[k,index],k:n]
            U[[index,k],k:n] = U[[k,index],k:n] 
            PF = np.dot(P,PF)
            LF = np.dot(P,LF)
        
        #frobenius
        L = np.identity(n)
        for j in range(k+1,n):
            t = U[j,k] / U[k,k]
            L[j,k] = -t
            LF[j,k] = t
        U = np.dot(L,U)
    np.fill_diagonal(LF, 1)
    return PF, LF, U

def LU(matrix):
    # lu without pivoting
    n, _ = matrix.shape
    U = matrix.copy()
    LF = np.zeros((n,n))
    for k in range(0, n-1):
        #frobenius
        L = np.identity(n)
        for j in range(k+1,n):
            t = U[j,k] / U[k,k]
            L[j,k] = -t
            LF[j,k] = t
        U = np.dot(L,U)
    np.fill_diagonal(LF, 1)
    return LF, U
    

def crouts_algorithm(A):
    # Seite 65
    # doesnt work
    n = A.shape[0]
    L = np.identity(n)
    R = np.identity(n)
    
    for j in range(n):
        for i in range(j):
            R[i,j] = A[i,j] - sum([L[i,k] * R[k,j] for k in range(i)])
        for i in range(j+1,n):
            L[i,j] = (A[i,j] - sum([L[i,k] * R[k,j] for k in range(j)])) / R[j,j]
    return L, R
            
    
# A = np.array([[1,2],
#               [3,4]])

# L,U = LU(A)
# print(L, U)



# L1,U1 = crouts_algorithm(A)
# print(np.dot(L1,U1))
    
    
def cholesky_decomposition(A):
    # assume for now A is symmetric and positive definite
    # Seite 68
    n = A.shape[0]
    L = np.zeros((n,n))
    
    for j in range(n):
        # diagonal
        s = 0
        for k in range(j):
            s += L[j,k] ** 2
        L[j,j] = np.sqrt(A[j,j] - s)
        
        # non-diagonal
        for i in range(j+1, n):
            s = 0
            for k in range(j):
                s += L[i,k] * L[j,k]
            L[i,j] = ((A[i,j] - s) / L[j,j])
    return L
        
A = np.array([[4,12,-16],
              [12,37,-43],
              [-16,-43,98]])

# print(cholesky_decomposition(A))


def householder(a):
    """Use this version of householder to reproduce the output of np.linalg.qr 
    exactly (specifically, to match the sign convention it uses)
    
    based on https://rosettacode.org/wiki/QR_decomposition#Python
    """
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    
    return v


def qr_decomposition_householder(A):
    m,n = A.shape
    R = A.copy()
    Q = np.identity(m)
    
    for j in range(0, n):
        # Apply Householder transformation.
        v = householder(R[j:, j, np.newaxis])
        H = np.identity(m)
        H[j:, j:] -= 2 * (v @ v.T) / (v.T @ v) 
        R = H @ R
        Q = H @ Q # Q.T
    return Q[:n].T, np.triu(R[:n])

# A = np.array([[ 12, -51,   4],
#       [  6, 167, -68],
#       [ -4,  24, -41],
#       [ -1,   1,   0],
#       [  2,   0,   3]])


# Q,R = qr_decomposition_householder(A)
# #print(np.dot(Q,R))
# print(Q)
# print(R)

def overdetermined_linear_system_solve(A, b):
    Q,R = qr_decomposition_householder(A)
    bs = Q @ b
    x = np.linalg.solve(R,bs)
    return x

def givens_rotation(a, b):
    # a - diagonal element
    # b - element to be zeroed
    r = np.hypot(a, b)
    c = a / r
    s = -b / r
    return c, s
    

def qr_decomposition_givens(A):
    num_rows, num_cols = np.shape(A)
    
    Q = np.identity(num_rows)
    R = np.copy(A)
    
    tril_rows, tril_cols = np.tril_indices(num_rows,-1,num_cols)
    
    for row, col in zip(tril_rows, tril_cols):
        if R[row, col] != 0.:
            c, s = givens_rotation(R[col, col], R[row, col])
        
            G = np.identity(num_rows)
            G[col,col] = c
            G[row,row] = c
            G[row,col] = s
            G[col, row] = -s
            
            R = np.dot(G, R)
            Q = np.dot(Q, G.T)
    return Q[:,:num_cols], np.triu(R[:num_cols])


# QG, RG = qr_decomposition_givens(A)
# print(QG)
# print(RG)


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
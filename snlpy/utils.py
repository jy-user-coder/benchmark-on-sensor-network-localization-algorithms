import numpy as np
import scipy.sparse as sm
import scipy.spatial as sp
import scipy.optimize as op
from timeit import default_timer
import scs


def my_format(a):
    if isinstance(a, float):
        return '{:.2e}'.format(float(a))
    else:
        return a


def my_print(*args):
    for arg in args:
        print(my_format(arg), end=' ')
    print('')


def mytable(out, myname=['SDR', 'SGD', 'Compound', 'GD', 'SGD($\\mathbb{R}^{n+d}$)',\
           'SGD($\\mathbb{R}^{n+d}$)+SGD', 'GALP 21', 'GALP-R 21', 'GALP 22', 'GALP-R 22',\
            'SDR+GALP 21', 'SDR+GALP-R 21', 'SDR+GALP 22', 'SDR+GALP-R 22']):
    xx, yy = out.shape
    for x in range(xx):
        print(myname[x], end=' & ')
        for y in range(yy):
            if y == yy-1:
                print(out[x, y], '\\\\\hline')
            else: 
                print(out[x, y], end='\\% & ')


def RMSD(x, y, DIM=2):
    x = x.reshape((-1,))
    y = y.reshape((-1,))
    return np.linalg.norm(x - y) * DIM / np.sqrt(len(x))


def Linf(x, y, DIM=2):
    x = x.reshape((-1,))
    y = y.reshape((-1,))
    return np.max(np.abs(x - y))


def generate_data(m, n, d, num_test, rng, add='data/'):
    anchors_list = rng.random((m, d, num_test)) - 0.5
    sensors_list = rng.random((n, d, num_test)) - 0.5
    np.save(add+'anchors_list.npy', anchors_list)
    np.save(add+'sensors_list.npy', sensors_list)
    print('anchors and sensors positions have already stored in the folder')


def scs_sub_solver(b, A, u, indices, eps=10**(-4)):
    # solve the subproblem argmin ||Ax + b||_{1, indices} + ux^Tx
    m, n = A.shape
    assert len(b) == m
    Im = sm.eye(m, format='csc')
    In = sm.eye(n, format='csc')
    Om = sm.csc_matrix((m, m))
    O_likeA = sm.csc_matrix((m, n))
    _A1 = sm.hstack([A, -Im, Im])
    _A2 = sm.hstack([O_likeA, -Im, Om])
    _A3 = sm.hstack([O_likeA, Om, -Im])
    _A = sm.vstack([_A1, _A2, _A3])
    _P = 2*u*sm.block_diag([In, Om, Om], format='csc')
    _b = np.hstack([-b, np.zeros(2*m)])
    _c = np.hstack([np.zeros(n), np.ones(2*m)])
    if len(indices) < m:
        for i in range(m):
            if not (i in indices):
                _c[n + i] = 0
    data = dict(P=_P, A = _A, b=_b, c=_c)
    cone = dict(z=m, l=2*m)
    solver = scs.SCS(
        data,
        cone,
        eps_abs=eps,
        eps_rel=eps,
    )
    sol_dict = solver.solve()
    sol = sol_dict['x']
    x = sol[0:n]
    y1 = sol[n:n+m]
    y2 = sol[n+m:]
    res_dict = dict(x=x, fun=np.sum(_c*np.hstack([x, y1, y2]))+u*np.sum(x**2))
    return res_dict


def ALP(Fk, wk, gradF, h, gradh, xk, \
            is_exact=False, sub_method='min', optimizer='BFGS', \
                sigma=0.005, theta=0.5, rho=2, alpha=1):
    gFk = gradF(xk)
    uk = min(sigma, theta*(wk**alpha))
    fxu = lambda d: h(Fk + gFk @ d) + uk*np.sum(d**2)
    gradfxu = lambda d: gradh(Fk + gFk @ d) @ gFk + 2*uk*d
    g0_norm = np.linalg.norm(gradh(Fk) @ gFk)
    if is_exact:
        gtol = 0.
        if sub_method == 'root':
            res_dict = op.root(gradfxu, np.zeros_like(xk))
        elif sub_method == 'min':
            res_dict = op.minimize(fxu, np.zeros_like(xk), jac=gradfxu, method=optimizer)
    else:
        eps_k = theta * (wk ** rho)
        while g0_norm <= np.sqrt(2 * uk * eps_k):
            eps_k = eps_k * theta
        gtol = np.sqrt(2 * uk *eps_k)
        if sub_method == 'root':
            res_dict = op.root(gradfxu, np.zeros_like(xk), tol=gtol)
        elif sub_method == 'min':
            options = {'gtol': gtol}
            res_dict = op.minimize(fxu, np.zeros_like(xk), jac=gradfxu, \
                                    method=optimizer, options=options)
    return res_dict, fxu, gtol, g0_norm


def GALP(F, gradF, h, gradh, xk, tol=10**(-6), epochs=100, \
          is_exact=False, sub_method='min', optimizer='BFGS', \
                gamma=0.9, lam=0.9, sigma=0.00001, theta=0.5, rho=2, alpha=1,\
                    eps=10**(-4), **kwargs):
    params = dict(tol=10**(-4), epochs=100, is_exact=False, \
                sub_method='min', optimizer='BFGS', \
                    gamma=0.9, lam=0.9, sigma=0.00001, theta=0.5, rho=2, alpha=1, eps=10**(-4))
    if len(kwargs) > 0:
        assert all((item in params.keys() or item == 'indices') for item in kwargs.keys())
        locals().update(kwargs)
        params.update(kwargs)
    if 'indices' in kwargs.keys() and len(kwargs['indices']) > 0 and sub_method == 'scs':
        indices = kwargs['indices']


    for k in range(epochs):
        Fk = F(xk)
        wk = h(Fk)
        if k>0 and (-wk+wk_old < tol and stp_size * np.linalg.norm(dk) < tol):
            break
        if sub_method == 'scs':
            gFk = gradF(xk)
            uk = min(sigma, theta*(wk**alpha))
            fxu = lambda d: h(Fk + gFk @ d) + uk*np.sum(d**2)
            g0_norm = np.linalg.norm(gradh(Fk) @ gFk)
            gtol = 0.
            res_dict = scs_sub_solver(Fk, gFk, uk, indices, eps)
        else:
            res_dict, fxu, gtol, g0_norm = ALP(Fk, wk, gradF, h, gradh, xk, \
                    is_exact, sub_method, optimizer, sigma, theta, rho, alpha)
        dk = res_dict['x']

        if sub_method == 'root' or sub_method == 'scs':
            fxudk = fxu(dk)
        elif sub_method == 'min':
            fxudk = res_dict['fun']

        term_rhs = fxudk - wk
        stp_size = 1.0
        while stp_size >= 2**(-32):
            if h(F(xk + stp_size * dk)) - wk <= term_rhs * lam * stp_size:
                break
            else:
                stp_size = gamma * stp_size

        xk = xk + stp_size * dk
        current_grad = np.linalg.norm(gradh(F(xk)) @ gradF(xk))
       
        my_print('epoch', k, 'loss', wk, 'norm_upd', \
            np.linalg.norm(dk), 'stepsize', stp_size, \
                'gtol', gtol, 'g0_norm', g0_norm, \
                    'end point grad', current_grad)

        wk_old = wk
    
    return xk, h(F(xk)), current_grad


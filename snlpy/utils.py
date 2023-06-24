import numpy as np
import scipy.sparse as sm
import scipy.spatial as sp
import scipy.optimize as op
from timeit import default_timer


def my_format(a):
    if isinstance(a, float):
        return '{:.2e}'.format(float(a))
    else:
        return a


def myprint(*args):
    for arg in args:
        print(my_format(arg), end=' ')
    print('')


def RMSD(x, y, DIM=2):
    x = x.reshape((-1,))
    y = y.reshape((-1,))
    return np.linalg.norm(x - y) * DIM / np.sqrt(len(x))


def generate_data(m, n, d, num_test, rng):
    anchors_list = rng.random((m, d, num_test)) - 0.5
    sensors_list = rng.random((n, d, num_test)) - 0.5
    np.save('data/anchors_list.npy', anchors_list)
    np.save('data/sensors_list.npy', sensors_list)
    print('anchors and sensors positions have already stored in the folder')


def ALP(Fk, wk, gradF, h, gradh, xk, method='findroot', \
            is_exact=False, sigma=0.005, theta=0.5, rho=2, alpha=1):
    gFk = gradF(xk)
    uk = min(sigma, theta*(wk**alpha))
    fxu = lambda d: h(Fk + gFk @ d) + uk*np.sum(d**2)
    gradfxu = lambda d: gradh(Fk + gFk @ d) @ gFk + 2*uk*d
    if is_exact:
        res_dict = op.minimize(fxu, np.zeros_like(xk), jac=gradfxu, method='BFGS')
    else:
        g0_norm = np.linalg.norm(gradh(Fk) @ gFk)
        eps_k = 0.1 * theta * (wk ** rho)
        while g0_norm <= np.sqrt(2 * uk * eps_k):
            eps_k = eps_k * theta
        gtol = np.sqrt(2 * uk *eps_k)
        myprint('gtol', gtol, '||gradf(0)||', g0_norm)
        if method == 'findroot':
            res_dict = op.root(gradfxu, np.zeros_like(xk), tol=gtol)
            return res_dict, fxu
        elif method == 'minimize':
            options = {'gtol': gtol}
            res_dict = op.minimize(fxu, np.zeros_like(xk), jac=gradfxu, method='BFGS', options=options)
            return res_dict, fxu



def GALP(F, gradF, h, gradh, xk, tol=10**(-4), method='findroot', epochs=100, is_exact=False, \
            gamma=0.9, lam=0.9, sigma=0.005, theta=0.5, rho=2, alpha=1, **kwargs):
    params = {method:'findroot', epochs:100, is_exact:False, \
               gamma:0.9, lam:0.9, sigma:0.005, theta:0.5, rho:2, alpha:1}
    if len(kwargs) > 0:
        assert all(item in kwargs.keys() for item in params.keys())
        locals().update(kwargs)
        params.update(kwargs)

    for k in range(epochs):
        Fk = F(xk)
        wk = h(Fk)
        if wk < tol:
            break
        res_dict, fxu = ALP(Fk, wk, gradF, h, gradh, xk, method, is_exact, sigma, theta, rho, alpha)
        # print(res_dict)
        dk = res_dict['x']
        if method == 'findroot':
            fxudk = fxu(dk)
        elif method == 'minimize':
            fxudk = res_dict['fun']
        term_rhs = fxudk - wk
        stp_size = 1.0
        while stp_size >= 2**(-32):
            if h(F(xk + stp_size * dk)) - wk <= term_rhs * lam * stp_size:
                break
            else:
                stp_size = gamma * stp_size
        xk = xk + stp_size * dk
        myprint('epoch', k, 'loss', wk, 'norm_upd',\
            np.linalg.norm(dk), 'stepsize', stp_size)
    return xk, h(F(xk)), k


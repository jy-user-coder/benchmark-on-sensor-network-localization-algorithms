import unittest
from snlpy import snl_case
from snlpy import utils
import numpy as np
import scipy.sparse as sm
import scipy.spatial as sp
import scipy.optimize as op
import os
from timeit import default_timer

# In the terminal, run 'python -m unittest' to check if all the algorithms work.
r = 0.5
m, n, d = 10, 50, 2
anchors = np.random.random((m, d)) - 0.5
sensors = np.random.random((n, d)) - 0.5
x0 = np.random.random((n*d,)) - 0.5
G = snl_case(anchors, sensors, r, is_relaxed=False)


class test_snl_case(unittest.TestCase):
    def test_init(self):
        sensors_test = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        anchors_test = np.array([[4, 0], [5, 0], [6, 0]])
        G1 = snl_case(anchors_test, sensors_test, 100)
        print('dda2s', G1.dda2s)
        print('dds2s', G1.dds2s)
        k0 = 0
        for i in range(G1.n):
            for j in range(i+1, G1.n):
                entry = G1.n * i + j - ((i + 2) * (i + 1)) // 2 
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
                dis = np.linalg.norm(sensors_test[i, :] - sensors_test[j, :])
                self.assertEqual(G1.target_relaxed[entry], dis)
                k0 = k0 + 1
        self.assertEqual(k0, G1.n*(G1.n-1)//2)
        k1 = k0
        for i in range(G1.n):
            for j in range(G1.m):
                entry = j + G1.m*i + k0
                dis = np.linalg.norm(sensors_test[i, :] - anchors_test[j, :])
                self.assertEqual(G1.target_relaxed[entry], dis)
                k1 = k1 + 1


    def test_grad(self):
        for is_relaxed in [True, False]:
            G.set_is_relaxed(is_relaxed)
            F = G.gen_F()
            gradF = G.gen_gradF()
            eps = 0.0001
            dx = (np.random.random(n*d) - 0.5) * eps
            self.assertEqual(gradF(x0).shape, (F(x0).shape[0], n*d))
            self.assertAlmostEqual(np.linalg.norm((F(x0+dx)-F(x0)-gradF(x0) @ dx)/eps), 0, places=2)

    
    def test_noisy_sdp(self):
        x_ans, z_ans = G.solve_by_sdp_with_noise()
        self.assertEqual(x_ans.shape, (d, n))
        self.assertEqual(z_ans.shape, (n+d, n+d))
        self.assertEqual(x_ans.tolist(), z_ans[0:d, d:].tolist())


    def test_noise_free_sdp(self):
        x_ans, z_ans = G.solve_by_sdp_noise_free()
        self.assertEqual(x_ans.shape, (d, n))
        self.assertEqual(z_ans.shape, (n+d, n+d))
        self.assertEqual(x_ans.tolist(), z_ans[0:d, d:].tolist())


    def test_two_sgds(self):
        rng = np.random.default_rng(12213)
        x_ans, loss, grad = G.sgd_21(sensors, rng, lr=0.1, dr=0.99, batch=100, penalty=0.1)
        self.assertEqual(x_ans.shape, (d, n))
        x_ans, loss, grad = G.sgd_21_high_dim(sensors, rng, lr=0.1, dr=0.99, batch=100, penalty=0.1)
        self.assertEqual(x_ans.shape, (d, n))


    def test_solver(self):
        F = G.gen_F()
        gradF = G.gen_gradF()
        h1, gradh1 = G.gen_h(p=1)
        print(len(G.indices), G.N)
        x_ans, wk, jac = utils.GALP(F, gradF, h1, gradh1, x0, optimizer='L-BFGS-B',\
                        indices=G.indices, epochs=50, sub_method='min', tol=10**(-6), sigma=0.00001, is_exact=True, eps=10**(-4))
        print(wk, utils.RMSD(x_ans, sensors), utils.Linf(x_ans, sensors), np.linalg.norm(jac))


    def test_solver_scs(self):
        F = G.gen_F()
        gradF = G.gen_gradF()
        h1, gradh1 = G.gen_h(p=1)
        gFk = gradF(x0)
        Fk = F(x0)
        wk = h1(Fk)
        sigma = 0.005
        theta = 0.5
        rho = 2
        alpha = 1
        uk = min(sigma, theta*(wk**alpha))
        fxu = lambda d: h1(Fk + gFk @ d) + uk*np.sum(d**2)
        gradfxu = lambda d: gradh1(Fk + gFk @ d) @ gFk + 2*uk*d
        res_dict = op.minimize(fxu, np.zeros_like(x0), jac=gradfxu, method='BFGS', tol=10**(-6))
        scsdic = utils.scs_sub_solver(Fk, gFk, uk, G.indices, 10**(-6))
        self.assertAlmostEqual(scsdic['fun'], res_dict['fun'], places=2)
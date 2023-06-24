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
r = 0.3
m, n, d = 20, 200, 2
anchors = np.random.random((m, d))
sensors = np.random.random((n, d))
x0 = np.random.random(n*d) - 0.5
G = snl_case(anchors, sensors, r, method='default')


class test_snl_case(unittest.TestCase):
    # def test_init(self):
    #     sensors_test = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    #     anchors_test = np.array([[4, 0], [5, 0], [6, 0]])
    #     G1 = snl_case(anchors_test, sensors_test, 100)
    #     print('dda2s', G1.dda2s)
    #     print('dds2s', G1.dds2s)
    #     k0 = 0
    #     for i in range(G1.n):
    #         for j in range(i+1, G1.n):
    #             entry = G1.n * i + j - ((i + 2) * (i + 1)) // 2 
    #             # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    #             dis = np.linalg.norm(sensors_test[i, :] - sensors_test[j, :])
    #             self.assertEqual(G1.target_relaxed[entry], dis)
    #             k0 = k0 + 1
    #     self.assertEqual(k0, G1.n*(G1.n-1)//2)
    #     k1 = k0
    #     for i in range(G1.n):
    #         for j in range(G1.m):
    #             entry = j + G1.m*i + k0
    #             dis = np.linalg.norm(sensors_test[i, :] - anchors_test[j, :])
    #             self.assertEqual(G1.target_relaxed[entry], dis)
    #             k1 = k1 + 1


    # def test_grad(self):
    #     for is_relaxed in [True, False]:
    #         F = G.gen_F(is_relaxed)
    #         gradF = G.gen_gradF(is_relaxed)
    #         eps = 0.0001
    #         dx = (np.random.random(n*d) - 0.5) * eps
    #         self.assertEqual(gradF(x0).shape, (F(x0).shape[0], n*d))
    #         self.assertAlmostEqual(np.linalg.norm((F(x0+dx)-F(x0)-gradF(x0) @ dx)/eps), 0, places=2)


    # def test_galp_1loss(self):
    #     F = G.gen_F(True)
    #     gradF = G.gen_gradF(True)
    #     h1 = G.gen_h1(True)
    #     gradh1 = G.gen_gradh1(True)
    #     x_ans, wk, jac = utils.GALP(F, gradF, h1, gradh1, x0, epochs=50, sigma=0.0001)
    #     print(wk, utils.RMSD(x_ans, sensors), utils.Linf(x_ans, sensors), np.linalg.norm(jac))


    # def test_galp_2loss(self):
    #     G = snl_case(anchors, sensors, r, method='default')
    #     F = G.gen_F(True)
    #     gradF = G.gen_gradF(True)
    #     h2 = G.gen_h2(True)
    #     gradh2 = G.gen_gradh2(True)
    #     x_ans, wk, jac = utils.GALP(F, gradF, h2, gradh2, x0, is_exact=True, sub_method='min', optimizer='CG', epochs=100)
    #     print(wk, utils.RMSD(x_ans, sensors), utils.Linf(x_ans, sensors), np.linalg.norm(jac))
    #     self.assertEqual(x_ans.shape, (d*n,))
    #     self.assertEqual(h2(F(x_ans)), wk)

    
    # def test_noisy_sdp(self):
    #     x_ans, z_ans = G.solve_by_sdp_with_noise()
    #     self.assertEqual(x_ans.shape, (d, n))
    #     self.assertEqual(z_ans.shape, (n+d, n+d))
    #     self.assertEqual(x_ans.tolist(), z_ans[0:d, d:].tolist())


    # def test_noise_free_sdp(self):
    #     x_ans, z_ans = G.solve_by_sdp_noise_free()
    #     self.assertEqual(x_ans.shape, (d, n))
    #     self.assertEqual(z_ans.shape, (n+d, n+d))
    #     self.assertEqual(x_ans.tolist(), z_ans[0:d, d:].tolist())


    # def test_two_sgds(self):
    #     rng = np.random.default_rng(12213)
    #     x_ans, loss = G.sgd_21(sensors, rng, lr=0.1, dr=0.99, batch=100, penalty=0.1)
    #     self.assertEqual(x_ans.shape, (d, n))
    #     x_ans, loss = G.sgd_21_high_dim(sensors, rng, lr=0.1, dr=0.99, batch=100, penalty=0.1)
    #     self.assertEqual(x_ans.shape, (d, n))


    def test_solver(self):
        F = G.gen_F(True)
        gradF = G.gen_gradF(True)
        h1 = G.gen_h1(True)
        gradh1 = G.gen_gradh1(True)
        x_ans, wk, jac = utils.GALP(F, gradF, h1, gradh1, x0, indices=[0])
        print(wk, utils.RMSD(x_ans, sensors), utils.Linf(x_ans, sensors), np.linalg.norm(jac))
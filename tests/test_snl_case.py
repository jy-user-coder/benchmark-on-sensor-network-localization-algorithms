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
    def test_galp_1loss(self):
        F = G.gen_F(False)
        gradF = G.gen_gradF(False)
        h1 = G.gen_h1(False)
        gradh1 = G.gen_gradh1(False)
        x_ans, wk, k = utils.GALP(F, gradF, h1, gradh1, x0, method='minimize', epochs=5, sigma=0.0001)
        self.assertEqual(x_ans.shape, (d*n,))
        self.assertEqual(h1(F(x_ans)), wk)
    

    def test_galp_2loss(self):
        G = snl_case(anchors, sensors, r, method='default')
        F = G.gen_F(False)
        gradF = G.gen_gradF(False)
        h2 = G.gen_h2(False)
        gradh2 = G.gen_gradh2(False)
        x_ans, wk, k = utils.GALP(F, gradF, h2, gradh2, x0, method='minimize', epochs=100, sigma=0.0001)
        print(utils.RMSD(x_ans, sensors))
        self.assertEqual(x_ans.shape, (d*n,))
        self.assertEqual(h2(F(x_ans)), wk)

    
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



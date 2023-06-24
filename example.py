import numpy as np
import scipy.sparse as sm
import scipy.spatial as sp
import scipy.optimize as op
from snlpy import snl_case
from snlpy import utils
import os

rng = np.random.default_rng(12345)
address_data = 'data'

if not (os.path.exists(address_data + '/anchors_list.npy') and \
            os.path.exists(address_data + '/sensors_list.npy')):
    m, n, d = 20, 200, 2
    num_test = 100
    utils.generate_data(m, n, d, num_test, rng)


anchors_list = np.load('data/anchors_list.npy')
sensors_list = np.load('data/sensors_list.npy')
assert anchors_list.shape[1] == sensors_list.shape[1]
assert anchors_list.shape[2] == sensors_list.shape[2]
m = anchors_list.shape[0]
n = sensors_list.shape[0]
d = anchors_list.shape[1]
num_test = anchors_list.shape[2]

r_list = [0.3]
res = np.zeros((num_test, len(r_list), 3))


for i in range(num_test):
    anchors = anchors_list[:, :, i]
    sensors = sensors_list[:, :, i]
    x0 = rng.random(n*d) - 0.5
    for j, r in enumerate(r_list):
        G = snl_case(anchors, sensors, r, method='default')
        F = G.gen_F(False)
        gradF = G.gen_gradF(False)
        h = G.gen_h1(False)
        gradh = G.gen_gradh1(False)
        ans, wk, k = utils.GALP(F, gradF, h, gradh, x0, 100, sigma=0.0001)
        print('test no.', i, 'RMSD', utils.RMSD(ans, sensors))


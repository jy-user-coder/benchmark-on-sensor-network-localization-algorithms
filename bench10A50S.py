import unittest
from snlpy import snl_case
from snlpy import utils
import numpy as np
import scipy.sparse as sm
import scipy.spatial as sp
import scipy.optimize as op
import os
from timeit import default_timer
from tqdm import tqdm


m, n, d = 10, 50, 2
address_data = 'benchmark_data/' + str(m) + 'A' + str(n) + 'S/'
r_list = [0.25, 0.3, 0.35, 0.4]
num_test = 20
if os.path.exists(address_data + 'sdp_sols' + '.npy'):
    sdp_sols = np.load(address_data + 'sdp_sols' + '.npy')
    is_sdp_sols = True
else:
    sdp_sols = np.zeros((d, n, len(r_list), 5*num_test))
    is_sdp_sols = False
final_res = np.zeros((4, 14, len(r_list), num_test*5))


for random_case in tqdm(range(5)):
    rng = np.random.default_rng(random_case)
    if not (os.path.exists(address_data + str(random_case)+ 'anchors_list' + '.npy') and \
                os.path.exists(address_data + str(random_case) + 'sensors_list'  + '.npy')):
        utils.generate_data(m, n, d, num_test, rng, address_data + str(random_case))

    anchors_list = np.load(address_data + str(random_case) + 'anchors_list' + '.npy')
    sensors_list = np.load(address_data + str(random_case) + 'sensors_list' + '.npy')

    for i in tqdm(range(num_test)):
        current_case = random_case*num_test + i
        anchors = anchors_list[:, :, i]
        sensors = sensors_list[:, :, i]
        x0 = rng.random(n*d) - 0.5
        s0 = x0.reshape((2, -1))
        for j, r in enumerate(r_list):
            G = snl_case(anchors, sensors, r, method='default')
            old_dda2s = (G.dda2s).reshape((G.n, G.m))
            old_dds2s = np.triu(sp.distance.squareform(G.dds2s), 1)


            if not is_sdp_sols:
                x_ans_sdp, z_ans_sdp = G.solve_by_sdp_with_noise()
                sdp_sols[:, :, j, current_case] = x_ans_sdp
                rmsd_sdp = utils.RMSD(x_ans_sdp, sensors.T)
                grad_sdp_vec, loss_sdp = G._func_grad_21(G.m, G.n, (G.anchors).T, x_ans_sdp, sm.find(old_dda2s), sm.find(old_dds2s))
                grad_sdp = np.linalg.norm(grad_sdp_vec)
                # np.save(address_data + 'sdp_sols', sdp_sols)


            batch1 = len(G.indices) // 50
            x_ans_sgd, loss_sgd, grad_sgd = G.sgd_21(s0, rng, tol=10**(-4), epochs=1000, lr=0.1, batch=batch1, dr=0.995)
            rmsd_sgd = utils.RMSD(x_ans_sgd, sensors.T)

            x_ans_gd, loss_gd, grad_gd = G.sgd_21(s0, rng, tol=10**(-4), epochs=1000, lr=0.01, batch=1, dr=0.995)
            rmsd_gd = utils.RMSD(x_ans_gd, sensors.T)

            x_ans_sgd_hd, loss_sgd_hd1, grad_sgd_hd1 = G.sgd_21_high_dim(s0, rng, tol=10**(-4), epochs=500, lr=0.1, batch=batch1, dr=0.99, penalty=0.0)
            rmsd_sgd_hd = utils.RMSD(x_ans_sgd_hd, sensors.T)
            grad_sgd_hd_vec, loss_sgd_hd = G._func_grad_21(G.m, G.n, (G.anchors).T, x_ans_sgd_hd, sm.find(old_dda2s), sm.find(old_dds2s))
            grad_sgd_hd = np.linalg.norm(grad_sgd_hd_vec)     

            x_ans_sgdcom, loss_sgdcom, grad_sgdcom = G.sgd_21(x_ans_sgd_hd, rng, tol=10**(-4), epochs=500, lr=0.001, batch=batch1, dr=0.99)
            rmsd_sgdcom = utils.RMSD(x_ans_sgdcom, sensors.T)

            final_res[:, 0, j, current_case] = np.array([rmsd_sdp, loss_sdp, grad_sdp, 0])
            final_res[:, 1, j, current_case] = np.array([rmsd_sgd, loss_sgd, grad_sgd, 0])
            final_res[:, 3, j, current_case] = np.array([rmsd_gd, loss_gd, grad_gd, 0])
            final_res[:, 4, j, current_case] = np.array([rmsd_sgd_hd, loss_sgd_hd, grad_sgd_hd, 0])
            final_res[:, 5, j, current_case] = np.array([rmsd_sgdcom, loss_sgdcom, grad_sgdcom, 0])

            for p in [1, 2]:
                for is_relaxed in [True, False]:
                    G = G.set_is_relaxed(is_relaxed)
                    F = G.gen_F()
                    gradF = G.gen_gradF()
                    h, gradh = G.gen_h(p)
                    sigma = 0.00001
                    if  p == 2:
                        sub_method, optimizer = 'min', 'CG'
                    elif p == 1:
                        if is_relaxed:
                            sub_method, optimizer = 'min', 'BFGS'
                        else:
                            sub_method, optimizer = 'min', 'BFGS'
                    sol, loss, grad = utils.GALP(F, gradF, h, gradh, x0, is_exact=True, \
                                            indices=G.indices, sub_method=sub_method, optimizer=optimizer)
                    rmsd = utils.RMSD(sol, sensors)
                    linf = utils.Linf(sol, sensors)
                    final_res[:, 6 + 2*(p-1) + is_relaxed, j, current_case] = np.array([rmsd, linf, loss, np.linalg.norm(grad)])
                    print(final_res[:, 6 + 2*(p-1) + is_relaxed, j, current_case])


            if is_sdp_sols:
                x1 = sdp_sols[:, :, j, current_case]
            else:
                x1 = x_ans_sdp
            x_ans_com, loss_com, grad_com = G.sgd_21(x1, rng, tol=10**(-4), epochs=200, lr=0.001, batch=batch1, dr=0.98)
            rmsd_com = utils.RMSD(x_ans_com, sensors.T)
            final_res[:, 2, j, current_case] = np.array([rmsd_com, loss_com, grad_com, 0])
            x1 = (x1.T).reshape((-1,))
            for p in [1, 2]:
                for is_relaxed in [True, False]:
                    G = G.set_is_relaxed(is_relaxed)
                    F = G.gen_F()
                    gradF = G.gen_gradF()
                    h, gradh = G.gen_h(p)
                    sigma = 0.00001
                    if  p == 2:
                        sub_method, optimizer = 'min', 'CG'
                    elif p == 1:
                        if is_relaxed:
                            sub_method, optimizer = 'min', 'BFGS'
                        else:
                            sub_method, optimizer = 'min', 'BFGS'
                    sol, loss, grad = utils.GALP(F, gradF, h, gradh, x1, is_exact=True, \
                                            indices=G.indices, sub_method=sub_method, optimizer=optimizer)
                    rmsd = utils.RMSD(sol, sensors)
                    linf = utils.Linf(sol, sensors)
                    final_res[:, 10 + 2*(p-1) + is_relaxed, j, current_case] = np.array([rmsd, linf, loss, np.linalg.norm(grad)])
                    print(final_res[:, 10 + 2*(p-1) + is_relaxed, j, current_case])
            np.save(address_data + 'final_res_new', final_res)

import numpy as np
import scipy.sparse as sm
import scipy.spatial as sp
import scipy.optimize as op
import scs
from timeit import default_timer


class snl_case:
    def __init__(self, anchors, sensors_true, r, method='default'):
        assert anchors.shape[1] == sensors_true.shape[1]
        self.anchors = anchors
        self.m = anchors.shape[0]
        self.n = sensors_true.shape[0]
        self.DIM = anchors.shape[1]
        dda2s_full = sp.distance.cdist(anchors, sensors_true)
        dds2s_full = sp.distance.pdist(sensors_true)
        if method == 'default':
            adja2s = (dda2s_full <= r)
            adjs2s = (dds2s_full <= r)
            tmp_dda2s = (dda2s_full * adja2s).T
            tmp_dds2s = dds2s_full * adjs2s
            # self.is_sparse = False
            self.dda2s = (tmp_dda2s).reshape((-1,))
            self.dds2s = tmp_dds2s
            self.target_radius = np.hstack([(self.dds2s==0)*r + self.dds2s, (self.dda2s==0)*r + self.dda2s])
            self.target_relaxed = np.hstack([self.dds2s, self.dda2s])
            self.indices = np.nonzero(self.target_relaxed)


    def gen_F(self, is_relaxed):
        def F(x):
            re_x = x.reshape((-1, 2))
            current_dda2s = (sp.distance.cdist(self.anchors, re_x).T).reshape((-1,))
            current_dds2s = sp.distance.pdist(re_x)
            if is_relaxed:
                F_val = 0.5*(np.hstack([current_dds2s, current_dda2s])**2 - self.target_relaxed**2)[self.indices]
            else:
                F_val = 0.5*(np.hstack([current_dds2s, current_dda2s])**2 - self.target_radius**2)
            return F_val
        return F


    def gen_gradF(self, is_relaxed):
        m = self.m
        n = self.n
        if is_relaxed:
            def gradF(x):
                res = np.zeros((len(self.indices[0]), 2*n))
                re_x = x.reshape((-1, 2))
                k = 0
                for i in range(n):
                    for j in range(i+1, n):
                        entry = n * i + j - ((i + 2) * (i + 1)) // 2 
                        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
                        if self.dds2s[entry] > 0:
                            res[k, 2*i] = re_x[i, 0] - re_x[j, 0]
                            res[k, 2*i + 1] = re_x[i, 1] - re_x[j, 1]
                            res[k, 2*j] = -re_x[i, 0] + re_x[j, 0]
                            res[k, 2*j + 1] = -re_x[i, 1] + re_x[j, 1]
                            k = k + 1
                k1 = k
                for i in range(n):
                    for j in range(m):
                        entry = j + self.m*i
                        if self.dda2s[entry] > 0:
                            res[k, 2*i:2*i+2] = re_x[i, :] - self.anchors[j, :]
                            k = k + 1
                return res
        else:
            def gradF(x):
                re_x = x.reshape((-1, 2))
                res = np.zeros((n*(n-1)//2 + m*n, 2*n))
                for i in range(n):
                    res[(n*(n-1)//2 + i*m):(n*(n-1)//2 + (i+1)*m), 2*i] = re_x[i, 0] - self.anchors[:, 0]
                    res[(n*(n-1)//2 + i*m):(n*(n-1)//2 + (i+1)*m), 2*i+1] = re_x[i, 1] - self.anchors[:, 1]
                k = 0
                for i in range(n):
                    for j in range(i+1, n):
                        res[k, 2*i] = re_x[i, 0] - re_x[j, 0]
                        res[k, 2*i + 1] = re_x[i, 1] - re_x[j, 1]
                        res[k, 2*j] = -re_x[i, 0] + re_x[j, 0]
                        res[k, 2*j + 1] = -re_x[i, 1] + re_x[j, 1]
                        k = k + 1
                return res
        return gradF


    @staticmethod
    def _tmp_h2(x, dis):
        return ((dis==0) * (x<0) * x**2 + (dis!=0) * x**2)/2


    def gen_h2(self, is_relaxed):
        if is_relaxed:
            return lambda x: np.sum(x**2)/2
        else:
            def h(x):
                tmp_hlist = self._tmp_h2(x, self.target_relaxed)
                return np.sum(tmp_hlist)
            return h


    @staticmethod
    def _tmp_gradh2(x, dis):
        return ((dis==0) * (x<0) * x) + ((dis!=0) * x)


    def gen_gradh2(self, is_relaxed):
        if is_relaxed:
            return lambda x: x
        else: 
            def gradh(x):
                return (self._tmp_gradh2(x, self.target_relaxed))
            return gradh


    @staticmethod
    def _tmp_h1(x, dis):
        return ((dis==0) * (x<0) * (-x) + (dis!=0) * np.abs(x))


    def gen_h1(self, is_relaxed):
        if is_relaxed:
            return lambda x: np.sum(np.abs(x))
        else: 
            def h(x):
                tmp_hlist = self._tmp_h1(x, self.target_relaxed)
                return np.sum(tmp_hlist)
            return h


    @staticmethod
    def _tmp_gradh1(x, dis):
        return ((dis==0) * (x<0) * (-1) + (dis!=0) * np.sign(x))


    def gen_gradh1(self, is_relaxed):
        if is_relaxed:
            return lambda x: np.sign(x)
        else: 
            def gradh(x):
                return (self._tmp_gradh1(x, self.target_relaxed))
            return gradh


    ########################################################################
    # below: SDR methods for solving SNL 
    ########################################################################


    @staticmethod
    def _converting(i, j, N):


        # Let S be a (N, N) symmetrix matrix.
        # If S_{ij} = S_{ji} = 1, and all other elements of S are zero, 
        #   then vec(S) will have the 
        #       converting(i, j, N)
        #           -th element as its unique nonzero element. 
        # You should be VERY CAREFUL about the indices if using this function.


        if j < i: 
            i, j = j, i

        # return N + N-1 + ... + (N-i+1) + j-i

        return (2*N-i+1)*i//2 + j - i


    @staticmethod
    def _vec(S):


        # This is a general function dealing with semi-definite cones in scs.
        # See https://www.cvxgrp.org/scs/examples/python/mat_completion.html#py-mat-completion
        # Input: (n, n) symmetric matrix
        # Output: a padded (n(n+1)/2,) array s
        # e.g. [[1, 2, 3],
        #       [2, 5, 6],
        #       [3, 6, 9]]
        # is transformed into an array
        # [1, 2 \sqrt{2}, 3 \sqrt{2}, 5, 6 \sqrt{2}, 9]


        n = S.shape[0]
        # S = np.copy(S)
        S *= np.sqrt(2)
        S[range(n), range(n)] /= np.sqrt(2)
        return S[np.triu_indices(n)]


    @staticmethod
    def _mat(s):
        # This is a general function dealing with semi-definite cones in scs.
        # See https://www.cvxgrp.org/scs/examples/python/mat_completion.html#py-mat-completion
        # Input: (n(n+1)/2,) array s
        # Output: (n, n) symmetric matrix correspond to s
        # e.g. [1, 2 \sqrt{2}, 3 \sqrt{2}, 5, 6 \sqrt{2}, 9]
        # is transformed into a symmetric matrix
        #      [[1, 2, 3],
        #       [2, 5, 6],
        #       [3, 6, 9]]
        # We should note that 
        #   vec(mat(s)) = s, mat(vec(S)) = S.
        n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
        S = np.zeros((n, n))
        S[np.triu_indices(n)] = s / np.sqrt(2)
        S = S + S.T
        S[range(n), range(n)] /= np.sqrt(2)
        return S


    def _gen_input_noise_free(self):


        # This function generates the api. BE CAREFUL.
        # Input:
        # anchors: DIM*m matrix listing all anchors.
        # dd: n*(m+n) sparse matrix between known distance of anchors and sensors.
        # Output:
        #   weight, bias, z, s.
        # Dimension of the zero cone(number of equalities):
        #   Z_{1:DIM, 1:DIM} = I_DIM, 
        #       this generates DIM(DIM+1)/2 equalities.
        #   [0; e_{ij}]^T Z [0; e_{ij}] = d_{ij}^2,
        #   [a_k; e_j]^T Z [a_k, e_j] = h_{jk}^2, 
        #       this two generate len(dd1) equalities.
        # Dimension of the positive semi-definite cone:
        #   (n+DIM+1)(n+DIM)/2. 
        # Let Z \in \R^{(n+DIM)(n+DIM+1)/2} be the positive semi-definite matrix, then 
        #   -weight vec(Z) + bias \in K if and only if Z satisfies all constraints,   
        # where K is a convex cone. The first DIM(DIM+1)/2+len(dd1) entries are the zero cone, 
        #   the last (n+DIM+1)(n+DIM)/2 are the positive semi-definite cone.

        sp_dda2s = sm.csr_matrix((self.dda2s).reshape((self.n, self.m)))
        sp_dds2s = sm.csr_matrix(np.triu(sp.distance.squareform(self.dds2s), 1))
        dd = sm.hstack([sp_dda2s, sp_dds2s])
        anchors = (self.anchors).T
        # DIM: dimension of the question
        DIM = np.shape(anchors)[0]

        # n: number of sensors, m: number of anchors
        n, m = np.shape(dd)[0], np.shape(dd)[1]-np.shape(dd)[0]

        # split distance matrix
        a2s = dd[:, 0:m]
        s2s = dd[:, m:]
        a2s1, a2s2, a2s3 = sm.find(a2s)
        s2s1, s2s2, s2s3 = sm.find(s2s)
        

        # calculate the dimension of the zero cone and the positive semi-definite cone
        z = (DIM+1)*DIM//2 + len(a2s1) + len(s2s1)
        s = n+DIM

        weight = sm.lil_matrix((z + s*(s+1)//2, s*(s+1)//2))
        bias = np.zeros(np.shape(weight)[0])


        # Implement the constraints: Z_{1:d, 1:d} = I_d
        row_index, col_index = np.triu_indices(DIM)
        for i in range(len(row_index)):
            weight[i, i + row_index[i]*n] = 1.
            bias[i] = float(row_index[i] == col_index[i])


        # Implement the constraints: Z \succeq 0
        for i in range((n+DIM+1)*(n+DIM)//2):
            weight[z+i, i] = -1


        # Implement the constraints: [a_k; e_j]^T Z [a_k, e_j] = h_{jk}^2
        # An equivalent but less efficient approach
        # for i in range(len(a2s1)):
        #     ej = np.zeros((n,))
        #     ej[a2s1[i]] = -1
        #     tmp = np.hstack([anchors[:, a2s2[i]], ej]).reshape((n+DIM, 1))
        #     weight[(DIM+1)*DIM//2+i, :] = vec(tmp @ tmp.T)
        #     bias[(DIM+1)*DIM//2+i] = a2s3[i]**2
        
            
        for i in range(len(a2s1)):
            k2 = a2s1[i] + DIM
            weight[(DIM+1)*DIM//2+i, self._converting(k2, k2, DIM+n)] = 1.
            ak1 = anchors[:, a2s2[i]].reshape((DIM, 1))

            for k in range(len(row_index)):
                kk1 = row_index[k]
                kk2 = col_index[k]
                if kk1 == kk2:
                    weight[(DIM+1)*DIM//2 + i, self._converting(kk1, kk2, DIM+n)] = ak1[kk1]**2 
                else:
                    weight[(DIM+1)*DIM//2 + i, self._converting(kk1, kk2, DIM+n)] = np.sqrt(2) * ak1[kk1] * ak1[kk2]

            for k in range(DIM):
                weight[(DIM+1)*DIM//2 + i, self._converting(k, k2, DIM+n)] = -np.sqrt(2) * ak1[k]
                
            bias[(DIM+1)*DIM//2 + i] = a2s3[i]**2


        # Implement the constraints: [0; e_{ij}]^T Z [0; e_{ij}] = d_{ij}^2
        # An equivalent but less efficient approach
        # for i in range(len(s2s1)):
        #     tmp = np.zeros((n+DIM, 1))
        #     tmp[DIM + s2s1[i], 0] = -1
        #     tmp[DIM + s2s2[i], 0] = 1
        #     weight[(DIM+1)*DIM//2 + len(a2s1) + i, :] = vec(tmp @ tmp.T)
        #     bias[(DIM+1)*DIM//2 + len(a2s1) + i] = s2s3[i]**2


        for i in range(len(s2s1)):
            k1 = DIM + s2s1[i]
            k2 = DIM + s2s2[i]
            weight[(DIM+1)*DIM//2 + len(a2s1) + i, self._converting(k1, k1, DIM+n)] = 1.
            weight[(DIM+1)*DIM//2 + len(a2s1) + i, self._converting(k2, k2, DIM+n)] = 1.
            weight[(DIM+1)*DIM//2 + len(a2s1) + i, self._converting(k1, k2, DIM+n)] = -np.sqrt(2)
            bias[(DIM+1)*DIM//2 + len(a2s1) + i] = s2s3[i]**2


        return sm.csc_matrix(weight), bias, z, s


    def solve_by_sdp_noise_free(self):

        sp_dda2s = sm.csr_matrix((self.dda2s).reshape((self.n, self.m)))
        sp_dds2s = sm.csr_matrix(np.triu(sp.distance.squareform(self.dds2s), 1))
        dd = sm.hstack([sp_dda2s, sp_dds2s])
        anchors = (self.anchors).T
        DIM = np.shape(anchors)[0]
        n, m = np.shape(dd)[0], np.shape(dd)[1]-np.shape(dd)[0]
        w, b, z, s = self._gen_input_noise_free()
        P = sm.csc_matrix(((s+1)*s//2, (s+1)*s//2))
        c = self._vec(np.zeros((s, s)))
        data = dict(P=P, A=w, b=b, c=c)
        cone = dict(z=z, s=s)
        solver = scs.SCS(data, cone, eps_abs=1e-6, eps_rel=1e-6, max_iters=1000)
        mydict = solver.solve()
        ans_z = self._mat(mydict['x'])

        return ans_z[0:DIM, DIM:], ans_z


    def _gen_input_with_noise(self, regularization=False, lam=1.):
        
        # This function generates the api. BE CAREFUL.
        # anchors: (DIM, m) array listing the coordinate of anchors
        # dd: (n, m+n) array, the same as before
        # DIM=dimension of the question
        
        sp_dda2s = sm.csr_matrix((self.dda2s).reshape((self.n, self.m)))
        sp_dds2s = sm.csr_matrix(np.triu(sp.distance.squareform(self.dds2s), 1))
        dd = sm.hstack([sp_dda2s, sp_dds2s])
        anchors = (self.anchors).T
        DIM = np.shape(anchors)[0]

        # n: number of sensors, m: number of anchors
        n, m = np.shape(dd)[0], np.shape(dd)[1]-np.shape(dd)[0]

        # split the distance matrix
        a2s = dd[:, 0:m]
        s2s = dd[:, m:]
        a2s1, a2s2, a2s3 = sm.find(a2s)
        s2s1, s2s2, s2s3 = sm.find(s2s)
        
        # calculate the dimension of the zero cone and the positive semi-definite cone
        z = (DIM+1)*DIM//2 + len(a2s1) + len(s2s1)
        s = n+DIM
        l = 2*(len(a2s1) + len(s2s1)) 


        # Implement the constraints y_plus, y_minus, Z \succeq 0
        weight = sm.lil_matrix(sm.vstack([sm.lil_matrix((z, l + s*(s+1)//2)),\
            -sm.eye(l + s*(s+1)//2)]))
            
        bias = np.zeros(np.shape(weight)[0])

        if regularization:
            e_hat = np.ones(n)/np.sqrt(n+m)
            a_hat = np.sum(anchors, axis=1)/np.sqrt(n+m)
            a = np.hstack([e_hat.reshape((1, n)), a_hat.reshape((1, DIM))])
            tmp = sm.eye(n+DIM) - a.T @ a
            c = np.append(np.ones((l,)), -lam * self._vec(tmp))
            print('Regularization implemented with \lambda=', lam)
        else:
            c = np.hstack([np.ones(l), np.zeros((s+1)*s//2)])

        c = np.hstack([np.ones(l), np.zeros((s+1)*s//2)])
        print('processing 1/5')


        # The indices of the upper-triangle of a matrix.
        row_index, col_index = np.triu_indices(DIM)


        # Implement the constraints: Z_{1:d, 1:d} = I_d
        for i in range(len(row_index)):
            weight[i, l + i + row_index[i]*n] = -1.
            bias[i] = -float(row_index[i] == col_index[i])


        print('processing 2/5')


        # Implement the constraints: [a_k; e_j]^T Z [a_k, e_j] = h_{jk}^2
        # An equivalent but less efficient approach
        # for i in range(len(a2s1)):
            # ej = np.zeros((n,))
            # ej[a2s1[i]] = -1
            # tmp = np.hstack([anchors[:, a2s2[i]], ej]).reshape((n+d, 1))
            # weight[(d+1)*d//2+i, l:] = sm.lil_matrix(vec(tmp @ tmp.T))
            # bias[(d+1)*d//2+i] = a2s3[i]**2


        for i in range(len(a2s1)):

            k2 = a2s1[i] + DIM
            weight[(DIM+1)*DIM//2+i, l + self._converting(k2, k2, DIM+n)] = 1.
            ak1 = anchors[:, a2s2[i]].reshape((DIM, 1))

            for k in range(len(row_index)):
                kk1 = row_index[k]
                kk2 = col_index[k]
                if kk1 == kk2:
                    weight[(DIM+1)*DIM//2 + i, l + self._converting(kk1, kk2, DIM+n)] = ak1[kk1]**2 
                else:
                    weight[(DIM+1)*DIM//2 + i, l + self._converting(kk1, kk2, DIM+n)] = np.sqrt(2) * ak1[kk1] * ak1[kk2]

            for k in range(DIM):
                weight[(DIM+1)*DIM//2 + i, l + self._converting(k, k2, DIM+n)] = -np.sqrt(2) * ak1[k]
                
            bias[(DIM+1)*DIM//2 + i] = a2s3[i]**2
        

        print('processing 3/5')
        

        # Implement the constraints: [0; e_{ij}]^T Z [0; e_{ij}] = d_{ij}^2
        # An equivalent but less efficient approach
        # for i in range(len(s2s1)):
        #     tmp = np.zeros((n+DIM, 1))
        #     tmp[DIM + s2s1[i], 0] = -1
        #     tmp[DIM + s2s2[i], 0] = 1
        #     weight[(DIM+1)*DIM//2 + len(a2s1) + i, l:] = vec(tmp @ tmp.T)
        #     bias[(DIM+1)*DIM//2 + len(a2s1) + i] = s2s3[i]**2


        for i in range(len(s2s1)):
            k1 = DIM + s2s1[i]
            k2 = DIM + s2s2[i]
            weight[(DIM+1)*DIM//2 + len(a2s1) + i, l + self._converting(k1, k1, DIM+n)] = 1.
            weight[(DIM+1)*DIM//2 + len(a2s1) + i, l + self._converting(k2, k2, DIM+n)] = 1.
            weight[(DIM+1)*DIM//2 + len(a2s1) + i, l + self._converting(k1, k2, DIM+n)] = -np.sqrt(2)
            bias[(DIM+1)*DIM//2 + len(a2s1) + i] = s2s3[i]**2
            

        print('processing 4/5')


        for i in range(l//2):
            weight[(DIM+1)*DIM//2 + i, i] = -1.
            weight[(DIM+1)*DIM//2 + i, l//2 + i] = 1.
        

        # Be careful about the memory if using the code below.
        # weight[(DIM+1)*DIM//2:((DIM+1)*DIM//2+l//2), 0:l//2] = -sm.eye(l//2)
        # weight[(DIM+1)*DIM//2:((DIM+1)*DIM//2+l//2), l//2:l] = sm.eye(l//2)


        print('processing 5/5')
        

        return sm.csc_matrix(weight), bias, c, z, s, l


    def solve_by_sdp_with_noise(self, regularization=False, **kwargs):

        sp_dda2s = sm.csr_matrix((self.dda2s).reshape((self.n, self.m)))
        sp_dds2s = sm.csr_matrix(np.triu(sp.distance.squareform(self.dds2s), 1))
        dd = sm.hstack([sp_dda2s, sp_dds2s])
        anchors = (self.anchors).T
        DIM = np.shape(anchors)[0]
        n, m = np.shape(dd)[0], np.shape(dd)[1]-np.shape(dd)[0]

        t1 = default_timer()
        if 'lam' in kwargs.keys():
            lam = kwargs['lam']
        else: 
            lam = 1.

        A, b, c, z, s, l = self._gen_input_with_noise(regularization, lam)

        P = sm.csc_matrix(((s+1)*s//2 + l, (s+1)*s//2 + l))
        data = dict(P=P, A=A, b=b, c=c)
        cone = dict(z=z, l=l, s=s)
        t2 = default_timer()

        sol = scs.SCS(data, cone, eps_abs=1e-6, eps_rel=1e-6, max_iters=1000).solve()['x']
        mysol = self._mat(sol[l:])
        
        return mysol[0:DIM, DIM:], mysol
    

    ########################################################################
    # below: SGD methods for solving SNL 
    ########################################################################


    @staticmethod
    def _func_21(m, n, anchors, sensors, dda2s, dds2s, **kwargs):

        rowa2s, cola2s, disa2s = dda2s
        rows2s, cols2s, diss2s = dds2s

        if 'S1' in kwargs.keys():
            S1 = kwargs['S1']
        else:
            linka2s = len(rowa2s)
            S1 = sm.coo_matrix((np.ones(linka2s), (m + rowa2s, range(linka2s))), shape=(n+m, linka2s)) - \
                    sm.coo_matrix((np.ones(linka2s), (cola2s, range(linka2s))), shape=(n+m, linka2s))

        if 'S2' in kwargs.keys():
            S2 = kwargs['S2']
        else:
            links2s = len(rows2s)
            S2 = sm.coo_matrix((np.ones(links2s), (rows2s, range(links2s))), shape=(n, links2s)) - \
                    sm.coo_matrix((np.ones(links2s), (cols2s, range(links2s))), shape=(n, links2s))
            
        pp = np.hstack([anchors, sensors])
        Si_minus_Aj = pp @ S1
        normsquare_a2s = np.sum(Si_minus_Aj**2, axis=0)

        Si_minus_Sj = sensors @ S2
        normsquare_s2s = np.sum(Si_minus_Sj**2, axis=0)

        res = np.sum(np.abs(normsquare_a2s-disa2s**2)) + np.sum(np.abs(normsquare_s2s-diss2s**2))

        return res


    @staticmethod
    def _func_grad_21(m, n, anchors, sensors, dda2s, dds2s, **kwargs):

        rowa2s, cola2s, disa2s = dda2s
        rows2s, cols2s, diss2s = dds2s

        if 'S1' in kwargs.keys():
            S1 = kwargs['S1']
        else:
            linka2s = len(rowa2s)
            S1 = sm.coo_matrix((np.ones(linka2s), (m + rowa2s, range(linka2s))), shape=(n+m, linka2s)) - \
                    sm.coo_matrix((np.ones(linka2s), (cola2s, range(linka2s))), shape=(n+m, linka2s))

        if 'S2' in kwargs.keys():
            S2 = kwargs['S2']
        else:
            links2s = len(rows2s)
            S2 = sm.coo_matrix((np.ones(links2s), (rows2s, range(links2s))), shape=(n, links2s)) - \
                    sm.coo_matrix((np.ones(links2s), (cols2s, range(links2s))), shape=(n, links2s))
            
        pp = np.hstack([anchors, sensors])
        Si_minus_Aj = pp @ S1
        normsquare_a2s = np.sum(Si_minus_Aj**2, axis=0)
        difference_a2s = normsquare_a2s - disa2s**2
        grada2s =  2.0 * np.sign(difference_a2s) * Si_minus_Aj @ S1[m:, :].T

        Si_minus_Sj = sensors @ S2
        normsquare_s2s = np.sum(Si_minus_Sj**2, axis=0)
        difference_s2s = normsquare_s2s-diss2s**2
        grads2s =  2.0 * np.sign(difference_s2s) * Si_minus_Sj @ S2.T

        res = np.sum(np.abs(difference_a2s)) + np.sum(np.abs(difference_s2s))

        return (grada2s + grads2s), res


    def sgd_21(self, sensors, rng, tol=10**(-4), epochs=100, **kwargs):

        batches = kwargs['batch']
        in_rate = kwargs['lr']
        diminishing_rate = kwargs['dr']

        m = self.m
        n = self.n
        anchors = (self.anchors).T
        dda2s = sm.find((self.dda2s).reshape((self.n, self.m)))
        dds2s = sm.find(np.triu(sp.distance.squareform(self.dds2s), 1))
        s_iter = (sensors.reshape((self.DIM, self.n))).copy()

        rowa2s, cola2s, disa2s = dda2s
        rows2s, cols2s, diss2s = dds2s
        linka2s = len(disa2s)
        links2s = len(diss2s)
        batch_size_a2s = linka2s//batches
        batch_size_s2s = links2s//batches
        current_loss = self._func_21(m, n, anchors, s_iter, dda2s, dds2s)

        print(0, current_loss)

        for k in range(epochs):

            rate = in_rate * diminishing_rate**(k)
            big_indices_a2s = list(range(linka2s))
            big_indices_s2s = list(range(links2s))
            rng.shuffle(big_indices_a2s)
            rng.shuffle(big_indices_s2s)

            for j in range(batches):
            
                indices_a2s = big_indices_a2s[j*batch_size_a2s:(j+1)*batch_size_a2s]
                indices_s2s = big_indices_s2s[j*batch_size_s2s:(j+1)*batch_size_s2s]

                part_grad, part_loss = self._func_grad_21(m, n, anchors, s_iter,\
                                                        (rowa2s[indices_a2s], cola2s[indices_a2s], disa2s[indices_a2s]),\
                                                            (rows2s[indices_s2s], cols2s[indices_s2s], diss2s[indices_s2s]))

                s_iter = s_iter - rate*part_grad

                if True in np.isnan(s_iter):
                    return s_iter, np.inf

            current_loss = self._func_21(m, n, anchors, s_iter, dda2s, dds2s)

            if (k+1)%20 == 0:
                print(k+1, current_loss)
            if current_loss < tol:
                return s_iter, current_loss

        return s_iter, current_loss


    def sgd_21_high_dim(self, sensors, rng, tol=10**(-4), epochs=100, **kwargs):

        batches = kwargs['batch']
        in_rate = kwargs['lr']
        penalty = kwargs['penalty']
        diminishing_rate = kwargs['dr']

        m = self.m
        n = self.n
        anchors = (self.anchors).T
        dda2s = sm.find((self.dda2s).reshape((self.n, self.m)))
        dds2s = sm.find(np.triu(sp.distance.squareform(self.dds2s), 1))
        d = self.DIM

        new_anchors = np.vstack([anchors, np.zeros((n, m))])
        new_sensors = np.vstack([(sensors.reshape((self.DIM, self.n))), rng.random((n, n)) - 0.5])
        s_iter = new_sensors.copy()


        rowa2s, cola2s, disa2s = dda2s
        rows2s, cols2s, diss2s = dds2s
        linka2s = len(disa2s)
        links2s = len(diss2s)
        batch_size_a2s = linka2s//batches
        batch_size_s2s = links2s//batches
        current_loss = self._func_21(m, n, new_anchors, s_iter, dda2s, dds2s)

        print(0, current_loss)

        for k in range(epochs):

            rate = in_rate * diminishing_rate**(k)
            big_indices_a2s = list(range(linka2s))
            big_indices_s2s = list(range(links2s))
            rng.shuffle(big_indices_a2s)
            rng.shuffle(big_indices_s2s)

            for j in range(batches):
            
                indices_a2s = big_indices_a2s[j*batch_size_a2s:(j+1)*batch_size_a2s]
                indices_s2s = big_indices_s2s[j*batch_size_s2s:(j+1)*batch_size_s2s]

                part_grad, part_loss = self._func_grad_21(m, n, new_anchors, s_iter,\
                                                        (rowa2s[indices_a2s], cola2s[indices_a2s], disa2s[indices_a2s]),\
                                                            (rows2s[indices_s2s], cols2s[indices_s2s], diss2s[indices_s2s]))
                # print(part_loss)
                s_iter = s_iter - rate * part_grad
                s_iter[d:, :] -= rate * penalty * s_iter[d:, :]
        
                if True in np.isnan(s_iter):
                    return s_iter[0:d, :], np.inf

            current_loss = self._func_21(m, n, new_anchors, s_iter, dda2s, dds2s) + 0.5 * penalty * np.sum(np.sum(s_iter[d:, :]**2))

            if (k+1)%5 == 0:
                print(k+1, current_loss)
            if current_loss < tol:
                return s_iter[0:d, :], current_loss

        return s_iter[0:d, :], current_loss


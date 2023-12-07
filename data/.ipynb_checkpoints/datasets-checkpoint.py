import numpy as np

import torch
from torch.utils.data import Dataset



class SyntheticData(Dataset):
    def __init__(
            self,
            n_samples,
            d1,
            d2,
            d3,
            theta1,
            theta2,
            theta3,
            thetay,
            p,
            q,
            ):
        self.n_samples = n_samples

        # generate data
        N1 = self._gen_N(theta1, d1)
        X1 = self._f1(N1)

        Ny = self._gen_Ny(thetay)
        self.y = self._fy(X1, Ny, n_sets=p)

        N2 = self._gen_N(theta2, d2)
        X2 = self._f2(self.y, N2, n_sets=q)

        N3 = self._gen_N(theta3, d3)

        self.X = np.concatenate((X1, X2, N3), axis=1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




    def _gen_N(self, theta, d):
        return np.random.binomial(1, theta, size=(self.n_samples, d))
    def _gen_Ny(self, theta):
        return np.random.binomial(1, theta, size=self.n_samples)
    
    def _f1(self, N1):
        return N1
    
    def _fy(self, X, N, n_sets):
        assert X.shape[0] == N.shape[0]
        power_helper = 2 ** np.arange(X.shape[1])
        X_int = np.sum(X * power_helper, axis=1)
        X_sets = X_int % n_sets

        pos_sets = np.random.permutation(n_sets)[: n_sets // 2]
        y_orig = np.isin(X_sets, pos_sets).astype(int)
        y = np.where(N, y_orig, 1 - y_orig)
        return y

    def _f2(self, y, N, n_sets):
        assert y.shape[0] == N.shape[0]
        d = N.shape[1]

        perm_sets_helper = np.random.permutation(n_sets)
        pos_sets = perm_sets_helper[: n_sets // 2]
        neg_sets = perm_sets_helper[n_sets // 2 :]
        pos_array_helper = np.random.choice(pos_sets, size=self.n_samples)
        neg_array_helper = np.random.choice(neg_sets, size=self.n_samples)
        X_sets = np.where(y == 1, pos_array_helper, neg_array_helper)

        sample_from_set_fn = np.vectorize(self._sample_from_set, excluded=["n_sets", 'd'])
        X_orig_int = sample_from_set_fn(set_idx=X_sets, n_sets=n_sets, d=d)

        bin_pos = 2 ** np.arange(0, d, 1)
        X_orig = (X_orig_int[:, None] & bin_pos).astype(int) >> np.arange(0, d, 1)
        X = np.where(N, X_orig, 1 - X_orig)
        return X

    def _sample_from_set(self, set_idx, n_sets, d):
        space_card = 2 ** d
        standard_set_size = space_card // n_sets
        remainder = space_card % n_sets

        if set_idx < remainder:
            start = set_idx * (standard_set_size + 1)
            end = start + standard_set_size + 1
        else:
            start = set_idx * standard_set_size + remainder
            end = start + standard_set_size

        return np.random.randint(start, end)

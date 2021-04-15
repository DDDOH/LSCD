# %%
import numpy as np
import matplotlib.pyplot as plt


class MultivariateNormal():
    def __init__(self, seq_len, shape='sin'):
        self.seq_len = seq_len
        self.mean_ls = self.mean_curve(seq_len, shape)
        if shape == 'sin':
            self.cov_mat = self.cov_mat(seq_len, min=-0.1, max=0.1)
        else:
            self.cov_mat = self.cov_mat(seq_len)
        self.shape = shape

    def mean_curve(self, seq_len, shape):
        """Return an array of length seqence_dim containing the mean value at each time step.

        Args:
            seq_len (int): the length of the sequence.
        """
        def mean_func(x):
            if shape == 'sin':
                return np.sin(x / 3)
            else:
                return - 0.02 * (x - seq_len/2) ** 2 + 50
        return np.array([mean_func(x) for x in range(seq_len)])

    def cov_mat(self, seq_len, min=None, max=None):
        """Randomly sample a covariance matrix.

        Args:
            seq_len (int): the length of the sequence.
            min (int): minimum value in the sampled matrix.
            max (int): maximum value in the sampled matrix.
        """
        _ = np.random.uniform(size=(seq_len, seq_len))
        cov_mat = np.matmul(_, _.T)
        if min:
            cov_mat = (cov_mat - np.min(cov_mat)) / \
                (np.max(cov_mat) - np.min(cov_mat)) * (max - min) - min
        return cov_mat

    def sample(self, n_sample):
        return np.random.multivariate_normal(mean=self.mean_ls, cov=self.cov_mat, size=n_sample)

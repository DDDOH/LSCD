import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
# from .. import utils


class CGMM():
    """Conditional density estimation using GMM.
    """

    def __init__(self, condition, dependent, n_components):
        self.condition = self._handle_shape(condition)
        self.dependent = self._handle_shape(dependent)
        self.n_components = n_components
        self.joint = np.stack([condition, dependent], axis=1)
        self.gmm_joint = GaussianMixture(
            n_components=n_components, random_state=0).fit(self.joint)

    @classmethod
    def _handle_shape(cls, data):
        if isinstance(data, (np.ndarray, list)):
            data = np.array(data)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=1)
        else:
            data = np.array([[data]])
        return data

    @classmethod
    def get_conditional_mean_cov_multivariate_gaussian(cls, condition, mean, cov):
        """Calculate conditional distribution for multivariate gaussian distribution.

        Args:
            condition (np.array): The known value at the first q time steps. An 1D array of length cond_len.
            mean (np.array): The mean for each dimension of the multivariate gaussian distribution.
                                An 1D array of length seq_len.
            cov (np.array): The covariance matrix of the multivariate gaussian distribution.
                            An 2D array of size (seq_len * seq_len).

        Returns:
            mean_cond (np.array): The mean for each dimension of the conditional multivariate gaussian distribution.
                                    An 1D array of length (seq_len - cond_len).
            cov_cond (np.array): The covariance matrix for the conditional multivariate gaussian distribution.
                                    An 2D array of size (seq_len - cond_len) * (seq_len - cond_len).
        """
        cond_len = len(condition)
        mean_k = mean[:cond_len]
        mean_u = mean[cond_len:]
        cov_kk = cov[:cond_len, :cond_len]
        cov_ku = cov[:cond_len, cond_len:]
        cov_uk = cov[cond_len:, :cond_len]
        cov_uu = cov[cond_len:, cond_len:]

        mean_cond = mean_u + \
            cov_uk.dot(np.linalg.inv(cov_kk)).dot((condition - mean_k).T).T
        cov_cond = cov_uu - cov_uk.dot(np.linalg.inv(cov_kk)).dot(cov_ku)
        return mean_cond, cov_cond

    def get_cond_gm(self, gmm_joint, new_condition):
        """Get the GMM of the conditional distribution P(X_pq|X_q=condition).

        Args:
            gmm_joint (GaussianMixture): The GMM of X.
            condition (np.array): An 2D array of size (n_sample, cond_len).

        Returns:
            [type]: [description]
        """
        n_components = gmm_joint.n_components
        n_sample, cond_len = np.shape(new_condition)[
            0], np.shape(new_condition)[1]
        seq_len = np.shape(gmm_joint.means_)[1]
        cond_gm_ls = np.empty(n_sample, dtype=object)
        for i in range(n_sample):  # for the i-th condition
            condition = new_condition[i, :]
            marginal = 0
            for l in range(n_components):  # for the l-th component
                mean_l = gmm_joint.means_[l, :]
                cov_l = gmm_joint.covariances_[l, :, :]

                mean_l_known = mean_l[:cond_len]
                cov_l_known = cov_l[:cond_len, :cond_len]
                try:
                    marginal += gmm_joint.weights_[l] * multivariate_normal.pdf(
                        condition, mean=mean_l_known, cov=cov_l_known, allow_singular=True)
                except:
                    print(cov_l_known)
                # strange staff: the above line of code will raise error LinAlgError even if utils.utils.isPD(cov_l_known) is True.
                # it seems that scipy use a stricter way to determine whether a matrix is singular. See https://stackoverflow.com/questions/35273908/scipy-stats-multivariate-normal-raising-linalgerror-singular-matrix-even-thou.

            # cov, mean and weight for each components
            # the shape matches exactly the shape in GaussianMixture
            cond_weight_ls = np.zeros(n_components)
            cond_mean_ls = np.zeros((n_components, seq_len - cond_len))
            cond_cov_ls = np.zeros(
                (n_components, seq_len - cond_len, seq_len - cond_len))
            # the k-th component of the conditional gaussian mixture distribution
            for k in range(n_components):
                mean_k = gmm_joint.means_[k, :]
                cov_k = gmm_joint.covariances_[k, :, :]

                # conditional mean and covariance matrix for k-th component
                cond_mean_k, cond_cov_k = self.get_conditional_mean_cov_multivariate_gaussian(
                    condition, mean=mean_k, cov=cov_k)
                cond_mean_ls[k, :] = cond_mean_k
                cond_cov_ls[k, :, :] = cond_cov_k

                mean_k_known = mean_k[:cond_len]
                cov_k_known = cov_k[:cond_len, :cond_len]

                # weight for k-th component for the conditional GMM
                cond_weight_ls[k] = gmm_joint.weights_[k] * multivariate_normal.pdf(
                    condition, mean=mean_k_known, cov=cov_k_known, allow_singular=True) / marginal

            cond_gm = GaussianMixture(n_components=n_components)
            cond_gm.weights_ = cond_weight_ls
            cond_gm.means_ = cond_mean_ls
            cond_gm.covariances_ = cond_cov_ls
            cond_gm_ls[i] = cond_gm
        return cond_gm_ls

    def cond_pdf(self, new_condition, new_dependent):
        new_condition = self._handle_shape(new_condition)
        new_dependent = self._handle_shape(new_dependent)
        gmm_cond_ls = self.get_cond_gm(
            self.gmm_joint, new_condition=new_condition)
        n_sample = np.shape(new_condition)[0]
        raise NotImplementedError
        # TODO add pdf support for GaussianMixture if needed.
        # return [gmm_cond_ls[i].pdf(new_dependent[i, :]) for i in range(n_sample)]


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    bivaraite_dataset = pd.read_csv('CondRNN/lscd/models/bivariate_dataset.csv')
    bivaraite_dataset = bivaraite_dataset.values
    # get conditional density estimation
    cgmm = CGMM(condition=bivaraite_dataset[:, 0],
                dependent=bivaraite_dataset[:, 1], n_components=10)

    x = np.arange(-50, 300, 5)
    y = np.arange(1, 6, 0.1)
    xx = np.repeat(x, len(y))
    yy = np.tile(y, len(x))
    cond_pdf_val = cgmm.cond_pdf(
        new_condition=x, new_dependent=y)
    z = cond_pdf_val.reshape(len(x), len(y)).T
    plt.contourf(x, y, z)
    plt.scatter(bivaraite_dataset[:, 0],
                bivaraite_dataset[:, 1], s=5, c='y')
    plt.xlabel('Condition')
    plt.ylabel('Dependent')
    plt.colorbar()
    plt.savefig('2D ckde.jpg')

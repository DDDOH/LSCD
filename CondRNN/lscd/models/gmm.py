"""Implement the conditinal gaussian mixture model.
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# from .. import utils
from . import base_cde
import progressbar


class SelfGMM():
    """The self defined GMM which supports sample and pdf method.
    """

    def __init__(self, mean, cov, weight):
        """[summary]

        Args:
            mean (np.array): 2D array of shape (n_component, n_feature)
            cov (np.array): 3D array of shape (n_component, n_feature, n_feature)
            weight (np.array): 1D array of shape (n_component)
        """
        self.mean = mean
        self.cov = cov
        self.weight = weight
        assert np.isclose(np.sum(weight), 1)
        self.n_feature = np.shape(mean)[1]
        self.n_component = np.shape(mean)[0]

    def sample(self, n_sample):
        """Sampling from the GMM.

        Args:
            n_sample (int): The amount of samples.

        Returns:
            np.array: 2D array of shape (n_sample, n_feature)
        """
        sample_ls = np.zeros((n_sample, self.n_feature))
        for i in range(n_sample):
            which_component = np.random.choice(
                self.n_component, p=self.weight)
            one_training_sample = np.random.multivariate_normal(
                mean=self.mean[which_component, :], cov=self.cov[which_component, :, :])
            sample_ls[i, :] = one_training_sample
        self.n_sample = n_sample
        self.sample_ls = sample_ls
        return sample_ls

    def pdf(self, x):
        """Calculate the pdf at x.

        Args:
            x (np.array): [description]

        Returns:
            [type]: [description]
        """
        pdf_val = 0
        for i in range(self.n_component):
            pdf_val += self.weight[i] * multivariate_normal.pdf(
                x, mean=self.mean[i, :], cov=self.cov[i, :, :])
        return pdf_val


class CGMM(base_cde.BaseCDE):
    """Conditional density estimation using GMM.
    """

    def __init__(self, condition, dependent, n_components):
        self.condition = condition
        self.dependent = dependent
        self.n_components = n_components
        self.joint = np.hstack([condition, dependent])
        self.gmm_joint = GaussianMixture(
            n_components=n_components, random_state=0).fit(self.joint)

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
        # for the i-th condition
        # TODO use the same cond_gmm for the same conditions to make the code runs faster.
        for i in progressbar.progressbar(range(n_sample)):
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

            cond_gm = SelfGMM(mean=cond_mean_ls,
                              cov=cond_cov_ls, weight=cond_weight_ls)
            cond_gm_ls[i] = cond_gm
        return cond_gm_ls

    def joint_pdf(self, new_condition, new_dependent):
        """Use the fitted GMM, get the joint pdf value at new_condition and new_dependent.

        Args:
            new_condition ([type]): [description]
            new_dependent ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_condition = self._handle_shape(new_condition)
        new_dependent = self._handle_shape(new_dependent)
        joint_samples = np.hstack([new_condition, new_dependent])
        return np.exp(self.gmm_joint.score_samples(joint_samples))

    def cond_pdf(self, new_condition, new_dependent):
        """Use the fitted GMM, get the conditional pdf value at new_condition and new_dependent.

        Args:
            new_condition ([type]): [description]
            new_dependent ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_condition = self._handle_shape(new_condition)
        new_dependent = self._handle_shape(new_dependent)
        cond_gmm_ls = self.get_cond_gm(
            self.gmm_joint, new_condition=new_condition)
        n_sample = np.shape(new_condition)[0]
        pdf_val_ls = np.zeros(n_sample)
        for i in range(n_sample):
            pdf_val_ls[i] = cond_gmm_ls[i].pdf(new_dependent[i, :])
        return pdf_val_ls

    def cond_samples(self, new_condition, n_sample):
        """Sample from the fitted conditional distribution.

        Args:
            new_condition ([type]): [description]
        """
        new_condition = np.expand_dims(new_condition, axis=0)
        cond_gmm = self.get_cond_gm(
            self.gmm_joint, new_condition=new_condition)
        return cond_gmm[0].sample(n_sample)


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    # CondRNN/lscd/models/bivariate_dataset.csv')
    bivaraite_dataset = pd.read_csv('CondRNN/lscd/models/bivariate_dataset.csv')
    bivaraite_dataset = bivaraite_dataset.values

    cgmm = CGMM(condition=bivaraite_dataset[:, 0],
                dependent=bivaraite_dataset[:, 1], n_components=5)

    x = np.arange(-50, 300, 20)
    y = np.arange(1, 6, 0.5)
    xx = np.repeat(x, len(y))
    yy = np.tile(y, len(x))

    plt.figure(figsize=(15, 4))
    plt.subplot(121)
    cond_pdf_val = cgmm.cond_pdf(
        new_condition=xx, new_dependent=yy)
    z_cond = cond_pdf_val.reshape(len(x), len(y)).T

    plt.contourf(x, y, z_cond)
    plt.scatter(bivaraite_dataset[:, 0],
                bivaraite_dataset[:, 1], s=5, c='white', edgecolors='black')
    plt.xlabel('Condition')
    plt.ylabel('Dependent')
    plt.title('Conditional pdf')
    plt.colorbar()

    plt.subplot(122)
    z_joint = cgmm.joint_pdf(new_condition=xx, new_dependent=yy).reshape(
        len(x), len(y)).T
    plt.contourf(x, y, z_joint)
    plt.scatter(bivaraite_dataset[:, 0],
                bivaraite_dataset[:, 1], s=5, c='white', edgecolors='black')
    plt.xlabel('Condition')
    plt.ylabel('Dependent')
    plt.title('Joint pdf')
    # This joint pdf looks bad because the scale of condition is much more larger than the scale of dependent.
    plt.colorbar()
    plt.savefig('2D cgmm.jpg')

    # check that under the same condition but with different dependent,
    # the joint_pdf / cond_pdf all has the same ratio.
    for i in range(len(x)):
        ratio = z_joint[:, i] / z_cond[:, i]
        assert np.isclose(ratio.min(), ratio.max())

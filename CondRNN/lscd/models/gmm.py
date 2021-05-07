import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import pickle
from sklearn.mixture import GaussianMixture
from .. import utils


def get_conditional_mean_cov_multivariate_gaussian(x_q, mean, cov):
    """Calculate conditional distribution for multivariate gaussian distribution.

    Args:
        x_q (np.array): The known value at the first q time steps. An 1D array of length cond_len.
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
    cond_len = len(x_q)
    mean_k = mean[:cond_len]
    mean_u = mean[cond_len:]
    cov_kk = cov[:cond_len, :cond_len]
    cov_ku = cov[:cond_len, cond_len:]
    cov_uk = cov[cond_len:, :cond_len]
    cov_uu = cov[cond_len:, cond_len:]

    mean_cond = mean_u + \
        cov_uk.dot(np.linalg.inv(cov_kk)).dot((x_q - mean_k).T).T
    cov_cond = cov_uu - cov_uk.dot(np.linalg.inv(cov_kk)).dot(cov_ku)
    return mean_cond, cov_cond


def get_cond_gm(gmm_joint, x_q):
    """Get the GMM of the conditional distribution P(X_pq|X_q=x_q).

    Args:
        gmm_joint (GaussianMixture): The GMM of X.
        x_q (np.array): An 1D array of length cond_len. (may update to support multiple conditions)

    Returns:
        [type]: [description]
    """
    n_components = gmm_joint.n_components
    cond_len = len(x_q)
    seq_len = np.shape(gmm_joint.means_)[1]
    marginal = 0
    for l in range(n_components):  # the l-th component
        mean_l = gmm_joint.means_[l, :]
        cov_l = gmm_joint.covariances_[l, :, :]

        mean_l_known = mean_l[:cond_len]
        cov_l_known = cov_l[:cond_len, :cond_len]
        try:
            marginal += gmm_joint.weights_[l] * multivariate_normal.pdf(
                x_q, mean=mean_l_known, cov=cov_l_known, allow_singular=True)
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
        cond_mean_k, cond_cov_k = get_conditional_mean_cov_multivariate_gaussian(
            x_q, mean=mean_k, cov=cov_k)
        cond_mean_ls[k, :] = cond_mean_k
        cond_cov_ls[k, :, :] = cond_cov_k

        mean_k_known = mean_k[:cond_len]
        cov_k_known = cov_k[:cond_len, :cond_len]

        # weight for k-th component for the conditional GMM
        cond_weight_ls[k] = gmm_joint.weights_[k] * multivariate_normal.pdf(
            x_q, mean=mean_k_known, cov=cov_k_known, allow_singular=True) / marginal

    cond_gm = GaussianMixture(n_components=n_components)
    cond_gm.weights_ = cond_weight_ls
    cond_gm.means_ = cond_mean_ls
    cond_gm.covariances_ = cond_cov_ls

    return cond_gm


if __name__ == '__main__':
    gmm_joint = GaussianMixture(n_components=2, random_state=0).fit(
        training_set.squeeze(-1))

    gmm_cond = get_cond_gm(gmm_joint, x_q=conditions.squeeze(-1)[0, :])

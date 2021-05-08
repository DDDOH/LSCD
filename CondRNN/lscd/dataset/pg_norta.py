# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import poisson
import progressbar
from .. import utils


class PGnorta():
    def __init__(self, base_intensity, cov, alpha):
        """Initialize a PGnorta dataset loader.

        Args:
            base_intensity (np.array): A list containing the mean of arrival count in each time step.
            cov (np.array): Covariance matrix of the underlying normal copula.
            alpha (np.array): A list containing the parameter of gamma distribution in each time step.
        """
        assert len(base_intensity) == len(alpha) and len(alpha) == np.shape(
            cov)[0] and np.shape(cov)[0] == np.shape(cov)[1]
        assert min(base_intensity) > 0, 'only accept nonnegative intensity'
        self.base_intensity = base_intensity
        self.p = len(base_intensity)  # the sequence length
        self.cov = cov
        self.alpha = alpha
        self.seq_len = len(base_intensity)

    def z_to_lam(self, Z, first=True):
        """Convert Z to intensity.

        Args:
            Z (np.array): The value of the normal copula for the first or last severl time steps or the whole sequence.
                          For one sample, it can be either a one dimension list with length q, or a 1 * q array.
                          For multiple samples, it should be a n * q array, n is the number of samples.
            first (bool, optional): Whether the given Z is for the first several time steps or not. Defaults to True.

        Returns:
            intensity (np.array): The value of intensity suggested by Z. An array of the same shape as Z.
        """
        if Z.ndim == 1:
            n_step = len(Z)
        else:
            n_step = np.shape(Z)[1]
        U = norm.cdf(Z)
        if first:
            B = gamma.ppf(q=U, a=self.alpha[:n_step],
                          scale=1/self.alpha[:n_step])
            intensity = B * self.base_intensity[:n_step]
        else:
            B = gamma.ppf(q=U, a=self.alpha[-n_step:],
                          scale=1/self.alpha[-n_step:])
            intensity = B * self.base_intensity[-n_step:]
        return intensity

    def sample(self, n_sample):
        """Sample arrival count from PGnorta model.

        Args:
            n_sample (int): Number of samples to generate.

        Returns:
            samples (np.array): An array of size (n_sample * seq_len), each row is one sample.
        """
        z = multivariate_normal.rvs(np.zeros(self.p), self.cov, n_sample)
        intensity = self.z_to_lam(z)
        count = np.random.poisson(intensity)
        return count

    # conditional sample
    def sample_cond(self, X_1q, n_sample):
        """Sample the arrival count in the last q - p time steps given the arrival 
        count in the first q time steps.

        We first sample from P(Z_q|X_q). 
        Denote the target distribution pi(z) as P(Z_q|X_q) = P(X_q|Z_q)*P(Z_q)/P(X_q).
        Since P(X_q) is hard to evaluate, accept-Reject Algorithm is used.
        Denote f(z) = P(X_q|Z_q)*P(Z_q) and a easy to sample distribution h(z).
        We must have f(z) <= c*h(z) for all z.
        We'll sample Z from pi(z) using the following steps:
        1.  Draw a candidate z from h and u from U(0,1), the uniform distribution on
            the interval (0,1).
        2.  If u <= f(z)/(c*h(z)), return z.
        3.  Otherwise, return to 1.
        A properly chosen h(z) will significantly improve sampling efficiency.

        With sampled Z_q, use the conditional distribution of multivariate normal distribution,
        we can sample from P(Z_qp|Z_q), then convert Z_qp to X_qp.

        Reference:
            https://jblevins.org/notes/accept-reject

        Args:
            X_1q (np.array): A list contains the arrival count in the first q time steps.
        """
        q = len(X_1q)
        assert q < self.p, 'q must smaller than p'
        assert min(X_1q) >= 0, 'only accept nonnegative arrival count'

        def p_z(Z):
            """Compute the density of observing Z.

            Args:
                Z (np.array): The value of normal distribution for the first q time steps.

            Returns:
                float: The density of the normal copula for first q time steps at Z.
            """
            return multivariate_normal.pdf(Z, mean=np.zeros(q), cov=h_cov)

        def p_x_on_z(X, Z):
            """Compute the conditional probability of observing the arrival count given Z.

            Args:
                X (np.array): The arrival count for the first q time steps.
                Z (np.array): The value of normal distribution for the first q time steps.

            Returns:
                float: the computed conditional probability.
            """
            intensity = self.z_to_lam(Z)
            if intensity.ndim == 2:
                return np.prod(poisson.pmf(k=X, mu=intensity), axis=1)

        def f_z(z): return p_x_on_z(X_1q, z) * p_z(z)

        # *************************************** sample from P(Z_q|X_q) *************************************** #

        # covariance matrix of the multivariate normal distribution h
        h_cov = self.cov[:q, :q]
        h_mean = np.zeros(q)  # mean of the multivariate normal distribution h
        # empirically set h as a multivariate normal distribution with mean zero and covariance self.cov[:q, :q] seems good

        # TODO improve accept ratio for high diemnsional case

        n_accepted = 0
        n_candidate = 0
        batch_size = n_sample * 50  # how many samples in one while iteration.
        accepted_Z_q = np.zeros((n_sample, q))
        with progressbar.ProgressBar(max_value=n_sample) as bar:
            while n_accepted < n_sample:
                n_candidate += batch_size
                Z = multivariate_normal.rvs(np.zeros(q), h_cov, batch_size)
                h_Z = multivariate_normal.pdf(Z, mean=np.zeros(q), cov=h_cov)

                f_Z = f_z(Z)
                uniform = np.random.rand(batch_size)

                c = np.max(f_Z / h_Z) * 1.2
                accept_prob = f_Z / h_Z / c

                whether_accepted = uniform <= accept_prob
                new_accepted = sum(whether_accepted)
                # the accepted samples that have space to be written into accepted_Z_q
                useful_accepted = min(n_sample - n_accepted, new_accepted)
                accepted_Z_q[n_accepted:n_accepted +
                             useful_accepted, :] = Z[whether_accepted, :][:useful_accepted, :]
                n_accepted += useful_accepted
                bar.update(n_accepted)

        assert n_accepted >= n_sample, "failed sampling enough data, expected {}, but got {}".format(
            n_sample, n_accepted)
        accepted_Z_q = accepted_Z_q[:n_sample, :]

        print('Total sampled Z: {}, accepted: {}, accept ratio: {:.4f}%, desired number of samples:{}'.format(
            n_candidate, n_accepted, n_accepted/n_candidate*100, n_sample))

        # *************************************** sample from P(X_qp|Z_q) *************************************** #
        def multivariate_normal_conditional_mu_cov(mean, cov, Z_q):
            """Compute the conditional distribution of a multivariate normal distribution.

            Args:
                mean (np.array): A list contains the mean vector of the joint multivariate normal distribution.
                cov (np.array): The covariance matrix of the joint multivariate normal distribution.
                Z_q (np.array): The given value of the first q dimension that take condition on.

            Returns:
                conditional_mu (np.array): The mean vector of the conditional distribution.
                conditional_cov (np.array): The covariance matrix of the conditional distribution.
            """
            dim_cond = len(Z_q)
            dim_joint = len(mean)
            assert dim_joint > dim_cond, 'the dimension of the given condition must smaller than the dimension of the joint distribution'

            m_q = mean[:dim_cond]
            m_qp = mean[dim_cond:]

            c_kk = cov[:dim_cond, :dim_cond]
            c_kn = cov[:dim_cond, dim_cond:]
            c_nk = cov[dim_cond:, :dim_cond]
            c_nn = cov[dim_cond:, dim_cond:]

            conditional_mean = m_qp + \
                c_nk.dot(np.linalg.inv(c_kk)).dot((Z_q - m_q).T).T
            conditional_cov = c_nn - c_nk.dot(np.linalg.inv(c_kk)).dot(c_kn)

            return conditional_mean, conditional_cov

        mean = np.zeros(self.p)
        Z_qp = np.empty((n_sample, self.p - q))
        for i in range(n_sample):
            conditional_mu, conditional_cov = multivariate_normal_conditional_mu_cov(
                mean=mean, cov=self.cov, Z_q=accepted_Z_q[i, :])
            Z_qp[i, :] = np.random.multivariate_normal(
                conditional_mu, conditional_cov, 1)

        intensity_qp = self.z_to_lam(Z_qp, first=False)
        X_qp = np.random.poisson(intensity_qp)

        return X_qp


def get_PGnorata_from_img():
    """Get parameters for PGnorta model from the image.

    Returns:
        (PGnorta): A PGnorta instance with the parameters come from images.
    """
    # TODO add support for more images.

    # read correlation matrix.
    corr_img = 'CondRNN/lscd/dataset/2_commercial_call_center_corr.png'
    img = mpimg.imread(corr_img)
    if corr_img == 'CondRNN/lscd/dataset/2_commercial_call_center_corr.png':
        colorbar = img[:, 1255, :3]
        p = 24
        width = 1220
        height = 1115

    n_color = np.shape(colorbar)[0]
    width_interval = width / (p + 1)
    height_interval = height/(p+1)
    width_loc = np.arange(width_interval/2, width-width_interval/2, p)
    height_loc = np.arange(height_interval/2, height-height_interval/2, p)
    corr_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            rgb_ij = img[int(height_loc[i]), int(width_loc[j]), :3]
            color_dist = np.sum(np.abs(colorbar - rgb_ij), axis=1)
            corr_mat[i, j] = 1 - np.argmin(color_dist)/n_color
    corr_mat = utils.utils.nearestPD(corr_mat)

    # read mean curve.
    mean_img = 'CondRNN/lscd/dataset/2_commercial_call_center_mean.png'
    img = mpimg.imread(mean_img)
    line_img = img[28:625, 145:1235, :3]
    p = 24
    count_min = 100
    count_max = 350
    y_axis_length, x_axis_length = np.shape(line_img)[0], np.shape(line_img)[1]

    loc = np.linspace(0, x_axis_length-1, p)
    mean = np.zeros(p)
    for i in range(p):
        mean[i] = (1 - np.argmin(line_img[:, int(loc[i]), 1]) /
                   y_axis_length)*(count_max-count_min) + count_min

    alpha = np.random.uniform(6, 12, p)
    return PGnorta(base_intensity=mean, cov=corr_mat, alpha=alpha)


def get_random_PGnorta(p=24):
    # get covariance matrix of the normal copula
    cov_mat = np.zeros((p, p))
    a = 0.2
    c = 0.1
    rho = 0.1
    for j in range(p):
        for k in range(p):
            cov_mat[j, k] = a * (rho ** abs(j - k)) + c
    plt.matshow(cov_mat)
    plt.colorbar()
    plt.title('cov mat')

    # get the base intensity for each time interval
    plt.figure(figsize=(3, 2))

    def intensity_shape(n):
        value = - (n - p/2)**2 / 10 + (p/2)**2
        return value

    base_lam = np.array([intensity_shape(n) for n in range(p)])/60
    plt.plot(base_lam)
    plt.suptitle('base intensity')

    # get alpha for each time interval
    alpha = np.random.uniform(6, 12, p)
    plt.figure(figsize=(3, 2))
    plt.plot(alpha)
    plt.suptitle('alpha')

    pgnorta = PGnorta(base_intensity=base_lam, cov=cov_mat, alpha=alpha)

    return pgnorta


def test_sample_cond(pgnorta):
    # TODO Add a doc string.
    # TODO Add KS Test.
    q = 2
    X_q = np.round(pgnorta.base_intensity[:q])
    n_sample_cond = 10000
    n_sample_joint = 100000

    # sample_cond directly
    sampled_X_qp = pgnorta.sample_cond(X_q, n_sample_cond)

    # sample from joint distribution, then filter out the samples have X_1_q arrivals in the first q time steps
    joint_X = pgnorta.sample(n_sample=n_sample_joint)
    filter = np.prod(np.equal(joint_X[:, :q], X_q), axis=1) == 1
    joint_Xqp_with_X_q = joint_X[filter, q:]

    print('PGnorta.sample_cond get {} samples,\nPGnorta.sample get {} samples,\nAmong them {} has the same arrival count as X_1_q.'.format(
        n_sample_cond, n_sample_joint, np.shape(joint_Xqp_with_X_q)[0]))

    plt.figure()
    plt.hist(joint_Xqp_with_X_q[:, 1], density=True,
             bins=np.arange(7), alpha=0.2)
    plt.hist(sampled_X_qp[:, 1], density=True, bins=np.arange(7), alpha=0.2)

    dirname = os.path.dirname(__file__)
    result_dir = os.path.join(dirname, 'test_sample_cond_hist.jpg')
    plt.savefig(result_dir)


if __name__ == "__main__":
    # pgnorta = get_random_PGnorta()
    # test_sample_cond(pgnorta)

    pg_norta = get_PGnorata_from_img()


# TODO: test acceptance ratio and sampling performance on high dimensional case.

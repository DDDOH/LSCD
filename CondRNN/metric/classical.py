# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
"""Compare the estimated conditional distribution with the real conditional distribution.

Accept samples from the estimated conditional distribution and the real conditional distribution.
Report the result for the designed metrics.
"""

# TODO
"""
Evaluation metric
    Run through queue
    pairwise plot
    marginal mean                       [o]
    marginal variance                   [o]
    Pierre correlation                  [o]
    Gaussian approximated W-distance    [o]
"""
# TODO run through queue
# TODO pairwise plot


def marginal_mean_plot(real_cond_samples, fake_cond_samples, ax):
    # marginal mean
    real_marginal_mean = np.mean(real_cond_samples, axis=0)
    fake_marginal_mean = np.mean(fake_cond_samples, axis=0)
    ax.plot(real_marginal_mean, label='Real mean')
    ax.plot(fake_marginal_mean, label='Fake mean')
    ax.legend()
    ax.set_xticks(np.arange(1, cond_len, int(cond_len)/10))
    ax.set_title('Marginal mean')


def marginal_var_plot(real_cond_samples, fake_cond_samples, ax):
    real_marginal_var = np.var(real_cond_samples, axis=0)
    fake_marginal_var = np.var(fake_cond_samples, axis=0)
    ax.plot(real_marginal_var, label='Real variance')
    ax.plot(fake_marginal_var, label='Fake variance')
    ax.legend()
    ax.set_xticks(np.arange(1, cond_len, int(cond_len)/10))
    ax.set_title('Marginal variance')


# Pierre correlation
def pierre_corr_plot(real_cond_samples, fake_cond_samples, ax):
    def pierre_corr(samples):
        """Compute the pierre correlation.

        Args:
            samples (np.array): 2D array of shape (n_sample, seq_len)

        Returns:
            np.array: 1D array of length (seq_len - 1).
                    The j-th element (the first element is indexed with 1) represents the correlation between X_{1:j} and X_{j+1:p}.
        """
        seq_len = np.shape(samples)[1]
        corr_ls = np.zeros(seq_len - 1)
        for i in range(1, seq_len):
            sum_x_i = np.sum(samples[:, :i], axis=1)
            sum_x_ip = np.sum(samples[:, i:], axis=1)
            corr_ls[i - 1] = np.correlate(sum_x_i, sum_x_ip)
        return corr_ls

    real_pierre_corr = pierre_corr(real_cond_samples)
    fake_pierre_corr = pierre_corr(fake_cond_samples)
    ax.plot(real_pierre_corr, label='Real samples')
    ax.plot(fake_pierre_corr, label='Fake samples')
    ax.set_xticks(np.arange(1, cond_len, int(cond_len)/10))
    ax.set_xlabel('$j$')
    ax.set_ylabel('$Corr(X_{1:j}, X_{j+1:p})$')
    ax.set_title('Pierre correlation')
    ax.legend()


# %%
# TODO Gaussian approximated W-distance
def w_distance(data_1, data_2):
    """Compute the Wasserstein between two normal distribution.

    Reference: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

    Args:
        data_1 (np.array): 2D array of size (batch_size, seq_len)
        data_2 (np.array): 2D array of size (batch_size, seq_len)
    """
    m_1, m_2 = np.mean(data_1, axis=0), np.mean(data_2, axis=0)
    cov_1, cov_2 = np.cov(data_1, rowvar=False), np.cov(
        data_2, rowvar=False)
    sqrtm_cov_1 = scipy.linalg.sqrtm(cov_1).real
    w_dist = np.linalg.norm(m_1 - m_2)**2 + \
        np.trace(cov_1 + cov_2 - 2 * scipy.linalg.sqrtm(
            np.matmul(np.matmul(sqrtm_cov_1, cov_2), sqrtm_cov_1)).real)
    return w_dist


def w_distance_plot(real_cond_samples, fake_cond_samples, ax):
    w_dist = w_distance(real_cond_samples, fake_cond_samples)
    string = 'W-distance between real samples\n and fake samples is {number:.{digits}f}'.format(
        number=w_dist, digits=4)
    ax.text(0.5, 0.5, string, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)


def evaluate(real_cond_samples, fake_cond_samples, dir_filename):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    marginal_mean_plot(real_cond_samples, fake_cond_samples, axs[0])
    marginal_var_plot(real_cond_samples, fake_cond_samples, axs[1])
    pierre_corr_plot(real_cond_samples, fake_cond_samples, axs[2])
    w_distance_plot(real_cond_samples, fake_cond_samples, axs[3])
    fig.savefig(dir_filename)


if __name__ == '__main__':
    # 2D array of shape (n_sample, cond_len)
    real_cond_samples = np.random.uniform(10, 20, size=(100, 200))
    # 2D array of shape (n_sample, cond_len)
    fake_cond_samples = np.random.uniform(20, 30, size=(1000, 200))
    cond_len = np.shape(real_cond_samples)[1]
    evaluate(real_cond_samples, fake_cond_samples, dir_filename='test.jpg')

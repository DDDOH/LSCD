"""Compare the estimated conditional distribution with the real conditional distribution.

Accept samples from the estimated conditional distribution and the real conditional distribution.
Report the result for the designed metrics.
"""
if __name__ == '__main__':
    from queue_cffi import ffi
    from queue_cffi.lib import single_server_queue
    from queue_cffi.lib import changing_multi_server_queue
    from queue_cffi.lib import const_multi_server_queue
else:
    from lscd.metric.queue_cffi import ffi
    from queue_cffi.lib import single_server_queue
    from queue_cffi.lib import changing_multi_server_queue
    from queue_cffi.lib import const_multi_server_queue
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import torch

# recompile = False
# TODO the recompile option to rerun the CondRNN/queue/cffi/queue_cffi_build.py


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
    """Plot the marginal mean for real samples and fake samples.

    Args:
        real_cond_samples (np.array): 2D array of shape (n_samples, cond_len)
        fake_cond_samples (np.array): 2D array of shape (n_samples, cond_len)
        ax ([type]): [description]
    """
    cond_len = np.shape(real_cond_samples)[1]
    real_marginal_mean = np.mean(real_cond_samples, axis=0)
    fake_marginal_mean = np.mean(fake_cond_samples, axis=0)
    ax.plot(real_marginal_mean, label='Real mean')
    ax.plot(fake_marginal_mean, label='Fake mean')
    ax.legend()
    ax.set_xticks(np.arange(1, cond_len, int(cond_len)/10))
    ax.set_title('Marginal mean')


def marginal_var_plot(real_cond_samples, fake_cond_samples, ax):
    """Plot the marginal variance for real samples and fake samples.

    Args:
        real_cond_samples (np.array): 2D array of shape (n_samples, cond_len)
        fake_cond_samples (np.array): 2D array of shape (n_samples, cond_len)
        ax ([type]): [description]
    """
    cond_len = np.shape(real_cond_samples)[1]
    real_marginal_var = np.var(real_cond_samples, axis=0)
    fake_marginal_var = np.var(fake_cond_samples, axis=0)
    ax.plot(real_marginal_var, label='Real variance')
    ax.plot(fake_marginal_var, label='Fake variance')
    ax.legend()
    ax.set_xticks(np.arange(1, cond_len, int(cond_len)/10))
    ax.set_title('Marginal variance')


# Pierre correlation
def pierre_corr_plot(real_cond_samples, fake_cond_samples, ax):
    """Plot the pierre correlation for real samples and fake samples.

    Args:
        real_cond_samples (np.array): 2D array of shape (n_samples, cond_len)
        fake_cond_samples (np.array): 2D array of shape (n_samples, cond_len)
        ax ([type]): [description]
    """
    def pierre_corr(samples):
        """Compute the pierre correlation coefficients.

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
            corr_ls[i - 1] = np.corrcoef(sum_x_i, sum_x_ip)[0, 1]
        return corr_ls
    cond_len = np.shape(real_cond_samples)[1]
    real_pierre_corr = pierre_corr(real_cond_samples)
    fake_pierre_corr = pierre_corr(fake_cond_samples)
    ax.plot(real_pierre_corr, label='Real samples')
    ax.plot(fake_pierre_corr, label='Fake samples')
    ax.set_xticks(np.arange(1, cond_len, int(cond_len)/10))
    ax.set_xlabel('$j$')
    ax.set_ylabel('$Corr(X_{1:j}, X_{j+1:p})$')
    ax.set_title('Pierre correlation')
    ax.legend()


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
    """Plot the w-distance for real samples and fake samples.

    Args:
        real_cond_samples (np.array or torch.tensor): 2D array of shape (n_samples, cond_len)
        fake_cond_samples (np.array or torch.tensor): 2D array of shape (n_samples, cond_len)
        ax ([type]): [description]
    """
    w_dist = w_distance(real_cond_samples, fake_cond_samples)
    string = 'W-distance between real samples\n and fake samples is {number:.{digits}f}'.format(
        number=w_dist, digits=4)
    ax.text(0.5, 0.5, string, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)


def arrival_epoch_simulator(arrival_count):
    """Sample arrival epochs given the arrival count for each period.

    Args:
        arrival_count (np.array): 1D array of length (n_period) or 2D array of (n_sample, n_period).

    Returns:
        np.array: 2D array of size (n_sample, max_n_arrival), represents the arrival epoch for customers in order.
                  It's very likely that the total number of arrivals are not the same for each sample,
                  thus we use max_n_arrival to represent the greatest arrival count among all the samples.
                  TODO more doc and example. 1fill with -1.

    """
    if arrival_count.ndim == 1:
        total_arrival_count = np.sum(arrival_count)
        arrival_epoch = np.sort(np.random.rand(0, 1000, total_arrival_count))

    else:
        arrival_epoch = np.array([])

    return arrival_epoch


def run_through_queue(real_cond_samples, fake_cond_samples):
    """Run through queue on real_cond_samples and fake_cond_samples.

    Treat the value as the arrival count for each period. Assume each period has an equal length.
    The service time and number of servers are specified within this function.

    Args:
        real_cond_samples ([type]): [description]
        fake_cond_samples ([type]): [description]
    """
    # first convert arrival count to arrival epoch
    arrival_epoch_from_real_samples = arrival_epoch_simulator(real_cond_samples)
    arrival_epoch_from_fake_samples = arrival_epoch_simulator(fake_cond_samples)

    # arrival_time_ls_c = ffi.new("float[]", [3, 5, 20, 25, 30])
    # service_time_ls_c = ffi.new("float[]", [3, 5, 2, 1, 5])
    # wait_time_ls_c = ffi.new("float[]", n_customer)

    wait_time = np.array(list(wait_time_ls_c))

    MODE = 'single'  # 'single' or 'changing_multi' or 'const_multi'
    if MODE == 'single':
        real_wait_time = single_server_queue(arrival_time_ls_c, service_time_ls_c,
                                             wait_time_ls_c, n_customer)
    if MODE == 'changing_multi':
        real_wait_time = changing_multi_server_queue(arrival_time_ls_c, service_time_ls_c,
                                                     wait_time_ls_c, n_customer)

    if MODE == 'const_multi':
        real_wait_time = const_multi_server_queue(arrival_time_ls_c, service_time_ls_c,
                                                  wait_time_ls_c, n_customer)

    return real_wait_time


def run_through_queue_plot(real_cond_samples, fake_cond_samples, ax):
    return 1


def evaluate(real_cond_samples, fake_cond_samples, dir_filename):
    """Make a comparision for real samples and fake samples.

    Will create a figure contains the marginal mean, marginal variance, pierre correlation and w-distance.

    Args:
        real_cond_samples (np.array or torch.tensor): 2D array of shape (n_samples, cond_len)
        fake_cond_samples (np.array or torch.tensor): 2D array of shape (n_samples, cond_len)
        dir_filename (string): The location to save the figure. Should include both the directory path and the file name, e.g., 'results/figure.jpg'.
    """
    if isinstance(real_cond_samples, torch.Tensor):
        real_cond_samples = real_cond_samples.numpy()
    if isinstance(fake_cond_samples, torch.Tensor):
        fake_cond_samples = fake_cond_samples.numpy()
    assert np.shape(real_cond_samples)[1] == np.shape(fake_cond_samples)[1]
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
    evaluate(real_cond_samples, fake_cond_samples,
             dir_filename='test_metric.jpg')

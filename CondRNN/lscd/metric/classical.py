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
        np.array: 1D array of list, each list inside represents the ordered arrival epoch for one sample.

    """
    assert arrival_count.dtype == int, 'Input arrival_count must be an array with int type.'
    assert np.min(arrival_count) >= 0, 'Arrival count cannot be negative value.'
    if arrival_count.ndim == 1:
        arrival_count = np.expand_dims(arrival_count, 0)

    one_period_len = 3600
    p = np.shape(arrival_count)[1]
    interval_boundary = [one_period_len * i for i in range(p + 1)]
    n_sample = np.shape(arrival_count)[0]
    # total arrival count for each sample
    total_arrival = np.sum(arrival_count, axis=1)
    arrival_epoch = np.empty(n_sample, dtype=object)
    for i in range(n_sample):
        arrival_epoch_i_sample = np.zeros(total_arrival[i])
        for j in range(p):
            interval_start = interval_boundary[j]
            interval_end = interval_boundary[j + 1]
            n_arrival_current_interval = arrival_count[i, j]
            past_arrival_count = np.sum(arrival_count[i, :j])
            arrival_epoch_i_sample[past_arrival_count:past_arrival_count+n_arrival_current_interval] = np.random.uniform(
                interval_start, interval_end, n_arrival_current_interval)
        arrival_epoch_i_sample = np.sort(arrival_epoch_i_sample)
        arrival_epoch[i] = arrival_epoch_i_sample.tolist()
    return arrival_epoch


def run_through_queue(arrival_epoch):
    """Run through queue on the given arrival epoch.

    Treat the value as the arrival count for each period. Assume each period has an equal length.
    The service time and number of servers are specified within this function.

    Args:
        real_cond_samples ([type]): [description]
    """

    MODE = 'single'  # 'single' or 'changing_multi' or 'const_multi'
    n_sample = len(arrival_epoch)
    n_arrival_each_sample = [len(arrival_epoch[i]) for i in range(n_sample)]
    wait_time_c_list = [ffi.new('float[]', n_arrival_each_sample[i])
                        for i in range(n_sample)]

    n_server = 5
    wait_time = np.empty(n_sample, dtype=list)
    for i in range(n_sample):
        wait_time_c = ffi.new('float[]', n_arrival_each_sample[i])
        const_multi_server_queue(ffi.new("float[]", arrival_epoch[i]),
                                 ffi.new("float[]", np.random.exponential(
                                     scale=10000, size=n_arrival_each_sample[i]).tolist()),
                                 wait_time_c,
                                 n_server,
                                 n_arrival_each_sample[i])
        wait_time[i] = list(wait_time_c)

    return wait_time


def summary_wait_time(wait_time):
    """Summarize wait time to statistic value like mean and variance of waiting time within each interval.

    Args:
        wait_time ([type]): [description]

    Raises:
        NotImplementedError: [description]
    """
    raise NotImplementedError
    wait_time_mean = 0
    wait_time_std = 0
    return wait_time_mean, wait_time_std


def run_through_queue_plot(real_cond_samples, fake_cond_samples, ax):
    arrival_epoch_real = arrival_epoch_simulator(real_cond_samples)
    arrival_epoch_fake = arrival_epoch_simulator(fake_cond_samples)
    wait_time_real = run_through_queue(arrival_epoch_real)
    wait_time_fake = run_through_queue(arrival_epoch_fake)
    wait_time_mean_real, wait_time_std_real = summary_wait_time(wait_time_real)
    wait_time_mean_fake, wait_time_std_fake = summary_wait_time(wait_time_fake)
    ax.plot(wait_time_mean_real, label='Real samples')
    ax.plot(wait_time_mean_fake, label='Fake samples')
    ax.set_xticks(np.arange(1, cond_len, int(cond_len)/10))
    ax.set_xlabel('$$')
    ax.set_ylabel('Wait time')
    ax.set_title('Run through queue results')
    ax.legend()


def evaluate_joint(real_samples, fake_samples, dir_filename):
    """Make a comparision for real samples and fake samples.

    Will create a figure contains the marginal mean, marginal variance, pierre correlation and w-distance.

    Args:
        real_samples (np.array or torch.tensor): 2D array of shape (n_samples, cond_len)
        fake_samples (np.array or torch.tensor): 2D array of shape (n_samples, cond_len)
        dir_filename (string): The location to save the figure. Should include both the directory path and the file name, e.g., 'results/figure.jpg'.
    """
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.detach().numpy()
    if isinstance(fake_samples, torch.Tensor):
        fake_samples = fake_samples.detach().numpy()
    assert np.shape(real_samples)[1] == np.shape(fake_samples)[1]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    marginal_mean_plot(real_samples, fake_samples, axs[0])
    marginal_var_plot(real_samples, fake_samples, axs[1])
    pierre_corr_plot(real_samples, fake_samples, axs[2])
    w_distance_plot(real_samples, fake_samples, axs[3])
    # run_through_queue_plot(real_cond_samples, fake_samples, axs[4])
    fig.savefig(dir_filename)
    plt.close(fig)


def evaluate_cond(real_cond_dep, fake_cond_dep):
    conditions = real_cond_dep.conditions
    n_condition = real_cond_dep.n_condition
    assert (real_cond_dep.conditions == fake_cond_dep.conditions).all()
    for i in range(n_condition):
        real_dependents = real_cond_dep.get_dep(condition=conditions[i, :])
        fake_dependents = fake_cond_dep.get_dep(condition=conditions[i, :])

    assert np.shape(real_samples)[1] == np.shape(fake_samples)[1]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    marginal_mean_plot(real_samples, fake_samples, axs[0])
    marginal_var_plot(real_samples, fake_samples, axs[1])
    pierre_corr_plot(real_samples, fake_samples, axs[2])
    w_distance_plot(real_samples, fake_samples, axs[3])
    # run_through_queue_plot(real_cond_samples, fake_samples, axs[4])
    fig.savefig(dir_filename)
    plt.close(fig)


if __name__ == '__main__':
    # 2D array of shape (n_sample, cond_len)
    real_cond_samples = np.random.uniform(10, 20, size=(100, 200))
    # 2D array of shape (n_sample, cond_len)
    fake_cond_samples = np.random.uniform(20, 30, size=(1000, 200))
    cond_len = np.shape(real_cond_samples)[1]

    arrival_count = np.random.randint(100, 500, size=(200, 10))
    arrival_epoch = arrival_epoch_simulator(arrival_count)
    wait_time = run_through_queue(arrival_epoch)

    evaluate_joint(real_cond_samples, fake_cond_samples,
                   dir_filename='test_metric.jpg')

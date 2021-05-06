# benckmark
# KDE
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
kernel_joint = stats.gaussian_kde(
    dataset=training_set.squeeze(-1).T, bw_method=None, weights=None)
conditions = training_set[:, :cond_len, 0]
kernel_condition = stats.gaussian_kde(
    dataset=conditions.T, bw_method=None, weights=None)

kernel_condition.resample(size=10)
# TODO sample from estimated conditional distribution


def kde_cond_pdf(x_qp, x_q):
    """The conditional pdf estimated by KDE.

    Args:
        x_qp (np.array): 2D array of size n_sample * (seq_len - cond_len) or 1D array of size (seq_len - cond_len).
        x_q (np.array): 2D array of size n_sample * cond_len or 1D array of size cond_len.

    Returns:
        p_x_qp_given_x_q (np.array): Conditional pdf for each sample.
    """
    x = np.concatenate([x_qp, x_q])
    p_x_q = kernel_condition.pdf(x_q)
    p_x = kernel_joint.pdf(x)
    p_x_qp_given_x_q = p_x / p_x_q
    if x_qp.ndim == 1:
        print('p(x_q, x_qp)={}, p(x_q)={}, p(x_qp|x_q)={}'.format(
            p_x, p_x_q, p_x_qp_given_x_q))
    return p_x_qp_given_x_q


a = kde_cond_pdf(x_qp=training_set.squeeze(-1)
                 [0, cond_len:], x_q=conditions.squeeze(-1)[0, :])

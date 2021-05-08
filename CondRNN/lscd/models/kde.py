# %%
"""
Implement the KDE conditional density estimation.

Use the method as in https://arxiv.org/pdf/1206.5278.pdf equation (1), (but the bindwidth selection of this paper is not considered)

Reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
"""
import numpy as np
from scipy.stats import gaussian_kde

# TODO sample from estimated conditional distribution


class CKDE():
    def __init__(self, condition, )


def kde_get_cond_pdf(condition, dependent):
    """Get the conditional pdf using kde method.

    Args:
        condition (np.array): 2D array of shape (n_sample, cond_len) or 1D array of length (n_sample) in case cond_len == 1.
        dependent (np.array): 2D array of shape (n_sample, seq_len - cond_len) or 1D array of length (n_sample) in case (seq_len - cond_len) == 1.
                              The order of samples should be the same as in condition.

    Returns:
        function: Return the conditional pdf (a function).
    """
    if condition.ndim == 1:
        condition = np.expand_dims(condition, axis=1)
    if dependent.ndim == 1:
        dependent = np.expand_dims(dependent, axis=1)
    joint = np.concatenate([condition, dependent], axis=1)
    kernel_joint = gaussian_kde(dataset=joint.T, bw_method=None, weights=None)
    kernel_condition = gaussian_kde(
        dataset=condition.T, bw_method=None, weights=None)

    def kde_cond_pdf(new_condition, new_dependent):
        """Calculate the pdf value using the estimated kde.

        Args:
            new_condition (np.array or float or int): 2D array of shape (n_sample, cond_len)
                                                      or 1D array of length (n_sample) in case cond_len == 1
                                                      or float/int in case n_sample == 1 and cond_len == 1.
            new_dependent (np.array): 2D array of shape (n_sample, seq_len - cond_len)
                                      or 1D array of length (n_sample) in case (seq_len - cond_len) == 1
                                      or float/int in case n_sample == 1 and (seq_len - cond_len) == 1.
                                      The order of samples should be the same as in condition.

        Returns:
            np.array: The pdf value at (new_condition, new_dependent).
        """
        if isinstance(new_condition, (np.ndarray, list)):
            new_condition = np.array(new_condition)
            if new_condition.ndim == 1:
                new_condition = np.expand_dims(new_condition, axis=1)
        else:
            new_condition = np.array([[new_condition]])

        if isinstance(new_dependent, (np.ndarray, list)):
            new_dependent = np.array(new_dependent)
            if new_dependent.ndim == 1:
                new_dependent = np.expand_dims(new_dependent, axis=1)
        else:
            new_dependent = np.array([[new_dependent]])

        new_joint = np.concatenate([new_condition, new_dependent], axis=1)
        return kernel_joint.pdf(new_joint.T) / kernel_condition.pdf(new_condition.T)

    return kde_cond_pdf


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    bivaraite_dataset = pd.read_csv('CondRNN/lscd/models/bivariate_dataset.csv')
    bivaraite_dataset = bivaraite_dataset.values
    # get conditional density estimation
    cond_pdf = kde_get_cond_pdf(
        condition=bivaraite_dataset[:, 0], dependent=bivaraite_dataset[:, 1])

    x = np.arange(-50, 300, 5)
    y = np.arange(1, 6, 0.1)
    xx = np.repeat(x, len(y))
    yy = np.tile(y, len(x))
    z = cond_pdf(xx, yy).reshape(len(x), len(y)).T
    plt.contourf(x, y, z)
    plt.scatter(bivaraite_dataset[:, 0],
                bivaraite_dataset[:, 1], s=5, c='y')
    plt.xlabel('Condition')
    plt.ylabel('Dependent')
    plt.colorbar()
    plt.savefig('2D ckde.jpg')

# %%

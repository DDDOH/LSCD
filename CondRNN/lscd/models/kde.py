# %%
"""
Implement the KDE conditional density estimation.

Use the method as in https://arxiv.org/pdf/1206.5278.pdf equation (1), (but the bindwidth selection of this paper is not considered)

Reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
"""
import numpy as np
from scipy.stats import gaussian_kde
from . import base_cde
# TODO sample from estimated conditional distribution


class CKDE(base_cde.BaseCDE):
    """The class for kernel conditional density estimation.
    """

    def __init__(self, condition, dependent):
        self.condition = condition
        self.dependent = dependent
        self.joint = np.hstack([self.condition, self.dependent])

        self.kernel_joint = gaussian_kde(
            dataset=self.joint.T, bw_method=None, weights=None)
        self.kernel_condition = gaussian_kde(
            dataset=self.condition.T, bw_method=None, weights=None)

    def joint_pdf(self, new_condition, new_dependent):
        """Use the fitted KDE, get the joint pdf value at new_condition and new_dependent.

        Args:
            new_condition ([type]): [description]
            new_dependent ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_joint = np.hstack([new_condition, new_dependent])
        return self.kernel_joint.pdf(new_joint.T)

    def cond_pdf(self, new_condition, new_dependent):
        """Use the fitted KDE, get the conditional pdf value at new_condition and new_dependent.

        Args:
            new_condition ([type]): [description]
            new_dependent ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_joint = np.hstack([new_condition, new_dependent])
        return self.kernel_joint.pdf(new_joint.T) / self.kernel_condition.pdf(new_condition.T)

    def cond_samples(self, new_condition, n_sample):
        """Sample from the fitted conditional distribution.

        Args:
            new_condition ([type]): [description]
        """
        raise NotImplementedError


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    # CondRNN/lscd/models/bivariate_dataset.csv')
    bivaraite_dataset = pd.read_csv('CondRNN/lscd/models/bivariate_dataset.csv')
    bivaraite_dataset = bivaraite_dataset.values

    ckde = CKDE(condition=bivaraite_dataset[:, 0],
                dependent=bivaraite_dataset[:, 1])

    x = np.arange(-50, 300, 20)
    y = np.arange(1, 6, 0.5)
    xx = np.repeat(x, len(y))
    yy = np.tile(y, len(x))

    plt.figure(figsize=(15, 4))
    plt.subplot(121)
    cond_pdf_val = ckde.cond_pdf(
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
    z_joint = ckde.joint_pdf(new_condition=xx, new_dependent=yy).reshape(
        len(x), len(y)).T
    plt.contourf(x, y, z_joint)
    plt.scatter(bivaraite_dataset[:, 0],
                bivaraite_dataset[:, 1], s=5, c='white', edgecolors='black')
    plt.xlabel('Condition')
    plt.ylabel('Dependent')
    plt.title('Joint pdf')
    # This joint pdf looks bad because the scale of condition is much more larger than the scale of dependent.
    plt.colorbar()
    plt.savefig('2D ckde.jpg')

    # check that under the same condition but with different dependent,
    # the joint_pdf / cond_pdf all has the same ratio.
    for i in range(len(x)):
        ratio = z_joint[:, i] / z_cond[:, i]
        assert np.isclose(ratio.min(), ratio.max())

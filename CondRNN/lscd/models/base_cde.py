import numpy as np
import torch

NumberTypes = (int, float, complex)


class BaseCDE():
    """Base class for conditional density estimation.

    Methods:
        Initialize method accepts the dataset.
        Compute the conditional pdf given a new condition.
    """

    def __init__(self, condition, dependent):
        self.condition = condition
        self.dependent = dependent
        # self.cond_dim = np.shape(condition)[1]
        # self.dpdt_dim = np.shape(dependent)[1]

    def cond_pdf(self, condition):
        raise NotImplementedError

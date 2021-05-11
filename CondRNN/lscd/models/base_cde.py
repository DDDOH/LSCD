import numpy as np


class BaseCDE():
    """Base class for conditional density estimation.

    Methods:
        Initialize method accepts the dataset.
        Compute the conditional pdf given a new condition.
    """

    def __init__(self, condition, dependent):
        self.condition = condition
        self.dependent = dependent

    def cond_pdf(self, condition):
        raise NotImplementedError

    @classmethod
    def _handle_shape(cls, data):
        """Reshape the data to a 2D numpy array of shape (n_sample, n_feature).

        Args:
            data (list or np.array): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(data, (np.ndarray, list)):
            data = np.array(data)
            if data.ndim == 1:  # Only one sample
                data = np.expand_dims(data, axis=1)
            else:
                data = np.array([[data]])
        return data

"""Module for some useful data structures.
"""
import numpy as np


class CondDep():
    """Data structure to store condition - dependent pairs.
    """

    def __init__(self):
        self.conditions = None
        self.n_condition = 0
        self.dependent_for_each_condition = []

    def add_cond_dep_pair(self, condition, dependent):
        condition = np.expand_dims(condition, axis=0)
        if self.n_condition == 0:
            self.conditions = condition
        else:
            self.conditions = np.vstack([self.conditions, condition])
        self.n_condition += 1
        self.dependent_for_each_condition.append(dependent)

    def get_dep(self, condition):
        for i in range(self.n_condition):
            if (self.conditions[i, :] == condition).all():
                return self.dependent_for_each_condition[i]
        print('Cannot find matching condition.')
        return None

    def __repr__(self):
        return "Condition-Dependent pairs contains {} condition(s).".format(self.n_condition)


class Scaler():
    def __init__(self, data):
        """Initialize the Scaler class.

        Args:
            data (np.array): 2d numpy array. Each row represents one sequence.
                             Each column is one timestep.
                             Only works for sequence with one feature each time step, temporarily.
        """
        self.data = data
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data=None):
        """Transform data to standard normal distribution.

        Transform data for each time stamp to an individual standard normal distribution.

        Args:
            data (np.array, optional): If not specified, default to the dataset used to initialize this Scaler.
                                       Defaults to None.

        Returns:
            np.array: The transformed data.
        """
        if not data:
            data = self.data
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


if __name__ == '__main__':
    # for CondDep
    cond_dep = CondDep()
    conditions = np.random.uniform(3, 10, size=(100, 5))
    n_cond = np.shape(conditions)[0]
    for i in range(n_cond):
        dep_for_cur_cond = np.random.uniform(5, 20, size=(100, 10))
        cond_dep.add_cond_dep_pair(
            condition=conditions[i, :], dependent=dep_for_cur_cond)
    print(cond_dep.get_dep(condition=conditions[i, :]))

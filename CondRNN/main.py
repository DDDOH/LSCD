# %%
import torch
import torch.nn as nn
import synthetic_dataset
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import os
from torchsummary import summary

# Get synthetic training set.
seq_len = 100
cond_len = 40
n_sample = 50

dataset = synthetic_dataset.SyntheticDataset(seq_len=seq_len)
training_set = dataset.sample(n_sample=n_sample)
# plt.plot(training_set.T, color='r', alpha=0.02)

# pre processing
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

scale = False
if scale:
    scaler = Scaler(training_set)
    training_set = scaler.transform()

training_set = torch.Tensor(training_set).unsqueeze(-1)


# %%

class CondLSTM(nn.Module):
    """Conditional LSTM"""

    def __init__(self, in_feature, out_feature, n_layers=1, hiddem_dim=256):
        """Initialize a Conditional LSTM.

        Args:
            in_feature (int): Number of features of input sequence in each time step.
            out_feature (int): Number of features of output sequence in each time step.
            n_layers (int, optional): Number of LSTM layers. Defaults to 1.
            hiddem_dim (int, optional): Hidden dimension of LSTM. Defaults to 256.
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.lstm = nn.LSTM(input_size=in_feature, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, 100),
                                 nn.LeakyReLU(),
                                 nn.Linear(100, out_feature))

    def forward(self, value, noise, hidden=None, only_last=False):
        """
        Args:
            value ([type]): Previous value.
            noise ([type]): Noise.
            only_last (bool, optional): Whether only return the last predicted time step. Defaults to False.
            hidden ([type], optional): Previous LSTM hidden state. Defaults to None.

        Returns:
            output ([type]): Predicted value. If only_last set to True, return the predicted value for all time steps.
                             Else return the predicted value for only the last time step.
        """
        input = torch.cat((value, noise), dim=2)
        recurrent_features, hidden = self.lstm(input, hidden)
        output = self.mlp(recurrent_features)
        return output, hidden


# Train LSTM to predict
noise_dim = 2
hidden_dim = 256
n_feature = 1
# may also use embedded time step as input feature
in_feature = noise_dim + n_feature
n_layers = 1
out_feature = n_feature
# since the number of data in training set is small, use the whole training set in each iteration, temporarily
batch_size = n_sample

def predict_full(dataset, predictor):
    """Train predictor for one step.

    Args:
        dataset (torch.Tensor): Full data set of size (batch_size, seq_len, n_feature)

    Returns:
        pred_value (torch.Tensor): Predicted value.
    """
    dataset = torch.Tensor(dataset)
    batch_size, seq_len, n_feature = dataset.shape
    input = dataset[:,:-1,:]
    target = dataset[:,1:,:]
    noise = torch.randn(batch_size, seq_len - 1, noise_dim)
    pred_value, hidden = predictor(value=input, noise=noise)

    return pred_value


def predict_condition(condition, predictor):
    """Predict using CondLSTM with given condition.

    Args:
        condition (np.array): 2d array. Each row represents one time series. Each column represents one time step.
    """

    condition = torch.Tensor(condition)

    batch_size, cond_len, n_feature = condition.shape
    # run LSTM on the given condition
    noise = torch.randn(batch_size, cond_len, noise_dim)
    pred_value, hidden = predictor(value=condition, noise=noise)

    # get the prediction value for the first unknown time step
    pred_value = pred_value.detach()[:, -1, :].unsqueeze(1)

    # run LSTM on unknown time step
    cond_pred_value = []
    for i in range(seq_len - cond_len):
        noise = torch.randn(batch_size, 1, noise_dim)
        hidden[0].detach_()
        hidden[1].detach_()
        pred_value = pred_value.detach()
        pred_value, hidden = predictor(value=pred_value, noise=noise, hidden=hidden)
        cond_pred_value.append(pred_value)

    cond_pred_value = torch.cat(cond_pred_value, dim=1)
    return cond_pred_value


def train_iter(predictor, lr=0.0002, epochs=1000):

    dirname = os.path.dirname(__file__)
    result_dir = os.path.join(dirname, 'result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    loss = nn.MSELoss()
    target = training_set[:, 1:]
    optimizer = torch.optim.Adam(predictor.parameters(), lr)

    loss_curve = []
    for i in range(epochs):
        predictor.zero_grad()
        pred_value = predict_full(training_set, predictor)
        mse_loss = loss(pred_value, target)
        mse_loss.backward()
        print(mse_loss)
        loss_curve.append(mse_loss.detach())
        optimizer.step()

        if i % 2 == 0:
            # Plot predicted value vs target value.
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.plot(pred_value.squeeze(-1).detach().T, c='r', alpha=0.02)
            plt.plot(target.squeeze(-1).T, c='g', alpha=0.02)
            plt.subplot(122)
            condition = training_set[:, :cond_len]
            true_on_condition = training_set[:, cond_len:]
            pred_on_condition = predict_condition(condition, predictor)
            plt.plot(pred_on_condition.squeeze(-1).detach().T, c='r', alpha=0.02)
            plt.plot(true_on_condition.squeeze(-1).T, c='g', alpha=0.02)
            filename = os.path.join(result_dir, 'prediction_step_%d.jpg' % (i))
            plt.savefig(filename)
            plt.close()

            # Plot loss curve.
            plt.figure()
            plt.semilogy(loss_curve, c='r')
            filename = os.path.join(result_dir, 'loss_curve.jpg')
            plt.savefig(filename)
            plt.close()


predictor = CondLSTM(in_feature=in_feature, out_feature=out_feature)
# seems complicated for hidden if not use two seperate hidden states
# summary(predictor, [(cond_len, n_feature), (cond_len, noise_dim), (2, 1, hidden_dim)])

train_iter(predictor,lr=0.0005)

# %%

def w_distance(data_1, data_2):
    """Compute the Wasserstein between two normal distribution.

    Reference: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

    Args:
        data_1 (np.array): The first set of data. Each row is one sequence.
        data_2 (np.array): The second set of data. Each row is one sequence.
    """
    m_1, m_2 = np.mean(data_1, axis=0), np.mean(data_1, axis=0)
    cov_1, cov_2 = np.cov(data_1, rowvar=False), np.cov(data_2, rowvar=False)
    w_dist = np.linalg.norm(m_1 - m_2)**2 + np.trace(cov_1 + cov_2 -
                                                     2 * sqrtm(np.matmul(np.matmul(sqrtm(cov_1), cov_2), sqrtm(cov_1))))
    return w_dist


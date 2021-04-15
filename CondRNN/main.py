# %%
from geomloss import SamplesLoss
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import os
import io

import datetime
import utils
import dataset
import models

# See how python import self defined modules: https://realpython.com/python-import/

# Get synthetic training set.
seq_len = 100
cond_len = 40
n_sample = 300

data_name = 'PGnorta'  # 'PGnorta' or 'multivariate_normal'
model_name = 'CondMLP'  # 'CondLSTM' or 'CondMLP'
if data_name == 'multivariate_normal':
    data = dataset.multivariate_normal.MultivariateNormal(seq_len=seq_len)
if data_name == 'PGnorta':
    data = dataset.pg_norta.get_random_PGnorta(p=seq_len)
training_set = data.sample(n_sample=n_sample)


dirname = os.path.dirname(__file__)
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
result_dir = os.path.join(dirname, 'results', date)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
writer = SummaryWriter(result_dir)

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


scale = True
if scale:
    scaler = Scaler(training_set)
    training_set = scaler.transform()

training_set = torch.Tensor(training_set).unsqueeze(-1)


# TODO: Poisson count simulator layer


# MLP
class CondMLP(nn.Module):
    def __init__(self, seed_dim, cond_len, seq_len, hidden_dim=256):
        """Initialize a conditional MLP.

        Args:
            seed_dim (int): The dimension of random seed.
            cond_len (int): Length of the condition (q).
            seq_len (int): Length of the sequence (p).
            hidden_dim (int, optional): Size of hidden layer. Defaults to 256.
        """
        super().__init__()
        self.seed_dim = seed_dim
        self.cond_len = cond_len
        self.seq_len = seq_len
        main = nn.Sequential(
            nn.Linear(seed_dim + cond_len, hidden_dim), nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_dim, seq_len - cond_len),
        )
        self.main = main

    def forward(self, X_q, noise):
        """Generate arrival count in the last p-q time intervals, given the arrival count in the first q time steps.

        Args:
            X_q (torch.Tensor): Arrival count in the first q time steps. A matrix of shape batch_size * q.
            noise (torch.Tensor): Random noise. A matrix of shape batch_size * self.seed_dim

        Returns:
            output (torch.Tensor): Arrival count in the last p - q time steps. A matrix of shape batch_size * (p - q).
        """
        mlp_input = torch.cat((X_q, noise), dim=1)
        output = self.main(mlp_input)
        return output


def predict_MLP(predictor, dataset):
    """Predict the arrival count in the last p - q time steps, given the arrival count in the first p time steps.

    Args:
        dataset (np.array): Full data set of size (batch_size, seq_len, n_feature). Only accept n_feature=1.
        predictor (CondMLP): A CondMLP model.

    Returns:
        pred_x_qp (torch.Tensor): Predicted arrival count in the last p - q time steps.
    """
    assert isinstance(predictor, CondMLP), "input model must be a CondMLP"
    x_q = torch.Tensor(dataset[:, :predictor.cond_len, 0])
    batch_size = x_q.shape[0]
    noise = torch.randn((batch_size, predictor.seed_dim))
    pred_x_qp = predictor(X_q=x_q, noise=noise)
    return pred_x_qp.unsqueeze(-1)


# LSTM
class CondLSTM(nn.Module):
    """Conditional LSTM"""

    def __init__(self, in_feature, out_feature, n_layers=1, hiddem_dim=256):
        """Initialize a Conditional LSTM. Can only accept time series with one feature each time step.

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

        # trainable initial state
        h_0 = torch.randn(self.n_layers, 1, self.hidden_dim)
        c_0 = torch.randn(self.n_layers, 1, self.hidden_dim)
        self.h_0 = nn.Parameter(h_0, requires_grad=True)
        self.c_0 = nn.Parameter(c_0, requires_grad=True)

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
        if not hidden:
            batch_size = input.shape[0]
            hidden = (self.h_0.repeat(1, batch_size, 1),
                      self.c_0.repeat(1, batch_size, 1))
            recurrent_features, hidden = self.lstm(input, hidden)
        else:
            recurrent_features, hidden = self.lstm(input, hidden)
        output = self.mlp(recurrent_features)
        return output, hidden


def predict_full_RNN(dataset, predictor):
    """Predict the next time step on the whole sequence using RNN model.

    Args:
        dataset (torch.Tensor): Full data set of size (batch_size, seq_len, n_feature)

    Returns:
        pred_value (torch.Tensor): Predicted value.
    """
    dataset = torch.Tensor(dataset)
    batch_size, seq_len, n_feature = dataset.shape
    input = dataset[:, :-1, :]
    target = dataset[:, 1:, :]
    noise = torch.randn(batch_size, seq_len - 1, noise_dim)
    pred_value, hidden = predictor(value=input, noise=noise)

    return pred_value


def predict_condition_LSTM(condition, predictor):
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
        pred_value, hidden = predictor(
            value=pred_value, noise=noise, hidden=hidden)
        cond_pred_value.append(pred_value)

    cond_pred_value = torch.cat(cond_pred_value, dim=1)
    return cond_pred_value


def plot_mean_var_cov(time_series_batch, dpi=35):
    """Convert a batch of time series to a tensor with a grid of their plots of marginal mean, marginal variance and covariance matrix

    Args:
        time_series_batch (Tensor): TODO (batch_size, seq_len, dim) tensor of time series
        dpi (int): dpi of a single image

    Output:
        single (channels, width, height): shaped tensor representing an image
    """
    images = []

    # marginal mean
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Marginal mean')
    ax.plot(np.mean(time_series_batch.numpy(), axis=0))
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(canvas.get_width_height()[::-1] + (3,))
    images.append(img_data)
    plt.close(fig)

    # marginal variance
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Marginal variance')
    ax.plot(np.var(time_series_batch.numpy(), axis=0))
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(canvas.get_width_height()[::-1] + (3,))
    images.append(img_data)
    plt.close(fig)

    # covariance matrix
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Covariance matrix')
    ax.matshow(np.cov(time_series_batch.numpy().T))
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(canvas.get_width_height()[::-1] + (3,))
    images.append(img_data)
    plt.close(fig)

    # Swap channel
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    # Make grid
    grid_image = vutils.make_grid(images.detach(), nrow=3)
    return grid_image


def train_iter(model, loss_func, lr=0.0002, epochs=1000):
    """Train the model with given loss.

    Args:
        model ([type]): [description]
        loss_func ([type]): Loss function. The first input is sequence generated by model, second input is target.
        lr (float, optional): [description]. Defaults to 0.0002.
        epochs (int, optional): [description]. Defaults to 1000.
    Returns:
        model (torch.Tensor): Trained model.
    """

    optimizer = torch.optim.Adam(predictor.parameters(), lr)

    # loss_curve = []
    if isinstance(model, CondLSTM):
        h_records = []
        c_records = []
        target = training_set[:, 1:]
    if isinstance(model, CondMLP):
        target = training_set[:, cond_len:]

    for i in range(epochs):
        predictor.zero_grad()
        pred_value = predict_MLP(predictor, training_set)
        # TODO pred_value = predict_full_RNN(training_set, predictor)
        loss = loss_func(pred_value, target)
        loss.backward()
        print('[%d/%d] Loss: %.4f' % (i, epochs, loss.item()))
        # loss_curve.append(loss.detach())
        optimizer.step()

        if model == 'CondLSTM':
            h_records.append(predictor.h_0.clone().detach().squeeze())
            c_records.append(predictor.c_0.clone().detach().squeeze())

        if i % 10 == 0:
            # Plot predicted value vs target value.
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.plot(pred_value.squeeze(-1).detach().T, c='r', alpha=0.02)
            plt.plot(target.squeeze(-1).T, c='g', alpha=0.02)
            plt.subplot(122)
            condition = training_set[:, :cond_len]
            true_dependent = training_set[:, cond_len:]

            # TODO: modify old only LSTM code
            # TODO pred_on_condition = predict_condition(condition, predictor)
            # TODO plt.plot(pred_on_condition.squeeze(-1).detach().T,
            #  c='r', alpha=0.02)
            fake_dependent = predict_MLP(predictor, training_set).detach()

            plt.plot(fake_dependent.squeeze(-1).T, c='r', alpha=0.1)
            plt.plot(true_dependent.squeeze(-1).T, c='g', alpha=0.1)
            filename = os.path.join(result_dir, 'prediction_step_%d.jpg' % (i))
            plt.savefig(filename)
            plt.close()

            if isinstance(model, CondLSTM):
                plt.figure(figsize=(20, 5))
                # plt.pcolormesh(np.stack(h_records, axis=0))
                plt.subplot(121)
                plt.plot(np.stack(h_records, axis=0))
                plt.subplot(122)
                plt.plot(np.stack(c_records, axis=0))
                filename = os.path.join(result_dir, 'h_c_records.jpg')
                plt.savefig(filename)
                plt.close()

            fake_statistic = plot_mean_var_cov(
                fake_dependent.squeeze(-1), dpi=45)
            writer.add_image("fake", fake_statistic, i)

        writer.add_scalar('Loss curve',
                          loss.item(), i)

        writer.flush()

    return model


if model_name == 'CondLSTM':
    # Config for CondLSTM
    noise_dim = 2
    hidden_dim = 256
    n_feature = 1
    # may also use embedded time step as input feature
    in_feature = noise_dim + n_feature
    n_layers = 1
    out_feature = n_feature
    # since the number of data in training set is small, use the whole training set in each iteration, temporarily
    batch_size = n_sample
    predictor = CondLSTM(in_feature=in_feature, out_feature=out_feature)
if model_name == 'CondMLP':
    predictor = CondMLP(seed_dim=seq_len-cond_len,
                        cond_len=cond_len, seq_len=seq_len, hidden_dim=256)


# summary(predictor, [(cond_len, n_feature), (cond_len, noise_dim), (2, 1, hidden_dim)])

# train the model to minimize MSE
# loss_func = loss = nn.MSELoss()
# train_iter(predictor, loss_func, lr=0.0005)

# %%
# Training to minimize utils.w_distance seems quite unstable and fail to converge
# even if pretraining to minimize MSE
# reason unknown

# train_iter(predictor, loss_func=nn.MSELoss(), lr=0.0005, epochs=300)

# def weighted_dist(generated, target):
#     return utils.w_distance(generated.squeeze(-1), target.squeeze(-1))

# train_iter(predictor, loss_func=weighted_dist, lr=0.0005)

# %%

sinkorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)


def sinkhorn_dist(generated, target):
    return sinkorn_loss(generated.squeeze(-1), target.squeeze(-1))


train_iter(predictor, loss_func=sinkhorn_dist, lr=0.005, epochs=300)

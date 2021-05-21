import torch.nn as nn
import torch

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

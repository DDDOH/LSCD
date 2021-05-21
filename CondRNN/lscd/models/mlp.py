import torch.nn as nn
import torch

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

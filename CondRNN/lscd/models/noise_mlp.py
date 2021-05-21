import torch.nn as nn
import torch


class NoisyLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        after_noise = x + torch.randn_like(x)
        after_linear = self.linear(after_noise)
        return self.leaky_relu(after_linear)


class CondNoiseMLP(nn.Module):
    def __init__(self, cond_len, seq_len, hidden_dim=256):
        """Initialize a conditional MLP.

        Args:
            cond_len (int): Length of the condition (q).
            seq_len (int): Length of the sequence (p).
            hidden_dim (int, optional): Size of hidden layer. Defaults to 256.
        """
        super().__init__()
        self.cond_len = cond_len
        self.seq_len = seq_len
        self.layer_1 = NoisyLayer(input_dim=cond_len, output_dim=hidden_dim)
        self.layer_2 = NoisyLayer(input_dim=hidden_dim, output_dim=hidden_dim)
        self.layer_3 = NoisyLayer(input_dim=hidden_dim, output_dim=hidden_dim)
        self.layer_output = nn.Linear(hidden_dim, seq_len - cond_len)
        # main = nn.Sequential(
        #     nn.Linear(cond_len, hidden_dim), nn.LeakyReLU(0.1, True),
        #     nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1, True),
        #     nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1, True),
        #     nn.Linear(hidden_dim, seq_len - cond_len),
        # )
        # self.main = main

    def forward(self, X_q):
        """Generate arrival count in the last p-q time intervals, given the arrival count in the first q time steps.

        Args:
            X_q (torch.Tensor): Arrival count in the first q time steps. A matrix of shape batch_size * q.
            noise (torch.Tensor): Random noise. A matrix of shape batch_size * self.seed_dim

        Returns:
            output (torch.Tensor): Arrival count in the last p - q time steps. A matrix of shape batch_size * (p - q).
        """
        output_1 = self.layer_1(X_q)
        output_2 = self.layer_2(output_1)
        output_3 = self.layer_3(output_2)
        return self.layer_output(output_3)


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


if __name__ == '__main__':
    noise_mlp = CondNoiseMLP(cond_len=3, seq_len=10, hidden_dim=64)
    cond = torch.randn(size=(100, 3))
    output = noise_mlp(cond)

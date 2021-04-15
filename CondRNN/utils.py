import torch
import torch.nn as nn
# import dataset.multivariate_normal
import numpy as np
import scipy.linalg
from torch.autograd import Function


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov

    Reference: https://github.com/pytorch/pytorch/issues/19037

    Args:
        x (torch.Tensor): torch.Tensor of shape (batch_size, seq_len)
        rowvar (bool, optional): [description]. Defaults to False.
        bias (bool, optional): [description]. Defaults to False.
        ddof ([type], optional): [description]. Defaults to None.
        aweights ([type], optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: torch.Tensor of shape (seq_len, seq_len)
    """
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def assert_same_cov(A, w=None):
    c1 = np.cov(A, rowvar=False, aweights=w)
    c2 = cov(torch.tensor(A, dtype=torch.float), aweights=w)
    assert np.linalg.norm(c2.numpy() - c1) < 1e-5


def test_cov():
    a = [1, 2, 3, 4]
    assert_same_cov(a)
    A = [[1, 2], [3, 4]]
    assert_same_cov(A)

    assert_same_cov(a, w=[1, 1, 1, 1])
    assert_same_cov(a, w=[2, 0.5, 3, 1])

    assert_same_cov(A, w=[1, 1])
    assert_same_cov(A, w=[2, 0.5])


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    Reference: https://github.com/steveli/pytorch-sqrtm
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def test_sqrtm():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = (k.t().matmul(k)).requires_grad_()
    test = gradcheck(sqrtm, (pd_mat,))
    print(test)


def w_distance(data_1, data_2):
    """Compute the Wasserstein between two normal distribution.

    Reference: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

    Args:
        data_1 (np.array or torch.Tensor): array of size (batch_size, seq_len)
        data_2 (np.array or torch.Tensor): array of size (batch_size, seq_len)
    """
    if isinstance(data_1, np.ndarray):
        m_1, m_2 = np.mean(data_1, axis=0), np.mean(data_2, axis=0)
        cov_1, cov_2 = np.cov(data_1, rowvar=False), np.cov(
            data_2, rowvar=False)
        sqrtm_cov_1 = scipy.linalg.sqrtm(cov_1).real
        w_dist = np.linalg.norm(m_1 - m_2)**2 + \
            np.trace(cov_1 + cov_2 -
                     2 * scipy.linalg.sqrtm(np.matmul(np.matmul(sqrtm_cov_1,
                                                                cov_2),
                                                      sqrtm_cov_1)).real)
    if isinstance(data_1, torch.Tensor):
        m_1, m_2 = torch.mean(data_1, dim=0).squeeze(
        ), torch.mean(data_2, dim=0).squeeze()

        cov_1, cov_2 = cov(data_1), cov(data_2)

        sqrtm_cov_1 = sqrtm(cov_1)

        w_dist = torch.linalg.norm(m_1 - m_2) ** 2 + \
            torch.trace(cov_1 + cov_2 -
                        2 * sqrtm(torch.mm(torch.mm(sqrtm_cov_1,
                                                    cov_2),
                                           sqrtm_cov_1)))

    return w_dist


def test_w_distance():
    # w_distance on two same dataset should be zero.
    data_np = np.random.normal(size=(10, 50))
    result = w_distance(data_np, data_np)
    np.testing.assert_allclose(result, 0, atol=1e-5)

    # w_distance is symmetric.
    data_1_np, data_2_np = np.random.normal(
        size=(10, 50)), np.random.normal(size=(10, 50))
    result_12 = w_distance(data_1_np, data_2_np)
    result_21 = w_distance(data_2_np, data_1_np)
    np.testing.assert_allclose(result_12, result_21, atol=1e-5)

    for i in range(5):
        data_1_np, data_2_np = np.random.normal(
            size=(10, 50)), np.random.normal(size=(10, 50))
        data_1_torch, data_2_torch = torch.Tensor(
            data_1_np), torch.Tensor(data_2_np)

        np_result = w_distance(data_1_np, data_2_np)
        torch_result = w_distance(data_1_torch, data_2_torch)
        np.testing.assert_allclose(np_result, torch_result, rtol=1e-3,
                                   err_msg='numpy result:{}, torch result:{}'.format(np_result, torch_result))


# def plot_joint():


# def plot_cond(cond, pred_value, real_value):
#     if cond.ndim == 1:
#         cond_len = len(cond)
#     if cond_ndim == 2:
#         cond = cond[1, :]
#         cond_len = len(cond)
#     depend_len = np.shape(pred_value)[1]
#     plt.figure()
#     plt.plot(np.arange(cond_len)+1, cond, c='k', alpha=0.1)
#     plt.plot(np.arange(cond_len, depend_len)+1, pred_value.T, c='r', alpha=0.1)
#     plt.plot(np.arange(cond_len, depend_len)+1, real_value.T, c='g', alpha=0.1)


if __name__ == "__main__":
    test_cov()
    test_sqrtm()
    test_w_distance()

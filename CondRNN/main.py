# %%
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from geomloss import SamplesLoss
import progressbar

# TODO add verbose argument to print what is doing to the process need a progreeebar wrapper
from lscd import utils
from lscd import dataset
from lscd import models
from lscd import metric
from lscd import utils

# TODO
"""
Dataset
    GMM                 [o]
    PGnorta             [o]
    Bikeshare           [ ]
    other dataset       [ ]
Baseline
    KDE                 [ ]
    GMM                 [o]
ML model
    RNN             
    NoisyReinforcedMLP
    MLP
Evaluation metric
    Run through queue                   [ ]
    pairwise plot                       [ ]
    marginal mean                       [o]
    marginal variance                   [o]
    Pierre correlation                  [o]
    Gaussian approximated W-distance    [o]
Others
    Gaussian gradient approximation precision (when lambda is small)
"""
# See how python import self defined modules: https://realpython.com/python-import/


# Get synthetic training set.
COND_LEN = 10
N_SAMPLE = 300

DATA_NAME = 'PGnorta'  # 'PGnorta' or 'multivariate_normal'
# 'CondLSTM' or 'CondMLP' or 'BaselineGMM' or 'CondNoiseMLP'
MODEL_NAME = 'CondNoiseMLP'
if DATA_NAME == 'multivariate_normal':
    SEQ_LEN = 50
    data = dataset.multivariate_normal.MultivariateNormal(seq_len=SEQ_LEN)
if DATA_NAME == 'PGnorta':
    data = dataset.pg_norta.get_PGnorata_from_img()
    seq_len = data.seq_len
training_set = data.sample(n_sample=N_SAMPLE)

dirname = os.path.dirname(__file__)
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
result_time_dir = os.path.join(dirname, 'results', date)
result_dir = os.path.join(dirname, 'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
if not os.path.exists(result_time_dir):
    os.mkdir(result_time_dir)
writer = SummaryWriter(result_time_dir)


# pre processing
SCALE = False
if SCALE:
    scaler = utils.data_sturcture.Scaler(training_set)
    ori_training_set = training_set
    training_set = scaler.transform()

training_set = torch.Tensor(training_set).unsqueeze(-1)
conditions = training_set[:, :COND_LEN, :]
dependents = training_set[:, COND_LEN:, :]

# baseline methods
if MODEL_NAME == 'BaselineGMM':
    cgmm = models.gmm.CGMM(condition=conditions.squeeze(-1),
                           dependent=dependents.squeeze(-1), n_components=5)
    test_condition = conditions[0, :, 0]
    gmm_cond_samples = cgmm.cond_samples(
        new_condition=test_condition, n_sample=1000, verbose=True)
    gmm_cond_samples = np.round(gmm_cond_samples).astype(int)
    real_cond_samples = data.sample_cond(
        X_1q=test_condition, n_sample=200, verbose=True)
    metric.classical.evaluate_joint(
        real_samples=real_cond_samples, fake_samples=gmm_cond_samples, dir_filename='test_gmm.jpg')

if MODEL_NAME == 'BaselineKDE':
    ckde = models.kde.CKDE(condition=conditions.squeeze(-1),
                           dependent=dependents.squeeze(-1))
    test_condition = conditions[0, :, 0]
    kde_cond_samples = ckde.cond_samples(
        new_condition=test_condition, n_sample=1000)
    real_cond_samples = data.sample_cond(X_1q=test_condition, n_sample=200)
    metric.classical.evaluate_joint(
        real_samples=real_cond_samples, fake_samples=kde_cond_samples, dir_filename='test_kde.jpg')


# TODO: Poisson count simulator layer


def train_iter_old(model, loss_func, lr=0.0002, epochs=1000):
    """Train the model with given loss. Deprecated.
    But may contains some code for CondRNN and CondLSTM. So just put it here till updating CondRNN and CondLSTM.


    Args:
        model ([type]): [description]
        loss_func ([type]): Loss function. The first input is sequence generated by model, second input is target.
        lr (float, optional): [description]. Defaults to 0.0002.
        epochs (int, optional): [description]. Defaults to 1000.
    Returns:
        model (torch.Tensor): Trained model.
    """

    optimizer = torch.optim.Adam(model_instance.parameters(), lr)

    # loss_curve = []
    if isinstance(model, models.rnn.CondLSTM):
        h_records = []
        c_records = []
        target = training_set[:, 1:]
    if isinstance(model, models.mlp.CondMLP):
        target = training_set[:, COND_LEN:]

    for i in range(epochs):
        model_instance.zero_grad()
        pred_value = models.mlp.predict_MLP(model_instance, training_set)
        # TODO pred_value = predict_full_RNN(training_set, model_instance)
        loss = loss_func(pred_value, target)
        loss.backward()
        print('[%d/%d] Loss: %.4f' % (i, epochs, loss.item()))
        # loss_curve.append(loss.detach())
        optimizer.step()

        if model == 'CondLSTM':
            h_records.append(model_instance.h_0.clone().detach().squeeze())
            c_records.append(model_instance.c_0.clone().detach().squeeze())

        if i % 10 == 0:
            # Plot predicted value vs target value.
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.plot(pred_value.squeeze(-1).detach().T, c='r', alpha=0.02)
            plt.plot(target.squeeze(-1).T, c='g', alpha=0.02)
            plt.subplot(122)
            condition = training_set[:, :COND_LEN]
            true_dependent = training_set[:, COND_LEN:]

            # TODO: modify old only LSTM code
            # TODO pred_on_condition = predict_condition(condition, model_instance)
            # TODO plt.plot(pred_on_condition.squeeze(-1).detach().T,
            #  c='r', alpha=0.02)
            fake_dependent = models.mlp.predict_MLP(
                model_instance, training_set).detach()

            plt.plot(fake_dependent.squeeze(-1).T, c='r', alpha=0.1)
            plt.plot(true_dependent.squeeze(-1).T, c='g', alpha=0.1)
            filename = os.path.join(result_dir, 'prediction_step_%d.jpg' % (i))
            plt.savefig(filename)
            plt.close()

            if isinstance(model, models.rnn.CondLSTM):
                plt.figure(figsize=(20, 5))
                # plt.pcolormesh(np.stack(h_records, axis=0))
                plt.subplot(121)
                plt.plot(np.stack(h_records, axis=0))
                plt.subplot(122)
                plt.plot(np.stack(c_records, axis=0))
                filename = os.path.join(result_dir, 'h_c_records.jpg')
                plt.savefig(filename)
                plt.close()

            # fake_statistic = utils.plot_mean_var_cov(
            #     fake_dependent.squeeze(-1), dpi=45)
            # writer.add_image("fake", fake_statistic, i)

        writer.add_scalar('Loss curve',
                          loss.item(), i)

        writer.flush()

    return model


if MODEL_NAME == 'CondLSTM':
    # Config for CondLSTM
    noise_dim = 2
    hidden_dim = 256
    n_feature = 1
    # may also use embedded time step as input feature
    in_feature = noise_dim + n_feature
    n_layers = 1
    out_feature = n_feature
    # since the number of data in training set is small, use the whole training set in each iteration, temporarily
    batch_size = N_SAMPLE
    model_instance = models.rnn.CondLSTM(
        in_feature=in_feature, out_feature=out_feature)
if MODEL_NAME == 'CondMLP':
    model_instance = models.mlp.CondMLP(seed_dim=seq_len - COND_LEN,
                                        cond_len=COND_LEN, seq_len=seq_len, hidden_dim=256)
if MODEL_NAME == 'CondNoiseMLP':
    model_instance = models.noise_mlp.CondNoiseMLP(
        cond_len=COND_LEN, seq_len=seq_len, hidden_dim=64)

# ********** Training to minimize MSE ********** #
# loss_func = loss = nn.MSELoss()
# train_iter_old(model_instance, loss_func, lr=0.0005)
# ********** Training to minimize MSE ********** #

# ********** Training to minimize utils.w_distance ********** #
#
# seems quite unstable and fail to converge
# even if pretraining to minimize MSE
# reason unknown

# train_iter_old(model_instance, loss_func=nn.MSELoss(), lr=0.0005, epochs=300)

# def weighted_dist(generated, target):
#     return utils.w_distance(generated.squeeze(-1), target.squeeze(-1))

# train_iter_old(model_instance, loss_func=weighted_dist, lr=0.0005)
# ********** Training to minimize utils.w_distance ********** #


# ********** Training to minimize sinkhorn distance ********** #
sinkorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5)


def train_iter(model_instance, loss_func, lr, epochs):
    # make comparision on several conditions
    n_condition_to_compare = 5
    conditions_to_compare = conditions.squeeze(
        -1)[:n_condition_to_compare, :]
    real_cond_dep = utils.data_structure.CondDep()

    for i in range(n_condition_to_compare):
        real_dep = data.sample_cond(X_1q=conditions_to_compare[i, :],
                                    n_sample=50, verbose=True)
        real_cond_dep.add_cond_dep_pair(
            condition=conditions[i, :], dependent=real_dep)

    optimizer = torch.optim.Adam(model_instance.parameters(), lr)
    for i in progressbar.progressbar(range(epochs), redirect_stdout=True):
        model_instance.zero_grad()
        fake_x_qp = model_instance(conditions.squeeze(-1))
        fake_x = torch.cat([conditions.squeeze(-1), fake_x_qp], dim=1)
        loss = loss_func(training_set.squeeze(-1), fake_x)
        loss.backward()
        print('[%d/%d] Loss: %.4f' % (i, epochs, loss.item()))
        # loss_curve.append(loss.detach())
        optimizer.step()

        # evaluate_joint
        if i % 10 == 0:
            # joint distribution
            metric.classical.evaluate_joint(real_samples=training_set.squeeze(-1),
                                            fake_samples=fake_x,
                                            dir_filename='CondRNN/results/NoiseMLP{}.jpg'.format(i))
            # conditional distribution
            # TODO add evaluate_joint for multiple conditions
            metric.classical.evaluate_cond(
                real_cond_dep, fake_cond_dep=real_cond_dep)


def sinkhorn_dist(generated, target):
    return sinkorn_loss(generated, target)


loss_func = nn.MSELoss()
train_iter(model_instance, loss_func=loss_func, lr=0.005, epochs=300)
# ********** Training to minimize sinkhorn distance ********** #

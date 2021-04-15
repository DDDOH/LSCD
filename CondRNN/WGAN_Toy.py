
# coding: utf-8

# In[1]:


# %reset -f
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn
import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import pearsonr
from scipy.stats import poisson
import math
import time
import random
import progressbar

import os, sys, getopt
sys.path.append(os.getcwd())

import torch
import torch.autograd as autograd
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
use_cuda = bool(torch.cuda.device_count())
import shutil


# In[2]:


if not use_cuda:
#     get_ipython().run_line_magic('reset', '-f')
    get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


if use_cuda:
    opts, args = getopt.getopt(sys.argv[1:],"",["intensity=","sample=","condition=",
                                                "output=","gradient_penalty=","dropout=",
                                                "generator_penalty=","note=","dim=",
                                                "data=","critic=","result_dir=","c_gan=","n_iter="])
    #     except getopt.GetoptError:
    #         print('test.py -i <inputfile> -o <outputfile>')
    #         sys.exit(2)
    for opt, arg in opts:
        if opt in ("--output"): # count intensity or b
            OUTPUT = str(arg)
        elif opt in ("--gradient_penalty"): # coefficient for gradient penalty
            LAMBDA = float(arg)
        elif opt in ("--dropout"):
            DROPOUT = (int(arg) == 1)
        elif opt in ("--generator_penalty"):
            LAMBDA_GEN = float(arg)
        elif opt in ("--note"):
            NOTE = str(arg)
        elif opt in ("--dim"):
            DIM = int(arg)
        elif opt in ("--data"):
            DATA_NAME = str(arg)
        elif opt in ('--critic'):
            CRITIC_ITERS = int(arg)
        elif opt in ('--result_dir'):
            RESULT_DIR = 'tmp/' + str(arg)  # the name of the folder to write down result and add para   
        elif opt in ('--c_gan'): # if 1 then will set C_GAN to 1 if provide with "N_known_6" data.
            C_GAN = (int(arg) == 1)
        elif opt in ('--n_iter'): # if 1 then will set C_GAN to 1 if provide with "N_known_6" data.
            ITERS = int(arg)
else:
#     OUTPUT = 'intensity'
#     LAMBDA = 10#10#.1  # Smaller lambda seems to help for toy tasks specifically
#     DROPOUT = (int(0) == 1)
#     LAMBDA_GEN = 0
#     NOTE = ''
#     CRITIC_ITERS = 5 # 5 originally  # How many critic iterations per generator iteration
#     DIM = 512  # Model dimensionality
# #     DATA_NAME = 'N_sample: 300, N_known: 6, Magnitude:300, DI:1.71, '
#     DATA_NAME = 'N_sample_2000_N_known_6_Magnitude_300_DI_8.2_'
#     RESULT_DIR = '10_14_2000samples'
#     RESULT_DIR = 'tmp/' + str(RESULT_DIR)    
#     C_GAN = (int(1) == 0)

    OUTPUT = 'intensity'
    LAMBDA = 10#10#.1  # Smaller lambda seems to help for toy tasks specifically
    DROPOUT = (int(0) == 1)
    LAMBDA_GEN = 0
    NOTE = ''
    CRITIC_ITERS = 5 # 5 originally  # How many critic iterations per generator iteration
    DIM = 512  # Model dimensionality
    DATA_NAME = 'cir_multi_server'#'TRAIN_SIZE_300_P_KNOWN_0_Magnitude_300_DI_8.84_P_22'#
    RESULT_DIR = 'cir_multi_server'#'TRAIN_SIZE_300_P_KNOWN_0_Magnitude_300_DI_8.84_P_22'#'bikeshare'
    RESULT_DIR = 'tmp/' + str(RESULT_DIR) 
    C_GAN = (int(1) == 1)
    ITERS = 100

    
FILE_DIR = 'tmp/data/' + DATA_NAME + '/'
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
    shutil.copy2('WGAN_Toy.py', RESULT_DIR)
    shutil.copy2('test.bash', RESULT_DIR)
    shutil.copy2(FILE_DIR + 'training_set.npy', RESULT_DIR)
    shutil.copy2(FILE_DIR + 'test_set.npy', RESULT_DIR)
    if C_GAN:
        shutil.copy2(FILE_DIR + 'count_known_set.npy', RESULT_DIR)
        shutil.copy2(FILE_DIR + 'c_test_set_set.npy', RESULT_DIR)
    
    


# In[4]:


RESULT_DIR


# In[5]:


class Normalize():
    def __init__(self, ori_data, target_mean, target_std):
        self.ori_mean = np.mean(ori_data)
        self.ori_std = np.std(ori_data)
        self.ori_data = ori_data.copy()
        
        def rescale_func(data):
            data = data.copy()
            zero_mean_one_std_data = (data - self.ori_mean)/self.ori_std
            rescaled_data = zero_mean_one_std_data * target_std + target_mean
            return rescaled_data.copy()
        
        self.rescale_func = rescale_func
        
        self.rescaled_mean = target_mean
        self.rescaled_std = target_std
        self.rescaled_ori = self.rescale_func(self.ori_data)
        
    def get_rescaled_on_ori(self):
        # given target_mean, target_std, get the rescaled ori_data
        return self.rescaled_ori.copy()
    
    def get_rescale_on_new(self, new_data):
        new_data = new_data.copy()
        rescaled_new = self.rescale_func(new_data)
        return rescaled_new.copy()
    
    def rescale_on_new_para(self):
        rescale_para = {}
        rescale_para['multipler'] = 1/self.ori_std*self.rescaled_std
        rescale_para['adder'] = - self.ori_mean/self.ori_std*self.rescaled_std + self.rescaled_mean
        return rescale_para
    
    def get_ori(self):
        return self.ori_data.copy()


# In[7]:


training_set = np.load(FILE_DIR + 'training_set.npy')

# 如果用的是真实数据集/CIR，参数就自己写进去；
# 如果用的是Pierre model，参数已经存在 para.npy 文件里了
if 'call_center' in DATA_NAME:
    C_GAN = False
    para = {}
    para['P_KNOWN'] = 0
    para['P'] = 16
    para['TRAIN_SIZE'] = len(training_set)
    para['MAGNITUDE'] = np.mean(training_set)
    para['DI_VAL'] = np.mean(np.var(training_set, axis=0)/np.mean(training_set, axis=0))
    para['NOTE'] = ''
#     test_set = np.load(FILE_DIR + 'test_set.npy')
    

elif 'cir' in DATA_NAME:
    C_GAN = False
    para = {}
    para['P_KNOWN'] = 0
    para['P'] = 22
    para['TRAIN_SIZE'] = len(training_set)
    para['MAGNITUDE'] = np.mean(training_set)
    para['DI_VAL'] = np.mean(np.var(training_set, axis=0)/np.mean(training_set, axis=0))
    para['NOTE'] = ''
    para["GRADIENT_PENALTY"] = LAMBDA
    para["DROPOUT"] = DROPOUT
    para["GENERATOR_PENALTY"] = LAMBDA_GEN
    para["DATA_NAME"] = DATA_NAME
    para["CRITIC_ITERS"] = CRITIC_ITERS
    para["C_GAN"] = C_GAN
    
    
elif DATA_NAME == 'bikeshare':
    C_GAN = True
    para = {}
    para['P_KNOWN'] = 2 # 已知前几个维度的 作为condition
    para['GP_DIM'] = [0] # 计算 generator penalty 的哪几个维度
    para['P'] = 26
    para['TRAIN_SIZE'] = len(training_set)
    para['MAGNITUDE'] = np.mean(training_set)
    para['DI_VAL'] = np.mean(np.var(training_set, axis=0)/np.mean(training_set, axis=0))
    para['NOTE'] = ''
    
    temp_normalize = Normalize(ori_data=training_set[:,0],target_mean=300, target_std=100)
    weather_normalize = Normalize(ori_data=training_set[:,1],target_mean=300, target_std=100)
    
    training_set[:,0] = temp_normalize.get_rescaled_on_ori()
    training_set[:,1] = weather_normalize.get_rescaled_on_ori()
    
elif 'dswgan_real' in DATA_NAME: 
    C_GAN = False
    para = {}
    para['P_KNOWN'] = 0 # 已知前几个维度的 作为condition
    para['GP_DIM'] = [0] # 计算 generator penalty 的哪几个维度
    para['P'] = 16
    para['TRAIN_SIZE'] = len(training_set)
    para['MAGNITUDE'] = np.mean(training_set)
    para['DI_VAL'] = np.mean(np.var(training_set, axis=0)/np.mean(training_set, axis=0))
    para['NOTE'] = ''

else:
    para = np.load(FILE_DIR + 'para.npy', allow_pickle=True).item()
#     test_set = np.load(FILE_DIR + 'test_set.npy')
    if para['P_KNOWN'] > 0 and C_GAN:
        para['GP_DIM'] = np.arange(para['P_KNOWN'])
#         print(para['GP_DIM'])
#         c_test_set_set = np.load(FILE_DIR + 'c_test_set_set.npy', allow_pickle=True).item()
#         count_known_set = np.load(FILE_DIR + 'count_known_set.npy', allow_pickle=True).item()
    else:
        C_GAN = False
        para['P_KNOWN'] = 0
        
para["DIM"] = DIM
para["OUTPUT"] = OUTPUT
para["C_GAN"] = C_GAN
para["DATA_NAME"] = DATA_NAME

KNOWN_MASK = np.array([x < para['P_KNOWN'] for x in range(para['P'])])
UNKNOWN_MASK = np.array([not x for x in KNOWN_MASK])
SEED_DIM = np.sum(UNKNOWN_MASK)
# BATCH_SIZE = min(256, int(para['TRAIN_SIZE']/2))  # Batch size
BATCH_SIZE = 256
use_cuda = bool(torch.cuda.device_count())
PARA_CMT = "TRAIN_SIZE: {}, P_KNOWN: {}, Magnitude:{}, DI:{}, OUTPUT:{}, GradientPenalty:{}, GeneratorPenalty:{}, DROPOUT:{}, DIM:{}, {}".format(para['TRAIN_SIZE'], 
                                                        para['P_KNOWN'], 
                                                        para['MAGNITUDE'],
                                                        para['DI_VAL'],
                                                        OUTPUT,
                                                        LAMBDA,
                                                        LAMBDA_GEN,
                                                        DROPOUT,
                                                        DIM,
                                                        para['NOTE'])


# In[8]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        main = nn.Sequential(
            nn.Linear(SEED_DIM + sum(KNOWN_MASK), DIM), nn.LeakyReLU(0.1, True),
            nn.Linear(DIM, DIM), nn.LeakyReLU(0.1, True),
            nn.Linear(DIM, DIM), nn.LeakyReLU(0.1, True),
            nn.Linear(DIM, para['P'] - sum(KNOWN_MASK)), # generate the intensity at the unknown period
        )
        self.main = main
    def forward(self, noise):
        output = self.main(noise)
        return output

if not DROPOUT:
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            main = nn.Sequential(
                nn.Linear(para['P'], DIM), nn.LeakyReLU(0.1),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1),
                nn.Linear(DIM, 1),
            )
            self.main = main
        def forward(self, inputs):
            output = self.main(inputs)
            return output.view(-1)
else:
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            main = nn.Sequential(
                nn.Linear(para['P'], DIM), nn.LeakyReLU(0.1), nn.Dropout(0.1, True),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1), nn.Dropout(0.2, True),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1), nn.Dropout(0.4, True),
                nn.Linear(DIM, 1),
            )
            self.main = main
        def forward(self, inputs):
            output = self.main(inputs)
            return output.view(-1)


# In[85]:



def inf_train_iter(data_set):
    while True:
        select_id = np.random.choice(np.arange(len(data_set)),BATCH_SIZE,replace=True)
        yield data_set[select_id,:]


# In[9]:


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
#     interpolates = autograd.Variable(interpolates, requires_grad=True)
    # 9.15: changed here, to avoid all the interpolates lies on the condition in the training set
    
    # real data [1,2,3, xxxx] [4,3,5, xxx] [3,5,6,xxxx]
    # fake data  [4,3,5, xxx] [3,5,6,xxxx] [1,2,3, xxxx]
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data[torch.randperm(BATCH_SIZE),:])

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    
#     if iteration % 10 == 0:
#         print('                                 gradient_penalty: ', gradient_penalty)
        
    return gradient_penalty 

# ==================Definition End======================


# In[10]:


def calc_generator_penalty(netG, real_data, fake_data):
    assert para['P_KNOWN'] >= 1
    if LAMBDA_GEN == 0:
        return 0
    
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data[torch.randperm(BATCH_SIZE),:])

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    noise = torch.randn(BATCH_SIZE, SEED_DIM)
    noise = noise.cuda() if use_cuda else noise

    noisev = autograd.Variable(noise)
    gen_input = torch.cat([noisev, interpolates[:, KNOWN_MASK]], 1)

    # interpolates[:para['P_KNOWN'], :]
    gene_interpolates = netG(gen_input)

    gradients = autograd.grad(outputs=gene_interpolates, inputs=gen_input,
                              grad_outputs=torch.ones(gene_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  gene_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0][:,-para['P_KNOWN']:][:,para['GP_DIM']]

    
    generator_penalty = ((gradients.norm(2, dim=1)) ** 2).mean() * LAMBDA_GEN
    
#     if iteration % 10 == 0:
#         print('generator_penalty: ', generator_penalty)

    return generator_penalty
# ==================Definition End======================


# In[11]:


real_count_iter = inf_train_iter(training_set)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
#     print(classname)
    if classname.find('Linear') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#         m.bias.data.fill_(1)
        # initial parameter for count based gan, for intensity based gan also works well
        if OUTPUT == 'intensity' or OUTPUT == 'count':
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(3)
        if OUTPUT == 'b':
            m.weight.data.normal_(0.0, 0.01)
            m.bias.data.fill_(0.01)

    # useless
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(1)


# In[12]:


netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

one = torch.tensor(1, dtype=torch.float)
# one = torch.FloatTensor([1])
# torch.tensor(1, dtype=torch.float) 
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()


# In[13]:


# In[96]:


lr_final = 1e-6
lr_initial = 1e-4
gamma_G = (lr_final/lr_initial)**(1/ITERS)
gamma_D = (lr_final/lr_initial)**(1/ITERS/CRITIC_ITERS)

optimizerD = optim.Adam(netD.parameters(), lr=lr_initial, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=lr_initial, betas=(0.5, 0.9))
optimizerD_lrdecay = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=gamma_D, last_epoch=-1)
optimizerG_lrdecay = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=gamma_G, last_epoch=-1)


# In[14]:


# _count = real_count_iter.__next__()
# real_count = torch.Tensor(_count)
# # use naive random noise
# noise = torch.randn(BATCH_SIZE, SEED_DIM)

# if use_cuda:
#     noise = noise.cuda()
#     real_count = real_count.cuda()
# with torch.no_grad(): # totally freeze netG
#     noisev = autograd.Variable(noise)
#     gen_input = torch.cat([noisev, real_count[:, KNOWN_MASK]], 1)

# if OUTPUT == 'intensity':
#     intensity_pred = netG(gen_input)
#     sign = torch.sign(intensity_pred)
#     intensity_pred = intensity_pred * sign
#     count_fake = autograd.Variable(torch.poisson(intensity_pred.float().data))
#     count_fake = count_fake * sign

# if OUTPUT == 'count':
#     count_fake = netG(gen_input)
    
# plt.pcolormesh(count_fake.detach().numpy())
# plt.colorbar()


# In[15]:


for iteration in progressbar.progressbar(range(ITERS)):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):#CRITIC_ITERS):
        # real count for getting condition
        _count = real_count_iter.__next__()
        real_count = torch.Tensor(_count)
        
        if use_cuda:
            real_count = real_count.cuda()
        real_count_v = autograd.Variable(real_count)

        # train with real
        D_real = netD(real_count_v)
        D_real = D_real.mean()
# #         D_real.backward(mone)

        # train with fake
        # use a new batch of data to penalize discriminator
        _count = real_count_iter.__next__()
        real_count = torch.Tensor(_count)
        # use naive random noise
        noise = torch.randn(BATCH_SIZE, SEED_DIM)
        
        if use_cuda:
            noise = noise.cuda()
            real_count = real_count.cuda()
        with torch.no_grad(): # totally freeze netG
            noisev = autograd.Variable(noise)
            gen_input = torch.cat([noisev, real_count[:, KNOWN_MASK]], 1)
        
        if OUTPUT == 'intensity':
            intensity_pred = netG(gen_input)
            sign = torch.sign(intensity_pred)
            intensity_pred = intensity_pred * sign
            count_fake = autograd.Variable(torch.poisson(intensity_pred.float().data))
            count_fake = count_fake * sign
            
        if OUTPUT == 'count':
            count_fake = netG(gen_input)
            
        all_count = real_count
        all_count[:, UNKNOWN_MASK] = count_fake
        all_count = autograd.Variable(all_count)
        
        D_fake = netD(all_count)
        D_fake = D_fake.mean()
#         D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_count, all_count.data)
#         gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        
        netD.zero_grad()
        D_cost.backward()
        optimizerD.step()
        optimizerD_lrdecay.step()


    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    
    _count = real_count_iter.__next__()
    real_count = torch.Tensor(_count)
    
    if use_cuda:
        real_count = real_count.cuda()
    real_count = autograd.Variable(real_count)

    noise = torch.randn(BATCH_SIZE, SEED_DIM)
    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise)
    # generator 先给 noise，后给 condition
    gen_input = torch.cat([noisev, real_count[:, KNOWN_MASK]], 1)
    
    if OUTPUT == 'intensity':
        intensity_pred = netG(gen_input)
        intensity_val = intensity_pred.cpu().data.numpy() if use_cuda else intensity_pred.data.numpy()
        sign = np.sign(intensity_val)
        intensity_val = intensity_val * sign
        count_val = np.random.poisson(intensity_val)
        count_val = count_val * sign
        
        w_mid = 1 + (count_val - intensity_val)/(2 * intensity_val) 
        w_mid = np.maximum(w_mid, 0.5)
        w_mid = np.minimum(w_mid, 1.5)
        b_mid = count_val - w_mid * intensity_val

        w_mid_tensor = autograd.Variable(torch.Tensor([w_mid]))
        b_mid_tensor = autograd.Variable(torch.Tensor([b_mid]))
                             
        if use_cuda:
            w_mid_tensor = w_mid_tensor.cuda()
            b_mid_tensor = b_mid_tensor.cuda()
            
        pred_fake = intensity_pred * w_mid_tensor + b_mid_tensor
        
    if OUTPUT == 'count':
        pred_fake = netG(gen_input)

    all_fake = real_count
    all_fake[:, UNKNOWN_MASK] = pred_fake
    
    G = netD(all_fake)
    G = G.mean()

    if C_GAN:
        generator_penalty = calc_generator_penalty(netG, real_count, all_fake)
#         generator_penalty.backward()
        G_cost = -G + generator_penalty
    else:
        G_cost = -G
    
    netG.zero_grad()
    G_cost.backward()
    optimizerG.step()
    optimizerG_lrdecay.step()
    
#     count_compare()

#     if iteration % 10 == 0:
#         plt.figure()
#         count_known = (np.mean(training_set,axis=0) + 0.5 * np.std(training_set,axis=0))[:para['P_KNOWN']]
#         count_known = np.tile(count_known, (1000,1))
#         noise = torch.randn(1000, SEED_DIM)
#         noise = noise.cuda() if use_cuda else noise
#         output = netG(gen_input)
#         output = output.data.numpy()
#         plt.figure(figsize=(4,4))
#         plt.scatter(output[:,0],output[:,1],s=1)
#         with torch.no_grad():
#             count_known = torch.Tensor(count_known)
#             noisev = autograd.Variable(noise)
#             gen_input = torch.cat([noise, count_known], 1)

#         if OUTPUT == 'intensity':
#             intensity_pred = netG(gen_input)
#             intensity_val = intensity_pred.cpu().data.numpy() if use_cuda else intensity_pred.data.numpy()
#             sign = np.sign(intensity_val)
#             intensity_val = intensity_val * sign
#             if DATA_NAME == 'bikesharing' or 'call_center' in DATA_NAME:
#                 b_val = intensity_val / np.mean(training_set)
#                 intensity_val = b_val * np.mean(test_set)
#             pred_fake = np.random.poisson(intensity_val) * sign

#         if OUTPUT == 'count':
#             pred_fake = netG(gen_input)
#             pred_fake = pred_fake.cpu().data.numpy() if use_cuda else pred_fake.data.numpy()
# #         plt.plot(np.var(pred_fake,axis=0))

# torch.save({'state_dict': netG.state_dict()}, FILE_DIR + '/netG.pth.tar')
# torch.save({'state_dict': netD.state_dict()}, FILE_DIR + '/netD.pth.tar')


# In[16]:


TIME_STR = str(int(time.time()*1000000))
result_path = '{}/{}'.format(RESULT_DIR,TIME_STR)
# result_path = RESULT_DIR
os.mkdir(result_path)
torch.save(netG.state_dict(), result_path + '/netG.pt')
    
if DATA_NAME == 'bikeshare':
    training_set[:,0] = temp_normalize.get_ori()
    training_set[:,1] = weather_normalize.get_ori()
    
    np.save(result_path + '/training_set.npy', training_set)
    np.save(result_path + '/temp_rescale_para.npy', temp_normalize.rescale_on_new_para())
    np.save(result_path + '/weather_rescale_para.npy', weather_normalize.rescale_on_new_para())


# In[17]:


if 'call_center' in DATA_NAME:
    shutil.copy2(FILE_DIR + 'test_set.npy', result_path)
    np.save(result_path + '/training_set.npy', training_set)
    np.save(result_path + '/para.npy', para)  


# In[18]:


if DATA_NAME != 'bikeshare' and 'call_center' not in DATA_NAME:
    np.save(result_path + '/training_set.npy', training_set)
    np.save(result_path + '/para.npy', para)  
#     with open(result_path + '/para.npy', 'wb') as f:
#         np.save(f, para)


# In[19]:


# ####### compare all distribution ########################################################
# # this real count is for getting condition requirement
# if DATA_NAME != 'bikeshare' and DATA_NAME != 'cir':
#     if para['FAKE_SIZE'] >= len(training_set):
#         # if we need sample one data point in training set multiple times
#         known_count = np.concatenate([training_set[:,:para['P_KNOWN']]] * np.floor(para['FAKE_SIZE']/len(training_set)).astype(int))
#         known_count = np.concatenate([training_set[np.random.choice(len(training_set),
#                      size=para['FAKE_SIZE'] - np.floor(para['FAKE_SIZE']/len(training_set)).astype(int)*len(training_set),
#                      replace=False),:para['P_KNOWN']],
#                                   known_count])

#     else:
#         known_count = training_set[np.random.choice(len(training_set), size=para['FAKE_SIZE'], replace=False),:para['P_KNOWN']]

#     known_count = torch.Tensor(known_count)
#     noise = torch.randn(para['FAKE_SIZE'], SEED_DIM)
#     noise = noise.cuda() if use_cuda else noise
#     known_count = known_count.cuda() if use_cuda else known_count
#     with torch.no_grad():
#         noisev = autograd.Variable(noise)
#         gen_input = torch.cat([noise, known_count], 1)

#     if OUTPUT == 'intensity':
#         intensity_pred = netG(gen_input)
#         intensity_val = intensity_pred.cpu().data.numpy() if use_cuda else intensity_pred.data.numpy()
#         sign = np.sign(intensity_val)
#         intensity_val = intensity_val * sign
#         if DATA_NAME == 'bikesharing' or 'call_center' in DATA_NAME:
#             b_val = intensity_val / np.mean(training_set)
#             intensity_val = b_val * np.mean(test_set)
#         pred_fake = np.random.poisson(intensity_val) * sign

#     if OUTPUT == 'count':
#         pred_fake = netG(gen_input)
#         pred_fake = pred_fake.cpu().data.numpy() if use_cuda else pred_fake.data.numpy()

#     known_count = known_count.cpu().data.numpy() if use_cuda else known_count.data.numpy()
#     fake_count = np.concatenate([known_count, pred_fake], axis=1)

#     ######## compare only condition distribution ##################################################
#     if C_GAN:
#         c_all_set_set = {}
#         n_sample = para['FAKE_SIZE']
#         for key in count_known_set:
#             count_known = count_known_set[key]
#             c_test_set = c_test_set_set[key]
#             count_known_v = torch.Tensor(count_known).repeat((n_sample,1))
#             # fill unknown period with cgan
#             noise = torch.randn(n_sample, SEED_DIM)
#             if use_cuda:
#                 noise = noise.cuda()
#                 count_known_v = count_known_v.cuda()
#             with torch.no_grad(): # totally freeze netG
#                 noisev = autograd.Variable(noise)
#                 gen_input = torch.cat([noisev, count_known_v], 1)


#             if OUTPUT == 'intensity':
#                 if use_cuda:
#                     intensity_pred = netG(gen_input).cuda()
#                 else:
#                     intensity_pred = netG(gen_input)
#                 sign = torch.sign(intensity_pred)
#                 intensity_pred = intensity_pred * sign
#                 count_fake = autograd.Variable(torch.poisson(intensity_pred).float().data)
#                 count_fake = count_fake * sign

#             if OUTPUT == 'count':
#                 if use_cuda:
#                     count_fake = netG(gen_input).cuda()
#                 else:
#                     count_fake = netG(gen_input)
#                 count_fake = autograd.Variable(count_fake.data)

#             if use_cuda:
#                 cgan_all_count = torch.cat([count_known_v, count_fake], 1).detach().cpu().numpy()
#             else:
#                 cgan_all_count = torch.cat([count_known_v, count_fake], 1).data.numpy()
#             c_all_set_set[key] = cgan_all_count

#         ##### save model
#         #         torch.save({
#         #                     'iteration': iteration,
#         #                     'model_state_dict': netG.state_dict(),
#         # #                     'loss': loss,
#         #                     }, FILE_DIR + '/netG_{}.pth.tar'.format(iteration))

#         #         torch.save({
#         #             'iteration': iteration,
#         #             'model_state_dict': netD.state_dict(),
#         #             }, FILE_DIR + '/netD_{}.pth.tar'.format(iteration))


#     # ##### record D cost G cost and W-distance
#     # new_D_cost = D_cost.cpu().data.numpy() if use_cuda else D_cost.data.numpy()
#     # new_G_cost = G_cost.cpu().data.numpy() if use_cuda else G_cost.data.numpy()
#     # new_W_distance_ls = Wasserstein_D.cpu().data.numpy() if use_cuda else Wasserstein_D.data.numpy()
#     # D_cost_ls.append(new_D_cost)
#     # G_cost_ls.append(new_G_cost)
#     # W_distance_ls.append(new_W_distance_ls)
    
    
    
#     TIME_STR = str(int(time.time()*1000000))
#     result_path = '{}/{}'.format(RESULT_DIR,TIME_STR)
#     os.mkdir(result_path)

#     # gan 生成 overall distribution 的分布
#     with open(result_path + '/fake_count.npy', 'wb') as f:
#         np.save(f, fake_count)

#     # 训练 gan 用的 training set
#     with open(result_path + '/training_set.npy', 'wb') as f:
#         np.save(f, training_set)

#     # 用来表示真实分布的 test set
#     if DATA_NAME != 'cir':
#         with open(result_path + '/test_set.npy', 'wb') as f:
#             np.save(f, test_set)

#     if C_GAN:
#         # gan 补全的 conditional distribution
#         with open(result_path + '/c_all_set_set.npy', 'wb') as f:
#             np.save(f, c_all_set_set)

#         # MCMC 得到的真实的 conditional distribution
#         with open(result_path + '/c_test_set_set.npy', 'wb') as f:
#             np.save(f, c_test_set_set)
            


# In[20]:


df_file_name = RESULT_DIR + "/model_para.csv"
df_res = pd.DataFrame({"MAGNITUDE": para['MAGNITUDE'],
                       "TRAIN_SIZE": para['TRAIN_SIZE'],
                       "P_KNOWN": para['P_KNOWN'],
                       "NOTE": para['NOTE'],
                       #"TEST_SIZE": para['TEST_SIZE'],
                       "P": para['P'],
                       "DI_VAL": para['DI_VAL'],
                       "OUTPUT": OUTPUT,
                       "GRADIENT_PENALTY": LAMBDA,
                       "DROPOUT": DROPOUT,
                       "GENERATOR_PENALTY": LAMBDA_GEN,
                       "DIM": DIM,
                       "DATA_NAME": DATA_NAME,
                       "CRITIC_ITERS": CRITIC_ITERS,
                       "C_GAN": C_GAN}
                      ,index=[TIME_STR])
df_res.reset_index(inplace=True)
if os.path.isfile(df_file_name):
    df = pd.read_csv(df_file_name)
    df = df.append(df_res)
    df.to_csv(df_file_name, index=False)
else:
    df_res.to_csv(df_file_name, index=False)


# In[21]:


print('saved successfully')


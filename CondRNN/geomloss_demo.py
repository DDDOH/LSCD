#%%
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from geomloss import SamplesLoss
import utils
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# N, M = (100, 100) if not use_cuda else (50000, 50000)
N, M = (100, 100)

dim = 10

X_i = torch.randn(N, dim)
Y_j = torch.randn(M, dim)

# %%from geomloss import SamplesLoss

# Compute the Wasserstein-2 distance between our samples,
# with a small blur radius and a conservative value of the
# scaling "decay" coefficient (.8 is pretty close to 1):
# Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
Loss = SamplesLoss("sinkhorn", p=2, blur=1e-10, scaling=0.9)

start = time.time()
Wass_xy = Loss(X_i, Y_j)
if use_cuda:
    torch.cuda.synchronize()
end = time.time()

w_dist_by_formula = utils.w_distance(X_i, Y_j)
print(w_dist_by_formula)

print(
    "Wasserstein distance: {:.3f}, computed in {:.3f}s.".format(
        Wass_xy.item(), end - start
    )
)
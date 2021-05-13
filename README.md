# LSCD

## Progress

Dataset

- [x] GMM
- [x] PGnorta
- [ ] Bikeshare
- [ ] other dataset

Baseline

- [x] KDE
- [x] GMM

Generator ML model

- [ ] RNN
  - [ ] Attention layer
- [ ] Implement the Poisson count simulator as a PyTorch layer
- [ ] NoisyReinforcedMLP
- [x] MLP
- [ ] Generator penalty

Discriminator ML model

- [x] Geoless (compute and minimize W-distance directly)
- [x] MLP

Evaluation metric

- [ ] Run through queue
- [ ] pairwise plot
- [x] marginal mean
- [x] marginal variance
- [x] Pierre correlation
- [x] Gaussian approximated W-distance

Others

- [ ] Gaussian gradient approximation precision (when lambda is small)

## Timeline

The tasks are listed in order

| Task                                                         | Approximated time needed                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Finalize the C version Run through queue code                | 1 day                                                        |
| Link the C version Run through queue code to python          | 1 day                                                        |
| Implement NoisyReinforcedMLP                                 | 2 days                                                       |
| Check the old generator penalty code                         | 0.5 day                                                      |
| Compare the performance of (MLP + Geoless) (NoisyReinforcedMLP + Geoless) | Writing the code is fast, but there’s a lot of uncertainty in the performance of these model. |
|                                                              |                                                              |


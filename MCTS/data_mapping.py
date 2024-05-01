import numpy as np
import pandas as pd

obs = pd.read_csv('data/obs.csv',usecols=[1,2])
data = obs.values


class Data_mapping:
     
    def __init__(self, N, data):

        # N=20 # resolution size 1000000

        # chunk map into pieces
        n_points_per_bin = int(np.ceil(len(data) / N))
        # obs_sorted = np.sort(data, axis=0)
        obs_sorted = np.sort(data, axis=0)
        thresholds = obs_sorted[::n_points_per_bin, :]


        # add boundary to the map
        maxs = data.max(axis=0, keepdims=True)
        self.thresholds = np.concatenate([thresholds, maxs], axis=0)
        self.N = N

    def largest_nonzero_index(x, dim):
        N = x.shape[dim]
        arange = np.arange(N) + 1

        for i in range(dim):
            arange = np.expand_dims(arange, axis=0)
        for i in range(dim+1, x.ndim):
            arange = np.expand_dims(arange, axis=-1)

        inds = np.argmax(x * arange, axis=0)
        ## masks for all `False` or all `True`
        lt_mask = (~x).all(axis=0)
        gt_mask = (x).all(axis=0)

        inds[lt_mask] = 0
        inds[gt_mask] = N

        return inds

    def discretize(self, x, subslice=(None, None)):
        '''
            x : [ B x observation_dim ]
        '''

        ## enforce batch mode
        if x.ndim == 1:
            x = x[None]

        ## [ N x B x observation_dim ]
        start, end = subslice
        thresholds = self.thresholds[:, start:end]

        gt = x[None] >= thresholds[:,None]
        indices = self.largest_nonzero_index(gt, dim=0)

        if indices.min() < 0 or indices.max() >= self.N:
            indices = np.clip(indices, 0, self.N - 1)

        return indices



    def reconstruct(self, indices, subslice=(None, None)):


        ## enforce batch mode
        if indices.ndim == 1:
            indices = indices[None]

        if indices.min() < 0 or indices.max() >= self.N:
            print(f'[ utils/discretization ] indices out of range: ({indices.min()}, {indices.max()}) | N: {self.N}')
            indices = np.clip(indices, 0, self.N - 1)

        start, end = subslice
        thresholds = self.thresholds[:, start:end]

        left = np.take_along_axis(thresholds, indices, axis=0)
        right = np.take_along_axis(thresholds, indices + 1, axis=0)
        recon = (left + right) / 2.
        return recon
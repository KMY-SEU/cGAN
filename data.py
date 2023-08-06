import torch
import numpy as np
import pandas as pd
from config import args


class Data:
    def __init__(self, flow_path, adj_path):
        # read_data
        self.flow = pd.read_csv(flow_path, header=None)  # flow data
        self.adj = pd.read_csv(adj_path, header=None).values  # adjacency

        # preprocess adjacency matrix
        self.adj = self.adj / args.sigma
        self.adj = np.exp(-self.adj)
        r, c = np.diag_indices_from(self.adj)
        self.adj[r, c] = 0
        self.adj = torch.tensor(self.adj, dtype=torch.float)

        # scaling
        self.scaling()

    def split_train_test(self, val_ratio=0.1, perm=None):
        """
        split origin data into train and test set by validation ratio

        :param val_ratio: proportion of train set and test set
        :return: train set and test set
        """
        #
        T = self.flow.shape[0]
        N = self.flow.shape[1]

        # generate samples
        x = []
        for t in range(T - args.features):
            sp = self.flow.iloc[t: t + args.features, :].values.T  # samples
            x.append(sp)

        # split train and test set
        r = int(len(x) * (1 - val_ratio))
        if perm is None:
            perm = np.random.permutation(range(len(x)))[: r]
        perm_inv = [_ for _ in range(len(x)) if _ not in perm]
        print('perm ==', perm)

        # to be tensor
        x = torch.tensor(x, dtype=torch.float)
        train_x = x[perm]
        test_x = x[perm_inv]

        return train_x, test_x, r, perm

    def scaling(self):
        self.min = self.flow.min()
        self.max = self.flow.max()

        # print('min ==\n', self.min)
        # print('max ==\n', self.max)

        self.flow = (self.flow - self.min) / (self.max - self.min)

        self.min = self.min.values
        self.max = self.max.values

    def inv_scaling(self, flow):
        flow = flow.transpose([0, 2, 1])
        flow = flow * (self.max - self.min) + self.min
        flow = flow.transpose([0, 2, 1])

        return flow

    def destory_data(self, data, missing_rate=0, seed=None):
        if missing_rate == 0:
            return data, None

        if seed is None:
            seed = np.random.randint(0, len(data), size=len(data))
        print('seed ==', seed)

        for i, d in enumerate(data):
            # seed
            np.random.seed(seed[i])

            rand = np.random.random(size=d.shape)
            d[rand < missing_rate] = 0.

        return data, seed

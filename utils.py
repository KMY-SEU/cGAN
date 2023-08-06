import numpy as np
import torch
import os
import pandas as pd
import netCDF4 as nc

from config import args


def MAPE(true, pred):
    bits = true > 10.
    tt = true[bits]
    pp = pred[bits]

    return np.mean(np.abs((tt - pp) / tt)) * 100


def MSE(true, pred):
    return np.mean(np.square(true - pred))


def MAE(true, pred):
    return np.mean(np.abs(true - pred))


def check_cuda():
    # check cuda
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('use_gpu ==', use_gpu)
        print('device_ids ==', np.arange(0, torch.cuda.device_count()))
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def save_model(model):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    torch.save(model, args.save_path + 'generator.pkl')


def save_scenarios(scenarios, filename):
    #
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    #
    dataset = nc.Dataset(args.save_path + filename, 'w')

    time = dataset.createDimension('time', size=args.features)
    node = dataset.createDimension('node', size=args.nodes)
    num = dataset.createDimension('num', size=len(scenarios))

    sce = dataset.createVariable('scenarios', datatype=float, dimensions=['num', 'node', 'time'])

    sce[:] = scenarios


def save_loss(pltx, results, file_path):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    pd.DataFrame(zip(pltx, results)).to_csv(args.save_path + file_path, index=False)


def save_seed(seed):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    seed = pd.DataFrame(seed)
    seed.to_csv(args.save_path + 'seed.csv', header=False, index=False)
    print('seed is saved.')


def load_seed():
    seed = pd.read_csv(args.save_path + 'seed.csv', header=None)
    seed = np.array([_[0] for _ in seed.values])
    return seed


def save_perm(perm):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    perm = pd.DataFrame(perm)
    perm.to_csv(args.save_path + 'perm.csv', header=False, index=False)
    print('perm is saved.')


def load_perm():
    try:
        perm = pd.read_csv(args.save_path + 'perm.csv', header=None)
        perm = np.array([_[0] for _ in perm.values])
    except FileNotFoundError:
        perm = None
    return perm


def save_results(flow, file_name):
    pd.DataFrame(flow).to_csv(args.save_path + file_name, index=False)


def validate(true, pred, _, detransform):
    with torch.no_grad():
        # numpy
        true = true.numpy()
        pred = pred.cpu().numpy()

        # detransforms
        true = detransform(true)
        pred = detransform(pred)

        nmiss = (_ == 0).cpu().numpy()
        mse = MSE(true[nmiss], pred[nmiss])
        mape = MAPE(true[nmiss], pred[nmiss])

    return mse, mape


def impute_concat(batch, s):
    if s != 0:
        tensor = batch[0, :, -1].reshape([args.nodes, 1])
        for b in range(1, len(batch)):
            bb = batch[b]
            tensor = np.concatenate([tensor, bb[:, -1].reshape([args.nodes, 1])], axis=1)
    else:
        tensor = batch[0]
        for b in range(1, len(batch)):
            bb = batch[b]
            tensor = np.concatenate([tensor, bb[:, -1].reshape([args.nodes, 1])], axis=1)

    return tensor


def save_flow(flow, flag):
    if flag == 'true':
        pd.DataFrame(flow.T).to_csv(args.save_path + 'true.csv', index=False, header=False)
    elif flag == 'pred':
        pd.DataFrame(flow.T).to_csv(args.save_path + 'pred.csv', index=False, header=False)
    elif flag == 'sparse':
        pd.DataFrame(flow.T).to_csv(args.save_path + 'sparse.csv', index=False, header=False)
    else:
        print('save flow error.')


def calculate_MMD(true, pred):
    #
    B, N, T = true.shape

    def Gaussian_gram_matrix(s1, s2):
        gamma = 1.
        ones = np.ones(shape=[B, T])

        alpha = np.einsum('bnt, bnt -> bnt', s1, s1)
        alpha = np.einsum('bnt -> bt', alpha)

        beta = np.einsum('bnt, bnt -> bnt', s2, s2)
        beta = np.einsum('bnt -> bt', beta)

        amo = np.einsum('bi, bj -> bij', alpha, ones)
        omb = np.einsum('bi, bj -> bij', ones, beta)

        diff2 = 2 * np.einsum('bni, bnj -> bij', s1, s2) - amo - omb

        return np.exp(diff2 / gamma)

    # Calculate MMD distance
    K_xx = Gaussian_gram_matrix(true, true)
    K_xy = Gaussian_gram_matrix(true, pred)
    K_yy = Gaussian_gram_matrix(pred, pred)

    # calculate mmd
    ones = np.ones(shape=[B, T])
    kxxkyy = K_xx + K_yy
    kxxkyy = np.einsum('bi, bij -> bj', ones, kxxkyy)
    kxxkyy = np.einsum('bi, bi -> b', kxxkyy, ones)
    kxy = np.einsum('bi, bij -> bj', ones, K_xy)
    kxy = np.einsum('bi, bi -> b', kxy, ones)

    T2 = np.full(shape=[B], fill_value=2 * T)
    mmd = (1 / (T * (T - 1))) * (kxxkyy - T2) - (2 / T ** 2) * kxy
    return mmd

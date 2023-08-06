import torch
import numpy as np
import pandas as pd

from utils import *
from data import Data
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # device
    device = check_cuda()

    # read data, and then split train & test dataset
    data = Data(
        args.flow_path,
        args.adj_path,
    )

    # split train & test set
    perm = load_perm()
    train_x, test_x, r, perm = data.split_train_test(val_ratio=args.val_ratio, perm=perm)

    # create data missing
    seed = None if args.missing_rate == 0 else load_seed()
    train_x, seed = data.destory_data(train_x, missing_rate=args.missing_rate, seed=seed)

    # model
    generator = torch.load(args.save_path + 'generator.pkl')
    generator = generator.to(device=device)

    # create data loader
    torch_dataset = TensorDataset(train_x, train_x)
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    # compute indices
    mmds = []
    mses = []
    maes = []
    mapes = []
    true = test_x[: args.batch_size].clone().detach().cpu().numpy()

    for step, (bx, by) in enumerate(tqdm(loader)):
        # noise term
        z = torch.randn(size=[args.batch_size, 1, args.features])
        z = z.expand(args.batch_size, args.nodes, args.features)
        z = z.to(device=device)
        bx = bx.to(device=device)

        pred = generator(bx + z)
        pred = pred.clone().detach().cpu().numpy()

        # MMD distance
        mmd = calculate_MMD(true, pred)
        mmds.append(mmd)

        # RMSE & MAE
        pp = data.inv_scaling(pred)
        tt = data.inv_scaling(true)

        mse = MSE(tt, pp)
        mses.append(mse)

        mae = MAE(tt, pp)
        maes.append(mae)

        mape = MAPE(tt, pp)
        mapes.append(mape)

    print('mmd ==', np.mean(mmds[: -1]))
    print('rmse ==', np.sqrt(np.mean(mses[: -1])))
    print('mae ==', np.mean(maes[: -1]))
    print('mape ==', np.mean(mapes[: -1]))

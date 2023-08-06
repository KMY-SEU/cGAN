from cGAN import cGAN
from config import args
from data import Data
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
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
    cgan = cGAN(
        adj=data.adj,
        lambda_term=args.lambda_term,
        rc_term=args.rc_term
    )

    cgan.train(train_x, r)

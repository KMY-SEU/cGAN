'''
    Author: kangmingyu
    Email: kangmingyu@seu.edu.cn
    Institute: CCCS Lab, Southeast University
'''

import argparse

parser = argparse.ArgumentParser()

# data path
parser.add_argument('--flow_path', default='./pv_data/pv_data.csv')
parser.add_argument('--adj_path', default='./pv_data/A.csv')
parser.add_argument('--save_path', default='./saved_model/')

# validation ratio
parser.add_argument('--val_ratio', default=0.1)
# missing rate
parser.add_argument('--missing_rate', default=0)

# model setting
parser.add_argument('--nodes', default=69)
parser.add_argument('--features', default=96)  # 144 & 96
parser.add_argument('--sigma', default=100)  # 100 & 1000
# scale flow data from in_channels to embedding size
parser.add_argument('--embed_size', default=32)
# the number of time intervals
parser.add_argument('--time_num', default=96)
# the number of STTN blocks
parser.add_argument('--num_layers', default=2)
# the number of STTN heads
parser.add_argument('--heads', default=1)
# lambda & mmd term
parser.add_argument('--lambda_term', default=10.)
parser.add_argument('--rc_term', default=10.)

# model training
# learning rate
parser.add_argument('--lr', default=0.0001)
# training epoch
parser.add_argument('--epochs', default=50000)
# g / d iters
parser.add_argument('--g_iters', default=2)
parser.add_argument('--d_iters', default=3)
# batch size
parser.add_argument('--batch_size', default=32)
# parallel setting
parser.add_argument('--parallel', default=False)

# parse arguments
args = parser.parse_args()

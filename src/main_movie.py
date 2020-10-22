import argparse
import numpy as np
from data_loader import load_data
from train import train
import os


parser = argparse.ArgumentParser()


# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=40, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--cg_weight', type=float, default=1, help='weight of cycle gan')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.01, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=0.005, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')

show_loss = False
show_topk = False

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print(args)
data = load_data(args)
train(args, data, show_loss, show_topk)

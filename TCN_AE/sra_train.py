import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("./")
from model import TCN
from utils import load_MUSE_Fi_data,MUSEFiDataset
import numpy as np
from scipy.io import savemat

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=64,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='re_1', 
                    help='the dataset to run (default: my)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--batchsize', type=int, default=128,
                    help='batch_size (default: 512)')

args = parser.parse_args()

torch.manual_seed(args.seed)

print(args)
input_size = 32
X_train, Y_train, X_test, Y_test = load_MUSE_Fi_data(args.data)

n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

model = TCN(input_size, input_size, n_channels, kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def illustrate(X_data, Y_data):
    model.eval()
    dir_name = './result_data_'+args.data
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    x_list = []
    y_list = []
    output_list = []
    with torch.no_grad():
        for idx in range(len(X_data)):
            data_line = X_data[idx]
            label_line = Y_data[idx]
            x, y = Variable(data_line.unsqueeze(0)), Variable(label_line.unsqueeze(0))
            if args.cuda:
              x, y = x.cuda(), y.cuda()
            output = model(x).squeeze(-1)
            x_list.append(x.cpu().numpy().squeeze(0))
            y_list.append(y.cpu().numpy().squeeze(0))
            output_list.append(output.cpu().numpy().squeeze(0))
    savemat(dir_name + '/result_data_list.mat', {'x':x_list, 'y':y_list,
        'output':output_list})


def evaluate(X_data, Y_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    eval_set = MUSEFiDataset(X=X_data, Y=Y_data)
    loader_args = dict(batch_size=args.batchsize, pin_memory=True)
    eval_loader = DataLoader(eval_set, shuffle=True, **loader_args)
    with torch.no_grad():
        for batch in eval_loader:
            data_line, label_line = batch['data'], batch['label']
            x, y = Variable(data_line), label_line
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x).squeeze(-1)
            loss = torch.mean((y - output) ** 2)
            total_loss += loss.item() * data_line.shape[0]
            count += data_line.shape[0]
        eval_loss = total_loss / count
        print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    train_set = MUSEFiDataset(X=X_train, Y=Y_train)
    loader_args = dict(batch_size=args.batchsize, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    print("Number of data is %d\n" % len(train_set))
    global_step = 0
    for batch in train_loader:
        data_line, label_line = batch['data'], batch['label']
        x, y = Variable(data_line), label_line
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x).squeeze(-1)
        loss = torch.mean((y - output) ** 2)
        total_loss += loss.item() * data_line.shape[0]
        count += data_line.shape[0]

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        global_step += 1
        if global_step > 0 and global_step % args.log_interval == 0:
            cur_loss = total_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
            total_loss = 0.0
            count = 0

if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    tloss_list = []
    model_name = "./reconstruct_{0}.pt".format(args.data)
    for ep in range(1, args.epochs+1):
        train(ep)
        tloss = evaluate(X_test, Y_test, name='Test')
        print("Epoch {:2d} | Test loss {:.5f}".format(ep, tloss))
        tloss = evaluate(X_train, Y_train, name='Train')
        print("Epoch {:2d} | Train loss {:.5f}".format(ep, tloss))

        tloss_list.append(tloss)
    tloss = illustrate(X_test, Y_test)
 


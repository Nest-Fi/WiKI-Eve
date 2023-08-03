from scipy.io import loadmat
import torch
import numpy as np
import gc
from torch.utils.data import Dataset

class MUSEFiDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return{
            'data': self.X[idx],
            'label': self.Y[idx]
        }

def data_generator(dataset):
    
    X_train = torch.from_numpy(data['X_train'].astype(np.float32))
    Y_train = torch.from_numpy(data['Y_train'].astype(np.float32))
    X_valid = torch.from_numpy(data['X_valid'].astype(np.float32))
    Y_valid = torch.from_numpy(data['Y_valid'].astype(np.float32))
    X_test = torch.from_numpy(data['X_test'].astype(np.float32))
    Y_test = torch.from_numpy(data['Y_test'].astype(np.float32))


    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def load_MUSE_Fi_data(data_name):
    load_dict = loadmat('../Data/SRA/'+data_name+'_train.mat')
    X_train = load_dict['data']
    X_train_tensor = [torch.from_numpy(X_train[0][i].astype(np.float32)) for i in range(X_train[0].shape[0])]
    Y_train = load_dict['label']
    Y_train_tensor = [torch.from_numpy(Y_train[0][i].astype(np.float32)) for i in range(Y_train[0].shape[0])]
    del load_dict
    gc.collect()
    load_dict = loadmat('../Data/SRA/'+data_name+'_test.mat')
    X_test = load_dict['data']
    X_test_tensor = [torch.from_numpy(X_test[0][i].astype(np.float32)) for i in range(X_test[0].shape[0])]
    Y_test = load_dict['label']
    Y_test_tensor = [torch.from_numpy(Y_test[0][i].astype(np.float32)) for i in range(Y_test[0].shape[0])]
    return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor

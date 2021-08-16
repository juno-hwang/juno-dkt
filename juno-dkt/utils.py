import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

def progressBar(current, total, msg='', barLength = 30):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow) - 1)
    print('Progress: [%s%s] %.1f %%' % (arrow, spaces, percent)+msg+' '*10, end='\r')

def collate(batch):
	return nn.utils.rnn.pad_sequence(batch)

class TimeSeries(Dataset):
    def __init__(self, batches):
        self.batches = batches
        self.len = len(batches)

    def __getitem__(self,idx):
        return self.batches[idx]
  
    def __len__(self):
        return self.len
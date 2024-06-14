import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class HSICD_dataset(torch.utils.data.Dataset):
    def __init__(self, pre, after, label):
        pre_trans = pre.transpose(0, 3, 1, 2)
        after_trans = after.transpose(0, 3, 1, 2)
        self.pre = torch.FloatTensor(pre_trans)
        self.after = torch.FloatTensor(after_trans)
        self.label = torch.LongTensor(label)

    def __len__(self):
        return len(self.pre)

    def __getitem__(self, index):
        return self.pre[index], self.after[index], self.label[index]
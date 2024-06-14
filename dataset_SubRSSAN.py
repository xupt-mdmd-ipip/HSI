import torch
import numpy as np
from torch.utils.data import Dataset

class SubRAASN_dataset(Dataset):
    def __init__(self, pre, after, label, input_channels, output_size):
        # 转置数据，将通道维度放在正确的位置
        pre_trans = np.transpose(pre, (0, 3, 1, 2))
        after_trans = np.transpose(after, (0, 3, 1, 2))

        # 将修正后的数据保存到对象属性中
        self.pre = torch.FloatTensor(pre_trans)
        self.after = torch.FloatTensor(after_trans)
        self.label = torch.LongTensor(label)
        self.input_channels = input_channels
        self.output_size = output_size

    def __len__(self):
        return len(self.pre)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}")
        return self.pre[index], self.after[index], self.label[index]

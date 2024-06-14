import os
import numpy as np
import torch
import scipy.io as scio
from Model import HSICD_Model
from test import test_step
from dataset_my import HSICD_dataset
from torch.utils.data import DataLoader
from PIL import Image
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_all():
    # 导入数据
    data_pre = np.load('data_process/data_pre.npy')
    data_after = np.load('data_process/data_after.npy')
    label = np.load('data_process/label.npy')
    test_ds = HSICD_dataset(data_pre, data_after, label)
    test_dl = DataLoader(dataset=test_ds, batch_size=128, shuffle=False)
    # 创建模型
    model = HSICD_Model(30, 128, 1, 15).to(device)
    # 导入模型参数
    model.load_state_dict(torch.load('model/trained_model_20_1.pth'))
    _, _, pred_list = test_step(model, test_dl)
    os.makedirs('pred_list', exist_ok=True)
    np.save('pred_list/pred_all', pred_list)


if __name__ == '__main__':
    compute = 0
    if compute == 1:
        compute_all()
    else:
        label_truth = scio.loadmat('The River Data Set/groundtruth.mat')['lakelabel_v1']
        pred_list = np.load('pred_list/pred_all.npy')
        pred_class = np.reshape(pred_list, (label_truth.shape[0], label_truth.shape[1]))
        ground_truth = Image.fromarray(label_truth, 'L')
        change_map = Image.fromarray(np.uint8(pred_class * 255), 'L')
        os.makedirs('fig', exist_ok=True)
        ground_truth.save('fig/ground_truth.png')
        change_map.save('fig/change_map.png')
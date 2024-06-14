import argparse
import numpy as np
import torch
from data_process import data_load_process
from dataset_SubRSSAN import SubRAASN_dataset
from dataset_my import HSICD_dataset
from torch.utils.data import DataLoader
from Model import HSICD_Model
from Model import SubRSSAN
from train import train_model
from test import test_step

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parser_set():
    parser = argparse.ArgumentParser()

    # 数据地址

  #  parser.add_argument("-dpp", "--pre_path", type=str, default='./The River Data Set/river_before.mat')
  #  parser.add_argument("-dap", "--after_path", type=str, default='./The River Data Set/river_after.mat')
  #  parser.add_argument("-gtp", "--gt_path", type=str, default='./The River Data Set/groundtruth.mat')

  #  parser.add_argument("-dpp", "--pre_path", type=str, default='./data/farmland/China_Change_Dataset.mat')
  #  parser.add_argument("-dap", "--after_path", type=str, default='./data/farmland/China_Change_Dataset.mat')
  #  parser.add_argument("-gtp", "--gt_path", type=str, default='./data/farmland/China_Change_Dataset.mat')

    parser.add_argument("-dpp", "--pre_path", type=str, default='./data/Hermiston/hermiston2004.mat')
    parser.add_argument("-dap", "--after_path", type=str, default='./data/Hermiston/hermiston2007.mat')
    parser.add_argument("-gtp", "--gt_path", type=str, default='./data/Hermiston/rdChangesHermiston_5classes.mat')

    parser.add_argument("-ptp", "--train_pre", type=str, default='data_process/data_pre_train.npy')
    parser.add_argument("-pta", "--train_after", type=str, default='data_process/data_after_train.npy')
    parser.add_argument("-ptl", "--train_label", type=str, default='data_process/label_train.npy')
    parser.add_argument("-ppt", "--test_pre", type=str, default='data_process/data_pre_test.npy')
    parser.add_argument("-pat", "--test_after", type=str, default='data_process/data_after_test.npy')
    parser.add_argument("-plt", "--test_label", type=str, default='data_process/label_test.npy')

    # 数据处理参数
    parser.add_argument("-np", "--need_process", type=int, default=0)
    parser.add_argument("-ws", "--window_size", type=int, default=15)
    parser.add_argument("-pc", "--PCA_components", type=int, default=30)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.5)

    # 数据集描述
    parser.add_argument("-dsp", "--pre_ds", type=str, default='HypeRvieW')
    parser.add_argument("-dsa", "--after_ds", type=str, default='HypeRvieW')
    parser.add_argument("-dsg", "--gt_ds", type=str, default='gt5clasesHermiston')

    # 模型训练参数
    parser.add_argument("-nt", "--need_train", type=int, default=1)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-ep", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--lr", type=int, default=0.005)
    parser.add_argument("-ds", "--ds", type=int, default=1)
    parser.add_argument("-hc", "--hidden_c", type=int, default=128)

    # 模型地址
   # parser.add_argument("-mp", "--model_path", type=str, default='model/trained_model_100_1.pth')
    parser.add_argument("-mi", "--model_idx", type=int, default=1)
    args_set = parser.parse_args()
    args_set.model_idx = 1
    return args_set


if __name__ == '__main__':
    args = parser_set()

    # 数据处理
    if args.need_process == 1:
        train_pre, test_pre, train_after, test_after, train_label, test_label = data_load_process(args)
    else:
        if args.need_train == 1:
            train_pre = np.load(args.train_pre)
            train_after = np.load(args.train_after)
            train_label = np.load(args.train_label)
            test_pre = np.load(args.test_pre)
            test_after = np.load(args.test_after)
            test_label = np.load(args.test_label)

            # 创建模型
            if args.model_idx == 0:
                # 将数据创建Dataset
                train_ds = SubRAASN_dataset(train_pre, train_after, train_label, input_channels=30, output_size=2)
                test_ds = SubRAASN_dataset(test_pre, test_after, test_label, input_channels=30, output_size=2)
                # 创建DataLoader
                train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=False)
                test_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)
                model = SubRSSAN(input_channels=30, output_size=2)
                model.to(device)
                train_model(model, train_dl, test_dl, args)
            if args.model_idx == 1:
                # 将数据创建Dataset
                train_ds = HSICD_dataset(train_pre, train_after, train_label)
                test_ds = HSICD_dataset(test_pre, test_after, test_label)
                # 创建DataLoader
                train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=False)
                test_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)
                model = HSICD_Model(args.PCA_components, args.hidden_c, args.ds, args.window_size).to(device)
                train_model(model, train_dl, test_dl, args)
        else:
            test_pre = np.load(args.test_pre)
            test_after = np.load(args.test_after)
            test_label = np.load(args.test_label)
            test_ds = HSICD_dataset(test_pre, test_after, test_label)
            test_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)
            # 创建模型
            model = HSICD_Model(args.PCA_components, args.hidden_c, args.ds, args.window_size).to(device)
            # 导入模型参数
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            result, _, _ = test_step(model, test_dl)
            print(result)
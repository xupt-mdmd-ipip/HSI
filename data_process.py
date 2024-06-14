import os
import numpy as np
import scipy.io as scio
import random
from sklearn.decomposition import PCA


def data_load_process(args):
   # data_pre = scio.loadmat(args.pre_path)[args.pre_ds]
   # data_after = scio.loadmat(args.after_path)[args.after_ds]
   # label_all = scio.loadmat(args.gt_path)[args.gt_ds]

   # data_pre = scio.loadmat(args.pre_path)['T1']
   # data_after = scio.loadmat(args.after_path)['T2']
   # label_all = scio.loadmat(args.gt_path)['Binary']

    data_pre = scio.loadmat(args.pre_path)[args.pre_ds]
    data_after = scio.loadmat(args.after_path)[args.after_ds]
    label_all = scio.loadmat(args.gt_path)[args.gt_ds]

    label_new = np.reshape(label_all, -1)
    label_new = label_new // 255
    # 通过主成分分析法（PCA）进行数据降维
    # PCA需要二维形式，对数据变形
    data_pre_rs = np.reshape(data_pre, (-1, data_pre.shape[2]))
    data_after_rs = np.reshape(data_after, (-1, data_after.shape[2]))
    pca_func = PCA(n_components=args.PCA_components)
    data_pre_pca = pca_func.fit_transform(data_pre_rs)
    data_after_pca = pca_func.fit_transform(data_after_rs)
    data_pre_pca = np.reshape(data_pre_pca, (data_pre.shape[0], data_pre.shape[1], args.PCA_components))
    data_after_pca = np.reshape(data_after_pca, (data_after.shape[0], data_after.shape[1], args.PCA_components))
    # 归一化数据
    #data_pre = (data_pre - np.min(data_pre)) / (np.max(data_pre) - np.min(data_pre))
    #data_after = (data_after - np.min(data_after)) / (np.max(data_after) - np.min(data_after))
    # 对数据进行扩维
    data_offset = args.window_size // 2
    data_pre_exp = np.zeros((data_pre.shape[0] + 2 * data_offset, data_pre.shape[1] + 2 * data_offset, data_pre_pca.shape[2]))
    data_after_exp = np.zeros((data_after.shape[0] + 2 * data_offset, data_after.shape[1] + 2 * data_offset, data_after_pca.shape[2]))
    data_pre_exp[data_offset:data_pre.shape[0] + data_offset, data_offset:data_pre.shape[1] + data_offset, :] = data_pre_pca
    data_after_exp[data_offset:data_after.shape[0] + data_offset, data_offset:data_after.shape[1] + data_offset, :] = data_after_pca
    # 创造存储每个像素点邻域信息的空数组
    data_pre_new = np.zeros(
        (data_pre.shape[0] * data_pre.shape[1], args.window_size, args.window_size, data_pre_pca.shape[2]))
    data_after_new = np.zeros(
        (data_after.shape[0] * data_after.shape[1], args.window_size, args.window_size, data_after_pca.shape[2]))
    pre_index = 0
    after_index = 0
    # 获取每个像素点的邻域数据
    for i in range(0, data_pre.shape[0]):
        for j in range(0, data_pre.shape[1]):
            data_pre_new[pre_index, :, :, :] = data_pre_exp[i:i+15, j:j+15, :]
            data_after_new[after_index, :, :, :] = data_after_exp[i:i+15, j:j+15, :]
            pre_index += 1
            after_index += 1
    del data_pre, data_after, label_all, data_pre_rs, data_after_rs, data_pre_pca, data_after_pca, data_pre_exp, data_after_exp
    train_pre, test_pre, train_after, test_after, train_label, test_label = divide_data(data_pre_new, data_after_new, label_new, args)
    # 保存训练好的数据
    os.makedirs('data_process_Hermiston', exist_ok=True)
    np.save('data_process_Hermiston/data_pre_train.npy', train_pre)
    np.save('data_process_Hermiston/data_after_train.npy', train_after)
    np.save('data_process_Hermiston/label_train.npy', train_label)
    np.save('data_process_Hermiston/data_pre_test.npy', test_pre)
    np.save('data_process_Hermiston/data_after_test.npy', test_after)
    np.save('data_process_Hermiston/label_test.npy', test_label)
    return train_pre, test_pre, train_after, test_after, train_label, test_label


# 数据集划分函数
def divide_data(pre, after, label, args):
    index = [i for i in range(len(pre))]
    random.shuffle(index)
    pre_random = pre[index]
    after_random = after[index]
    label_random = label[index]
    divide_loca = int(len(pre) * args.train_ratio)
    train_pre = pre_random[:divide_loca]
    test_pre = pre_random[divide_loca:]
    train_after = after_random[:divide_loca]
    test_after = after_random[divide_loca:]
    train_label = label_random[:divide_loca]
    test_label = label_random[divide_loca:]
    return train_pre, test_pre, train_after, test_after, train_label, test_label
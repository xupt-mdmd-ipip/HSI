from scipy.io import loadmat
from PIL import Image
import numpy as np
import ipdb; ipdb.set_trace()
import matplotlib.pyplot as plt
from matplotlib import colors

color_mat = loadmat('F:\毕设\IEEE_TGRS_SSTFormer-main\data\farmland\China_Change_Dataset.mat')
color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]] #(17,3)

# pred = Image.open('barbara.tif')
# data_label = loadmat("F:\毕设\IEEE_TGRS_SSTFormer-main\data\farmland\China_Change_Dataset.mat")['HypeRvieW']

pred = Image.open('BayArea.tif')
data_label = loadmat("F:\毕设\IEEE_TGRS_SSTFormer-main\data\farmland\China_Change_Dataset.mat")['HypeRvieW']

# pred = Image.open('barbara.tif')
# data_label = loadmat("F:\毕设\IEEE_TGRS_SSTFormer-main\data\farmland\China_Change_Dataset.mat")['HypeRvieW']

data_label[data_label!=0]=1

pred = np.array(pred)
pred = pred/255
output=pred*data_label
output=output+data_label
# plt.subplot(1,1,1)
# plt.imshow(output, colors.ListedColormap(color_matrix))
# # plt.show()
# import ipdb; ipdb.set_trace()
im=Image.fromarray(output/2*255)
im=im.convert('RGB')
im.show()


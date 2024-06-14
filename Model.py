import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from sklearn import svm

import torch
import torch.nn as nn


class SubRSSAN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SubRSSAN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # 创建第二组卷积层，用于处理第二个输入
        self.conv1_2 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 修改全连接层的输入大小
        self.fc1 = nn.Linear(512, 512)  # 输入大小是连接后的特征图的大小
        self.fc2 = nn.Linear(512, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        # 对第一个输入进行处理
        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        x1 = self.pool(x1)
        x1 = self.relu(self.conv3(x1))
        x1 = self.relu(self.conv4(x1))
        x1 = self.pool(x1)
        x1 = self.relu(self.conv5(x1))
        x1 = self.relu(self.conv6(x1))
        x1 = self.pool(x1)
        # 对第二个输入进行相同的处理
        x2 = self.relu(self.conv1_2(x2))
        x2 = self.relu(self.conv2_2(x2))
        x2 = self.pool(x2)
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = self.pool(x2)
        x2 = self.relu(self.conv5(x2))
        x2 = self.relu(self.conv6(x2))
        x2 = self.pool(x2)
        # 将两个处理后的特征图在通道维度进行连接
        x = torch.cat((x1, x2), dim=1)
        # 展平张量，这里需要考虑两个特征图连接后的形状
        x = x.view(x.size(0), -1)
        # 通过全连接层和dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class BAM(nn.Module):
    """ 基础自注意力模块
    """
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        print('ds: ', ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  # 计算注意力权重

    def forward(self, x_input):
        x = self.pool(x_input)
        batch_size, C, width, height = x.size()  # 获取池化后的数据的形状
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # 计算注意力矩阵
        energy = (self.key_channel**-.5) * energy  # 归一化处理

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = F.interpolate(out, [width*self.ds,height*self.ds])
        out = out + x_input

        return out
    
# 初始化模型的参数

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class STA(nn.Module):
    """自注意力模块
    """
    def __init__(self, in_c, ds):
        super(STA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        self.Self_Att = BAM(self.in_C, ds=ds)
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)   # 在通道维度上连接两个输入特征图
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]  # 将处理后的特征图分割为两个部分并返回


class HSICD_Model(nn.Module):
    def __init__(self, in_chanel, hidden_c, ds, window_size):
        super(HSICD_Model, self).__init__()
        # 定义特征提取（2D卷积）
        self.feature = nn.Conv2d(in_chanel, hidden_c, kernel_size=3, stride=1, padding=1)
        # 定义光谱-时序注意力模块
        self.STA = STA(in_c=hidden_c, ds=ds)
        # 分类层
        self.fc = nn.Linear(2 * hidden_c * window_size * window_size, 2)

    def forward(self, x_t1, x_t2):
        x_t1 = self.feature(x_t1)
        x_t2 = self.feature(x_t2)
        x_t1, x_t2 = self.STA(x_t1, x_t2)
        pre_x = torch.cat((x_t1, x_t2), dim=1)
        pred_L = pre_x.view(pre_x.shape[0], -1)
        pred_cd = self.fc(pred_L)

        return pred_cd

# 创建两个随机张量作为输入，并实例化了HSICD_Model模型，对输入进行前向传播，得到输出
if __name__ == '__main__':
    x_1 = torch.rand([1, 200, 15, 15], dtype=torch.float32).cuda()
    x_2 = torch.rand([1, 200, 15, 15], dtype=torch.float32).cuda()
    x_3 = 0;
    if x_3 == 1:
        model = HSICD_Model(in_chanel=200, hidden_c=64, ds=1, window_size=15).cuda()
    if x_3 == 0:
        model = SubRSSAN(input_channels=30, output_size=2).cuda()
    y = model(x_1, x_2)
import os
import torch
import torch.nn as nn
from test import test_step
from tqdm import tqdm
device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_model(model, train_dl, test_dl, args):
    # 交叉熵损失函数
    Loss_fn = nn.CrossEntropyLoss()
    # Adam优化器
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)

    for epoch in tqdm(range(args.epochs)):
        # 训练
        model.train()
        train_acc = 0
        train_loss = 0
        for batch, (train_pre, train_after, train_label) in enumerate(train_dl):
            # print('batch:', batch+1)
            # 将数据转换进显存（如果有）
            train_pre = train_pre.to(device)
            train_after = train_after.to(device)
            train_label = train_label.to(device)


            # 获取预测结果
            #train_pred = model(train_pre, train_after)
            train_pred = model.forward(train_pre, train_after)



            # 计算损失
            loss = Loss_fn(train_pred, train_label)
            train_loss += loss.item()
            # 优化器零梯度
            optim.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optim.step()
            # 计算准确率
            pred_class = torch.argmax(torch.softmax(train_pred, dim=1), dim=1)
            train_acc_now = (pred_class == train_label).sum().item() / len(train_label)
            train_acc += train_acc_now
            # print('now batch acc:', train_acc_now)

        # 按批次调整train_loss和train_acc
        train_loss /= len(train_dl)
        train_acc /= len(train_dl)
        # print('now epoch acc:', train_acc)

        test_result, test_acc, _ = test_step(model, test_dl)
        print(test_result)
        os.makedirs('model_River_SVM', exist_ok=True)
        torch.save(model.state_dict(), 'model_River_SVM/trained_model_' + str(epoch+1) + '_' + str(args.model_idx) + '.pth')

        # 调整学习率
        scheduler.step()

        # 显示当前Epochs的指标结果
        print(
            f"Epoch: {epoch + 1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_step(model, test_dl):
    # 设置为测试模式，梯度不更新
    model.eval()
    # 定义空数组保存预测结果值
    test_pred_list = torch.empty(0).to(device)
    test_label_list = torch.empty(0).to(device)
    with torch.inference_mode():
        for batch, (test_pre, test_after, test_label) in enumerate(test_dl):
            test_pre = test_pre.to(device)
            test_after = test_after.to(device)
            test_label = test_label.to(device)
            test_pred = model(test_pre, test_after)
            # 保存预测结果
            test_pred = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_pred_list = torch.cat((test_pred_list, test_pred), dim=0)
            test_label_list = torch.cat((test_label_list, test_label), dim=0)

    # 计算其他结果
    test_pred_list = test_pred_list.detach().cpu().numpy()
    test_label_list = test_label_list.detach().cpu().numpy()
    test_acc = accuracy_score(test_pred_list, test_label_list)
    result = classification_report(test_pred_list, test_label_list, digits=6)
    return result, test_acc, test_pred_list
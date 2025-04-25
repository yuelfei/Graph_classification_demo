'''
@File name: GCN.py
@Author: Yuefei Wu
@Version: 1.0
@Creation time: 2025/3/28 - 11:17
@Description:
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
from torch_geometric.datasets import Planetoid
from model import GCN, GAT, GIN
from config import parse_args
import copy,shutil
import datetime
from loguru import logger
import pandas as pd


def dict_to_excel(data, file_name):
    logger.info("-" * 100)
    header = ["Train_acc", "Val_acc", "Test_acc"]
    logger.info("{:<10} {:<10} {:<10}".format(*header))

    value = [str(data[node])[:6] for node in header]
    logger.info("{:<10} {:<10} {:<10}".format(*value))
    logger.info("-" * 100)

    # 若文件存在则删除
    if os.path.exists(file_name+".xlsx"):
        os.remove(file_name+".xlsx")
        logger.info(f"Delete {file_name}.xlsx")

    df = pd.DataFrame(data, index=[0])
    df.to_excel(file_name+".xlsx", index=False)
    logger.info(f"Save to {file_name}.xlsx")


def train(model,optimizer,data,criterion):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return float(loss), copy.deepcopy(model.state_dict())

@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def main():

    # 解析参数
    args = parse_args()

    # 文件标识
    file_tag=args.dataset + "_" + str(args.lr) + "_" + args.model

    # 创建日志
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger.add("./log/log_" + file_tag + "_" + now_time + ".log",format='{time:YYYY-MM-DD HH:mm:ss} - {level} -  {file} - {line} - {message}', level="INFO")

    # 打印参数
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # 指定GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    dataset = Planetoid(root='./data', name=args.dataset)
    data = dataset[0]
    data = data.to(device)

    # 创建模型
    if args.model == "GCN":
        model = GCN(in_channels=dataset.num_features,  hidden_channels=args.hidden_channels,  out_channels=dataset.num_classes)
    elif args.model == "GAT":
        model = GAT(in_channels=dataset.num_features, hidden_channels=args.hidden_channels, out_channels=dataset.num_classes, heads=args.head_num)
    model = model.to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 准确率初始值
    best_val_acc= 0
    test_acc = 0
    state_dict_model=None
    best_epoch=0

    # 按args.output创建文件夹，不存在则创建，存在则清空
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        shutil.rmtree(args.output)
        os.makedirs(args.output)

    # 训练
    for epoch in range(1, args.epochs + 1):

        loss, state_dict_now = train(model, optimizer, data, criterion)

        train_acc, val_acc, tmp_test_acc = test(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            state_dict_model=state_dict_now
            best_epoch=epoch

        logger.info('Epoch: {}, Loss: {}, Train_acc: {}, Val_acc: {}, Test_acc: {}'.format(epoch, loss, train_acc, val_acc, test_acc))

    # 保存模型
    torch.save(state_dict_model, args.output + '/' + file_tag + "_" + str(best_epoch) +'_model.pth')

    # 加载最优模型
    model.load_state_dict(state_dict_model)

    # 测试
    final_train_acc, final_val_acc, final_tmp_test_acc = test(model, data)

    logger.info('Final: Train_acc: {}, Val_acc: {}, Test_acc: {}'.format(final_train_acc, final_val_acc, final_tmp_test_acc))

    dict_to_excel({
        "Train_acc": final_train_acc, "Val_acc": final_val_acc, "Test_acc": final_tmp_test_acc
    },file_name='./log/' + file_tag + "_" + str(best_epoch) +'_result')


if __name__ == '__main__':

    main()








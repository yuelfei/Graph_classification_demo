'''
@File name: config.py
@Author: Yuefei Wu
@Version: 1.0
@Creation time: 2025/3/28 - 11:26
@Description: 
'''


import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora',choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--model', type=str, default="GCN", choices=["GCN", "GAT"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='./output')

    return parser.parse_args()






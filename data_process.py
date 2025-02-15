# coding=utf-8
import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data


def load_10_fold_data(t_edge):
    t_data = pd.read_csv('./Dataset/Graph/node_feature_normalized.csv')
    t_data = t_data.drop(columns=['node', 'value_d', 'value_w', 'avg_value_d', 'avg_value_w'])

    names = t_data.columns.tolist()
    names = np.delete(names, 0)#删掉第一列

    texId2index = {}
    for index, row in t_data.iterrows():
        texId2index[int(row.iloc[0])] = index #[nodeid] = index 68419所有的行索引都在

    x = t_data.iloc[:, 1:]#从第二列开始的所有数据
    x = x.reset_index(drop=True)
    x = x.to_numpy().astype(np.float32)

    x[x == np.inf] = 1. #无穷大替换为1
    x[np.isnan(x)] = 0. #NaN值替换为0

    edges = []
    labels = []
    for _, row in t_edge.iterrows():#训练集92个pos+neg,183×3 验证集就小一点22×3
        id_1, id_2 = int(row.iloc[0]), int(row.iloc[1])
        label = int(row.iloc[2])
        if id_1 not in texId2index or id_2 not in texId2index: #如果训练边的序号在node_feature中找不到
            continue
        edges.append((texId2index[id_1], texId2index[id_2]))#降维到了184×2
        labels.append(label)#边做边 标签当标签 唯一区别
    x = torch.tensor(x, dtype=torch.float32) #68419 × 42
    d_edges = np.array(edges)
    edges = torch.tensor(d_edges.T, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)

    data = Data(x=x, edge_index=edges, edge_label=labels, edge_label_index=edges)#这一步有点像MIG配合42Dim的代码
    return data, names


if __name__ == '__main__':
    load_10_fold_data()
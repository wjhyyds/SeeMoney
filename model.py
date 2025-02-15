import torch
from torch_geometric.nn import GCNConv,GATConv, SAGEConv, JumpingKnowledge
from torch.nn import Linear
class AdvanceGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.sage = SAGEConv(in_channels, hidden_channels)
        self.gat = GATConv(hidden_channels, hidden_channels, heads=heads)
        self.jk = JumpingKnowledge(mode='cat')  # 或 'max', 'lstm'
        self.lin = Linear(hidden_channels * (heads + 1), out_channels)
        
    def encode(self, x, edge_index):
        x1 = self.sage(x, edge_index).relu()
        x2 = self.gat(x1, edge_index).relu()
        # 使用跳跃连接合并不同层的特征
        x = self.jk([x1, x2])
        x = self.lin(x)
        return x #68419x16
        
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]] #184x16
        dst = z[edge_label_index[1]] #184x16
        r = (src * dst).sum(dim=-1)  #184x1
        # print("z",z.shape)
        # print("src",src.shape)
        # print("dst",dst.shape)
        # print("r",r)
        return r
        # # 使用注意力机制计算边的预测
        # attention = torch.nn.functional.cosine_similarity(src, dst, dim=-1)
        # return attention * (src * dst).sum(dim=-1)
        
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
    
class GNN_NET(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        # self.conv1 = SAGEConv(in_channels, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
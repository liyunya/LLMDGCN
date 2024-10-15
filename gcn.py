from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        super(GCN, self).__init__()
        if num_layers == 1:
            hidden_size = out_size

        self.num_layers = num_layers
        self.dropout = dropout

        self.gnn_layers = nn.ModuleList([GCNConv(in_size, hidden_size)])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.gnn_layers.append(GCNConv(hidden_size, out_size))
            else:
                self.gnn_layers.append(GCNConv(hidden_size, hidden_size))

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature, edge_index):
        h = feature
        for i, layer in enumerate(self.gnn_layers):
            if i == self.num_layers - 1:
                h = layer(h, edge_index)
            else:
                h_ = layer(h, edge_index)
                h = F.relu(h_)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return F.softmax(h, dim=1)


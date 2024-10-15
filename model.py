# GNN/model.py

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class JosephusGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(JosephusGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        x = pyg_nn.global_mean_pool(x, data.batch)
        
        x = self.fc(x)
        return x.squeeze(-1)  # 确保输出是一维的
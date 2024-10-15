# GNN/dataset.py

import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np

class JosephusDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        N, M, Result = row['N'], row['M'], row['Result']
        
        # Create graph
        x = torch.tensor([[n, M] for n in range(N)], dtype=torch.float)
        edge_index = self._create_edges(N, M)
        
        # 确保标签是一个标量
        y = torch.tensor(Result, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def _create_edges(self, N, M):
        edges = []
        for i in range(N):
            for j in range(1, M+1):
                edges.append([i, (i+j)%N])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
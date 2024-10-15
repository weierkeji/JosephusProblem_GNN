# inference.py

import torch
from model import JosephusGNN
from dataset import JosephusDataset
from torch_geometric.data import Data

def load_model(model_path):
    model = JosephusGNN(input_dim=2, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, N, M):
    x = torch.tensor([[n, M] for n in range(N)], dtype=torch.float)
    edge_index = JosephusDataset._create_edges(None, N, M)
    data = Data(x=x, edge_index=edge_index)
    
    with torch.no_grad():
        output = model(data)
    
    return output.item()

def main():
    model = load_model('josephus_model.pth')
    
    while True:
        try:
            N = int(input("Enter N (number of people): "))
            M = int(input("Enter M (skip count): "))
            
            result = predict(model, N, M)
            print(f"Predicted survivor: {round(result)}")
        except ValueError:
            print("Invalid input. Please enter integers.")
        
        cont = input("Continue? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()
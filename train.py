# GNN/train.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader
from model import JosephusGNN
from dataset import JosephusDataset
from utils import train_epoch, validate
import torch.optim as optim
from tqdm import tqdm

def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"Running on rank {rank} out of {world_size} processes.")

    # Load dataset
    train_dataset = JosephusDataset('josephus_train.csv')
    val_dataset = JosephusDataset('josephus_test.csv')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

    if rank == 0:
        print(f"训练集大小: {len(train_dataset)} 个图")
        print(f"验证集大小: {len(val_dataset)} 个图")

    # Initialize model
    model = JosephusGNN(input_dim=2, hidden_dim=64, output_dim=1)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)  # 将损失函数也移到GPU

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, MSE: {train_metrics['MSE']:.4f}, MAE: {train_metrics['MAE']:.4f}, R2: {train_metrics['R2']:.4f}, Accuracy: {train_metrics['Accuracy']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MSE: {val_metrics['MSE']:.4f}, MAE: {val_metrics['MAE']:.4f}, R2: {val_metrics['R2']:.4f}, Accuracy: {val_metrics['Accuracy']:.4f}")
            print("-----------------------------")

    # Final evaluation
    final_train_loss, final_train_metrics = validate(model, train_loader, criterion, device)
    final_val_loss, final_val_metrics = validate(model, val_loader, criterion, device)

    if rank == 0:
        print("Final Results:")
        print(f"Train Loss: {final_train_loss:.4f}, MSE: {final_train_metrics['MSE']:.4f}, MAE: {final_train_metrics['MAE']:.4f}, R2: {final_train_metrics['R2']:.4f}, Accuracy: {final_train_metrics['Accuracy']:.4f}")
        print(f"Val Loss: {final_val_loss:.4f}, MSE: {final_val_metrics['MSE']:.4f}, MAE: {final_val_metrics['MAE']:.4f}, R2: {final_val_metrics['R2']:.4f}, Accuracy: {final_val_metrics['Accuracy']:.4f}")
        torch.save(model.module.state_dict(), 'josephus_model.pth')

    cleanup()

if __name__ == "__main__":
    main()
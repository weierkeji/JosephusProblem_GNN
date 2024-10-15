# GNN/utils.py

import torch
import torch.distributed as dist
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    if dist.get_rank() == 0:
        pbar = tqdm(total=len(loader), desc="Training")
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        
        reduced_loss = reduce_tensor(loss.detach(), world_size())
        total_loss += reduced_loss.item() * data.num_graphs
        
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(data.y.cpu().numpy())
        
        if dist.get_rank() == 0:
            pbar.update(1)
    
    if dist.get_rank() == 0:
        pbar.close()
    
    all_preds = gather_tensor(torch.tensor(all_preds, device=device))
    all_targets = gather_tensor(torch.tensor(all_targets, device=device))
    
    metrics = calculate_metrics(all_preds.cpu(), all_targets.cpu())
    return total_loss / len(loader.dataset), metrics

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    if dist.get_rank() == 0:
        pbar = tqdm(total=len(loader), desc="Validating")
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            
            reduced_loss = reduce_tensor(loss.detach(), world_size())
            total_loss += reduced_loss.item() * data.num_graphs
            
            all_preds.extend(output.detach().cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
            
            if dist.get_rank() == 0:
                pbar.update(1)
    
    if dist.get_rank() == 0:
        pbar.close()
    
    all_preds = gather_tensor(torch.tensor(all_preds, device=device))
    all_targets = gather_tensor(torch.tensor(all_targets, device=device))
    
    metrics = calculate_metrics(all_preds.cpu(), all_targets.cpu())
    return total_loss / len(loader.dataset), metrics

def calculate_metrics(preds, targets):
    preds = preds.numpy()
    targets = targets.numpy()
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    accuracy = (preds.round() == targets).mean()
    return {'MSE': mse, 'MAE': mae, 'R2': r2, 'Accuracy': accuracy}

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    output_tensors = [torch.zeros_like(tensor, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)

def world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
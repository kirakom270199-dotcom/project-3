import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        total_mass = batch['total_mass'].to(device)
        calories = batch['calories'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred_calories = model(images, input_ids, attention_mask, total_mass)
        loss = criterion(pred_calories, calories)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        all_preds.extend(pred_calories.detach().cpu().numpy())
        all_targets.extend(calories.cpu().numpy())

        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return epoch_loss, mae, rmse

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    dish_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move data to device
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            total_mass = batch['total_mass'].to(device)
            calories = batch['calories'].to(device)

            # Forward pass
            pred_calories = model(images, input_ids, attention_mask, total_mass)
            loss = criterion(pred_calories, calories)

            # Update metrics
            running_loss += loss.item()
            all_preds.extend(pred_calories.cpu().numpy())
            all_targets.extend(calories.cpu().numpy())
            dish_ids.extend(batch['dish_id'])

    epoch_loss = running_loss / len(dataloader)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return epoch_loss, mae, rmse, all_preds, all_targets, dish_ids

def plot_training_history(train_losses, val_losses, train_mae, val_mae, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot MAE
    axes[1].plot(train_mae, label='Train MAE')
    axes[1].plot(val_mae, label='Val MAE')
    axes[1].set_title('Training and Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_model(model, path):
    """Save model state dict"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """Load model state dict"""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model
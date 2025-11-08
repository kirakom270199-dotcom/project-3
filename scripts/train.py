import torch
import torch.nn as nn
import os
import json
from datetime import datetime
from .config import Config
from .dataset import create_data_loaders
from .model import FoodCaloriePredictor
from .utils import (
    set_seed, train_epoch, validate_epoch,
    plot_training_history, save_model
)

def train(config_path=None):
    """Main training function"""

    # Load configuration
    if config_path:
        # Load from JSON config file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
    else:
        # Use default config
        config = Config()

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Create output directory
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Load data
    import pandas as pd
    df_dish = pd.read_csv(config.DISH_CSV_PATH)
    df_ingredients = pd.read_csv(config.INGREDIENTS_CSV_PATH)

    print(f"Loaded {len(df_dish)} dishes and {len(df_ingredients)} ingredients")

    # Create data loaders
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
        df_dish, df_ingredients, config
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = FoodCaloriePredictor(config).to(config.DEVICE)

    # Loss and optimizer
    criterion = nn.L1Loss()  # MAE loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    # Training history
    train_losses = []
    val_losses = []
    train_mae_values = []
    val_mae_values = []

    best_val_mae = float('inf')
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # Train
        train_loss, train_mae, train_rmse = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, scheduler
        )

        # Validate
        val_loss, val_mae, val_rmse, _, _, _ = validate_epoch(
            model, test_loader, criterion, config.DEVICE
        )

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mae_values.append(train_mae)
        val_mae_values.append(val_mae)

        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_model(model, config.MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"New best model saved with Val MAE: {val_mae:.2f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break

        # Unfreeze encoders after some epochs
        if epoch == 10:
            print("Unfreezing encoders for fine-tuning...")
            model.unfreeze_encoders()

    # Plot training history
    plot_training_history(
        train_losses, val_losses,
        train_mae_values, val_mae_values,
        save_path=os.path.join(config.LOG_DIR, 'training_history.png')
    )

    print(f"\nTraining completed!")
    print(f"Best Val MAE: {best_val_mae:.2f}")

    # Save training results
    results = {
        'best_val_mae': best_val_mae,
        'final_train_mae': train_mae_values[-1],
        'final_val_mae': val_mae_values[-1],
        'epochs_trained': len(train_losses),
        'config': config.to_dict()
    }

    results_path = os.path.join(config.LOG_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return model, results

if __name__ == "__main__":
    # Train with default config
    model, results = train()
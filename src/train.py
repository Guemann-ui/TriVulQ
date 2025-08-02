"""
Model training and evaluation functions
"""
import time
import copy
import torch
import logging
import numpy as np
from torch.optim import Adam
from config import MODEL_SAVE_PATH, USE_AMP, LEARNING_RATE, DEVICE


def train_epoch(model, loader, optimizer, criterion, scaler):
    """
    Train model for one epoch

    Args:
        model: Model instance
        loader: Data loader
        optimizer: Optimizer
        criterion: Loss function
        scaler: GradScaler for AMP

    Returns:
        epoch_loss: Average loss for epoch
        epoch_acc: Average accuracy for epoch
    """
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    total_batches = len(loader)

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # Automatic Mixed Precision context
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(input_ids)
            loss = criterion(outputs, labels)

        # Backpropagation with scaler for AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        epoch_loss += loss.item()
        epoch_acc += acc

        # Log progress
        if (i + 1) % 100 == 0 or (i + 1) == total_batches:
            logging.info(f"Batch {i + 1}/{total_batches} - Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    return epoch_loss / total_batches, epoch_acc / total_batches


def evaluate(model, loader, criterion):
    """
    Evaluate model performance

    Args:
        model: Model instance
        loader: Data loader
        criterion: Loss function

    Returns:
        epoch_loss: Average loss
        epoch_acc: Average accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    # Concatenate all batches
    all_preds = np.vstack(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    return epoch_loss / len(loader), epoch_acc / len(loader), all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, epochs):
    """
    Full training loop with validation

    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        epochs: Number of epochs

    Returns:
        best_model: Best model state dict
        best_val_loss: Best validation loss
    """
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        # Validation phase
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        epoch_time = time.time() - start_time

        # Log epoch results
        logging.info(f"Epoch {epoch + 1}/{epochs} | "
                     f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                     f"Time: {epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Validation loss improved to {val_loss:.4f}, model saved")

    return best_model, best_val_loss

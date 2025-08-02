"""
Main execution script
"""
import warnings

warnings.filterwarnings('ignore')

import logging
import torch
import torch.nn as nn
import os
from clang.cindex import Config

from config import (DATASET_NAME, MODEL_SAVE_PATH, TEST_ONLY, DEVICE, MULTIGPU,
                    MODEL_TYPE, EPOCHS, CLASS_WEIGHTS, CLANG_LIBRARY_PATH)
from data_loader import create_dataloaders
from model import VulBERTaCNN
from train import train_model, evaluate
from utils import (setup_logging, set_seeds, evaluate_predictions,
                   compute_class_weights)


def load_model(model, path, multi_gpu=False):
    """Load model state dict with multi-GPU handling"""
    try:
        state_dict = torch.load(path, map_location=DEVICE)

        # Handle multi-GPU to single-GPU conversion
        if not multi_gpu and any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        return False


def main():
    # Setup environment
    set_seeds()
    setup_logging()
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Multi-GPU: {MULTIGPU}, Test-only mode: {TEST_ONLY}")
    logging.info(f"Model type: {MODEL_TYPE}, Epochs: {EPOCHS if not TEST_ONLY else 'N/A'}")

    # Configure Clang
    if CLANG_LIBRARY_PATH and os.path.exists(CLANG_LIBRARY_PATH):
        Config.set_library_file(CLANG_LIBRARY_PATH)
        logging.info(f"Using Clang library at: {CLANG_LIBRARY_PATH}")
    else:
        logging.warning("Clang library path not configured or invalid")

    # Create dataloaders
    if TEST_ONLY:
        _, _, test_loader = create_dataloaders()
        train_loader = val_loader = None
    else:
        train_loader, val_loader, test_loader = create_dataloaders()

    # Initialize model
    model = VulBERTaCNN().to(DEVICE)
    if MULTIGPU:
        model = nn.DataParallel(model)
        # Save multi-GPU models separately
        model_save_path = f"{MODEL_SAVE_PATH}_multigpu"
    else:
        model_save_path = MODEL_SAVE_PATH

    logging.info(
        f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # Prepare loss function with class weighting
    criterion = nn.CrossEntropyLoss()
    if CLASS_WEIGHTS and not TEST_ONLY and train_loader:
        try:
            # Get labels directly from dataset
            all_labels = train_loader.dataset.labels
            class_weights = compute_class_weights(all_labels).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logging.info(f"Using class weights: {class_weights.cpu().numpy()}")
        except AttributeError:
            logging.warning("Couldn't access dataset labels directly, using default class weights")

    # Training or evaluation mode
    if not TEST_ONLY:
        logging.info("Starting training...")
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            EPOCHS
        )
        logging.info("Training completed")
    else:
        logging.info("Skipping training (TEST_ONLY mode)")

    # Evaluation
    logging.info("Starting evaluation...")
    if TEST_ONLY:
        # Load the best saved model
        success = load_model(model, MODEL_SAVE_PATH, multi_gpu=MULTIGPU)
        if not success:
            logging.warning("Using randomly initialized model for evaluation")

    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)
    logging.info(f"\n{'=' * 50}")
    logging.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Detailed evaluation metrics with safe access
    metrics = evaluate_predictions(all_preds, all_labels)
    print(metrics)

    # Safe logging of metrics
    logging.info("\nDetailed Metrics:")
    if 'report' in metrics:
        logging.info(f"Classification Report:\n{metrics['report']}")
    else:
        logging.info("Classification report not available")

    if 'confusion_matrix' in metrics:
        logging.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    else:
        logging.info("Confusion matrix not available")
    logging.info("Evaluation completed!!")


if __name__ == "__main__":
    main()

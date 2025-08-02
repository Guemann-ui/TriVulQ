"""
Utility functions for data processing, evaluation, and logging
"""
import re
import logging
import sklearn.metrics
import torch
import numpy as np
from config import SEED, CLANG_LIBRARY_PATH, DEVICE, LOG_FILE

# Configure Clang
import clang.cindex
clang.cindex.Config.set_library_file(CLANG_LIBRARY_PATH)

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized")

def clean_code(code: str) -> str:
    """
    Remove comments and unnecessary whitespace from code

    Args:
        code: Source code string

    Returns:
        Cleaned code string
    """
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    return re.sub(pat, '', code).replace('\n', '').replace('\t', '')

def softmax_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate accuracy from softmax probabilities

    Args:
        probs: Tensor of class probabilities
        labels: True class labels

    Returns:
        Accuracy score
    """
    predictions = probs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / len(labels)

def evaluate_predictions(all_pred: np.ndarray, all_labels: np.ndarray):
    """
    Calculate and print evaluation metrics

    Args:
        all_pred: Array of prediction probabilities (2D: [samples, classes])
        all_labels: Array of true labels (1D)
    """
    predictions = np.argmax(all_pred, axis=1)
    probs = all_pred[:, 1]  # Positive class probabilities

    confusion = sklearn.metrics.confusion_matrix(all_labels, predictions)
    logging.info('Confusion matrix:\n%s', confusion)

    try:
        tn, fp, fn, tp = confusion.ravel()
        metrics = {
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Accuracy': sklearn.metrics.accuracy_score(all_labels, predictions),
            'Precision': sklearn.metrics.precision_score(all_labels, predictions),
            'Recall': sklearn.metrics.recall_score(all_labels, predictions),
            'F1': sklearn.metrics.f1_score(all_labels, predictions),
            'AP': sklearn.metrics.average_precision_score(all_labels, probs),
            'AUC': sklearn.metrics.roc_auc_score(all_labels, probs),
            'MCC': sklearn.metrics.matthews_corrcoef(all_labels, predictions)
        }

        for name, value in metrics.items():
            logging.info(f'{name}: {value:.4f}' if isinstance(value, float) else f'{name}: {value}')

        return metrics
    except ValueError as e:
        logging.error("Evaluation failed - check label distribution: %s", e)
        return {}

def set_seeds():
    """Set all random seeds for reproducibility"""
    import os
    import random
    import torch
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info("Seeds set for reproducibility")

def compute_class_weights(labels):
    """Compute class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(labels), y=labels
    )
    logging.info(f"Class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)

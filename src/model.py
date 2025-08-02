"""
VulBERTa-CNN model definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from config import EMBED_DIM, PAD_IDX, EMBED_SIZE, VULBERTA_PATH, DROPOUT_RATE
import logging


class VulBERTaCNN(nn.Module):
    """VulBERTa-CNN model architecture"""

    def __init__(self):
        super().__init__()
        try:
            # Load pre-trained embeddings
            pretrained = RobertaModel.from_pretrained(VULBERTA_PATH)
            self.embed = nn.Embedding.from_pretrained(
                pretrained.embeddings.word_embeddings.weight,
                freeze=True,
                padding_idx=PAD_IDX
            )

            # Convolutional layers
            self.conv1 = nn.Conv1d(EMBED_DIM, 200, 3)
            self.conv2 = nn.Conv1d(EMBED_DIM, 200, 4)
            self.conv3 = nn.Conv1d(EMBED_DIM, 200, 5)

            # Fully connected layers
            self.dropout = nn.Dropout(DROPOUT_RATE)
            self.fc1 = nn.Linear(600, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 2)

            # Initialize padding
            self.embed.weight.data[PAD_IDX] = torch.zeros(EMBED_DIM)
            logging.info("VulBERTa-CNN model initialized")
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise

    def forward(self, input_ids):
        """Forward pass"""
        # Embedding layer
        x = self.embed(input_ids)

        # Permute for convolutional layers: [batch, channels, seq_len]
        x = x.permute(0, 2, 1)

        # Convolutional layers with ReLU and max pooling
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, kernel_size=x1.size(2)).squeeze(2)

        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, kernel_size=x2.size(2)).squeeze(2)

        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, kernel_size=x3.size(2)).squeeze(2)

        # Concatenate features
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

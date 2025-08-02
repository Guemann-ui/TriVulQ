"""
Data loading and preprocessing utilities for VulBERTa-CNN
Replicates the tokenization and dataset handling from the original notebook
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, normalizers, processors
from tokenizers.models import BPE
from tokenizers.normalizers import StripAccents, Replace
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import NormalizedString, PreTokenizedString
from clang.cindex import Index, TokenKind, Config
from typing import List, Tuple, Optional
from config import (DATA_PATH, DATASET_NAME, TEST_ONLY, PAD_IDX,
                    TOKENIZER_PATH, BATCH_SIZE, DEVICE,
                    CLANG_LIBRARY_PATH)

# Configure Clang
try:
    Config.set_library_file(CLANG_LIBRARY_PATH)
except Exception as e:
    logging.warning(f"Could not set Clang library path: {e}")


def clean_code(code: str) -> str:
    """
    Remove comments and unnecessary whitespace from code
    Replicates the cleaner function from the notebook

    Args:
        code: Source code string

    Returns:
        Cleaned code string
    """
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    return re.sub(pat, '', code).replace('\n', '').replace('\t', '')


class ClangTokenizer:
    """Custom Clang tokenizer from the VulBERTa notebook"""

    def __init__(self):
        self.index = Index.create()

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """Tokenize using Clang as in notebook"""
        tokens = []
        try:
            tu = self.index.parse('tmp.c',
                                  args=[''],
                                  unsaved_files=[('tmp.c', str(normalized_string.original))],
                                  options=0)
            for t in tu.get_tokens(extent=tu.cursor.extent):
                spelling = t.spelling.strip()
                if spelling == '':
                    continue
                tokens.append(NormalizedString(spelling))
        except Exception as e:
            logging.error(f"Clang tokenization failed: {e}")
            tokens = [normalized_string]  # Fallback to original
        return tokens

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)


def create_tokenizer() -> Tokenizer:
    """
    Create tokenizer exactly as in the VulBERTa notebook

    Returns:
        Configured tokenizer instance
    """
    try:
        # Verify tokenizer files exist
        vocab_path = os.path.join(TOKENIZER_PATH, "drapgh-vocab.json")
        merges_path = os.path.join(TOKENIZER_PATH, "drapgh-merges.txt")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"Merges file not found: {merges_path}")

        # Load vocabulary and merges
        vocab, merges = BPE.read_file(vocab=vocab_path, merges=merges_path)

        # Create tokenizer
        tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

        # Configure normalizer
        tokenizer.normalizer = normalizers.Sequence([
            StripAccents(),
            Replace(" ", "Ä")
        ])

        # Add custom pre-tokenizer
        tokenizer.pre_tokenizer = PreTokenizer.custom(ClangTokenizer())

        # Configure post-processor
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>", 0),
                ("<pad>", PAD_IDX),
                ("</s>", 2),
                ("<unk>", 3),
                ("<mask>", 4)
            ]
        )

        # Enable truncation and padding
        tokenizer.enable_truncation(max_length=1024)
        tokenizer.enable_padding(
            direction='right',
            pad_id=PAD_IDX,
            pad_token='<pad>'
        )

        logging.info("Tokenizer created successfully")
        return tokenizer
    except Exception as e:
        logging.error(f"Error creating tokenizer: {e}")
        raise


class CodeDataset(Dataset):
    """PyTorch Dataset for code classification"""

    def __init__(self, encodings: list, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            'input_ids': torch.tensor(self.encodings[idx].ids, dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings[idx].attention_mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_datasets(tokenizer: Tokenizer) -> Tuple:
    """
    Load and preprocess datasets based on dataset name
    Replicates the dataset handling from the notebook

    Args:
        tokenizer: Configured tokenizer instance

    Returns:
        Tuple of (train_data, val_data, test_data) or test_data only
    """

    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across datasets"""
        if 'functionSource' in df.columns:
            df = df.rename(columns={'functionSource': 'code'})
        elif 'func' in df.columns:
            df = df.rename(columns={'func': 'code'})
        elif 'function' in df.columns:
            df = df.rename(columns={'function': 'code'})
        return df

    def get_labels(df: pd.DataFrame) -> np.ndarray:
        """Extract labels from dataframe"""
        if 'target' in df.columns:
            return df['target'].values
        elif 'combine' in df.columns:
            return (df['combine'] * 1).values
        elif 'label' in df.columns:
            return df['label'].values
        else:
            raise ValueError("Label column not found")

    try:
        # Devign Dataset
        if DATASET_NAME == 'devign':
            base_path = os.path.join(DATA_PATH, 'finetune/devign')
            full_df = pd.read_json(os.path.join(base_path, 'Devign.json'))
            full_df = standardize_columns(full_df)

            if TEST_ONLY:
                with open(os.path.join(base_path, 'test.txt')) as f:
                    test_idx = [int(line.strip()) for line in f]
                test_df = full_df.iloc[test_idx]
                test_labels = get_labels(test_df)
                test_df['code'] = test_df['code'].apply(clean_code)
                test_encodings = tokenizer.encode_batch(test_df['code'].tolist())
                return None, None, (test_encodings, test_labels)
            else:
                with open(os.path.join(base_path, 'train.txt')) as f:
                    train_idx = [int(line.strip()) for line in f]
                with open(os.path.join(base_path, 'valid.txt')) as f:
                    val_idx = [int(line.strip()) for line in f]
                with open(os.path.join(base_path, 'test.txt')) as f:
                    test_idx = [int(line.strip()) for line in f]

                train_df = full_df.iloc[train_idx]
                val_df = full_df.iloc[val_idx]
                test_df = full_df.iloc[test_idx]

                # Clean and tokenize
                for df in [train_df, val_df, test_df]:
                    df['code'] = df['code'].apply(clean_code)

                train_encodings = tokenizer.encode_batch(train_df['code'].tolist())
                val_encodings = tokenizer.encode_batch(val_df['code'].tolist())
                test_encodings = tokenizer.encode_batch(test_df['code'].tolist())

                return (
                    (train_encodings, get_labels(train_df)),
                    (val_encodings, get_labels(val_df)),
                    (test_encodings, get_labels(test_df))
                )

        # D2A Dataset
        elif DATASET_NAME == 'd2a':
            task = 'function'
            base_path = os.path.join(DATA_PATH, f'finetune/{DATASET_NAME}/{task}')

            if TEST_ONLY:
                test_file = os.path.join(base_path, f'd2a_lbv1_{task}_dev.csv')
                test_df = pd.read_csv(test_file)
                test_df = standardize_columns(test_df)
                test_df['code'] = test_df['code'].apply(clean_code)
                test_encodings = tokenizer.encode_batch(test_df['code'].tolist())
                test_labels = get_labels(test_df)
                return None, None, (test_encodings, test_labels)
            else:
                train_df = pd.read_csv(os.path.join(base_path, f'd2a_lbv1_{task}_train.csv'))
                val_df = pd.read_csv(os.path.join(base_path, f'd2a_lbv1_{task}_dev.csv'))
                test_df = pd.read_csv(os.path.join(base_path, f'd2a_lbv1_{task}_test.csv'))

                train_df = standardize_columns(train_df)
                val_df = standardize_columns(val_df)
                test_df = standardize_columns(test_df)

                # Clean and tokenize
                for df in [train_df, val_df, test_df]:
                    df['code'] = df['code'].apply(clean_code)

                train_encodings = tokenizer.encode_batch(train_df['code'].tolist())
                val_encodings = tokenizer.encode_batch(val_df['code'].tolist())
                test_encodings = tokenizer.encode_batch(test_df['code'].tolist())

                return (
                    (train_encodings, get_labels(train_df)),
                    (val_encodings, get_labels(val_df)),
                    (test_encodings, get_labels(test_df))
                )

        # Draper Dataset
        elif DATASET_NAME == 'draper':
            base_path = os.path.join(DATA_PATH, 'finetune/draper')

            if TEST_ONLY:
                test_file = os.path.join(base_path, 'draper_test.pkl')
                test_df = pd.read_pickle(test_file)
                test_df = standardize_columns(test_df)
                test_df['code'] = test_df['code'].apply(clean_code)
                test_encodings = tokenizer.encode_batch(test_df['code'].tolist())
                test_labels = get_labels(test_df)
                return None, None, (test_encodings, test_labels)
            else:
                train_df = pd.read_pickle(os.path.join(base_path, 'draper_train.pkl'))
                val_df = pd.read_pickle(os.path.join(base_path, 'draper_val.pkl'))
                test_df = pd.read_pickle(os.path.join(base_path, 'draper_test.pkl'))

                train_df = standardize_columns(train_df)
                val_df = standardize_columns(val_df)
                test_df = standardize_columns(test_df)

                # Clean and tokenize
                for df in [train_df, val_df, test_df]:
                    df['code'] = df['code'].apply(clean_code)

                train_encodings = tokenizer.encode_batch(train_df['code'].tolist())
                val_encodings = tokenizer.encode_batch(val_df['code'].tolist())
                test_encodings = tokenizer.encode_batch(test_df['code'].tolist())

                return (
                    (train_encodings, get_labels(train_df)),
                    (val_encodings, get_labels(val_df)),
                    (test_encodings, get_labels(test_df))
                )

        else:
            raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    except Exception as e:
        logging.error(f"Error loading {DATASET_NAME} dataset: {e}")
        raise


def create_dataloaders() -> Tuple[Optional[DataLoader], Optional[DataLoader], DataLoader]:
    """
    Create DataLoader instances for train, validation, and test sets
    Replicates the data iterator functionality from the notebook

    Returns:
        Tuple of (train_loader, val_loader, test_loader) or (None, None, test_loader)
    """
    try:
        tokenizer = create_tokenizer()
        datasets = load_datasets(tokenizer)

        def _create_dataloader(encodings: list, labels: list, shuffle: bool = False) -> DataLoader:
            """Helper to create DataLoader from encodings and labels"""
            if encodings is None or labels is None:
                return None
            dataset = CodeDataset(encodings, labels)
            return DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=shuffle,
                num_workers=0,  # Required for Windows compatibility
                pin_memory=DEVICE.type == 'cuda'
            )

        if TEST_ONLY:
            _, _, test_data = datasets
            test_loader = _create_dataloader(*test_data)
            return None, None, test_loader
        else:
            train_data, val_data, test_data = datasets
            train_loader = _create_dataloader(*train_data, shuffle=True)
            val_loader = _create_dataloader(*val_data)
            test_loader = _create_dataloader(*test_data)
            return train_loader, val_loader, test_loader

    except Exception as e:
        logging.error(f"Error creating data loaders: {e}")
        raise

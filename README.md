# TriVulQ
This repository implements a CNN-based fine-tuning approach for VulBERTa, a pre-trained language model for vulnerability detection in source code. Based on the original [VulBERTa](https://github.com/ICL-ml4csec/VulBERTa) research.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Supported Dataset](#data)
- [Installation](#installation)
- [Fine-tuning CNN](#fine-tuning-cnn)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Prerequisites

1. System Requirements:
   - Linux (recommended Ubuntu 20.04+)
   - NVIDIA GPU with ≥12GB VRAM
   - ≥32GB RAM
   - Python 3.8+

2. Software Dependencies:
   - CUDA 11.7+
   - cuDNN 8.5+
   - libclang (LLVM 16+)

3. Accounts:
   - [Hugging Face Account](https://huggingface.co/join) (for model access)
   - [OSF Account](https://osf.io/) (for dataset access)

## Supported Dataset

| Dataset | Language | Samples | Vuln. Rate | Granularity | Special Characteristics |
|---------|----------|---------|------------|-------------|-------------------------|
| **Devign** | C/C++ | 27,546 | 33.2% | Function-level | Balanced vulnerability distribution |
| **Draper** | C/C++ | 70,874 | 15.8% | Function-level | Contains synthetic vulnerabilities, highly imbalanced |
| **D2A** | C/C++ | 1.2M+ | 5-10% | Commit-level | Real-world vulnerabilities from GitHub, large-scale |

### Key Characteristics

1. **Devign**:
   - Curated from 4 open-source projects (QEMU, FFmpeg, Wireshark, Linux Debian)
   - Manually verified vulnerabilities
   - Balanced classes suitable for baseline models

2. **Draper**:
   - Contains both real and synthetic vulnerabilities
   - Highly imbalanced (recommend using class weighting)
   - Focuses on specific vulnerability patterns

3. **D2A**:
   - Automatically labeled from GitHub commits
   - Contains vulnerability type labels (CWE IDs)
   - Requires more preprocessing but larger scale

## Installation

### 1. Clone Repository
```
git clone https://github.com/Guemann-ui/TriVulQ.git
cd TriVulQ
```
### 2. Install Python Dependencies
```
$pip install -r requirements.txt
```
### 3. Setup libclang
```
sudo apt-get update
sudo apt-get install llvm-18 clang
echo "export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```
## Fine-tuning CNN
```
$python3 main.py
```

## Repository Structure

```
TriVulQ/
├── configs/ # Configuration files
│ ├── base_config.py # Base configuration
│ ├── draper_config.py # Draper dataset config
│ ├── devign_config.py # Devign dataset config
│ └── d2a_config.py # D2A dataset config
│
├── data/ # Datasets (git-ignored)
│ ├── raw/ # Raw JSONL files
│ ├── processed/ # Preprocessed data
│ └── final/ # Train/val/test splits
│
├── saved_models/ # Trained models (git-ignored)
│ └── VB-CNN_devign.pt
│
├── scripts/ # Utility scripts
│ ├── download_model.py # Download pretrained models
│ └── download_data.py # Download datasets
│
├── src/ # Core source code
│ ├── data_loader.py # Data loading and processing
│ ├── model.py # VulBERTa-CNN implementation
│ ├── trainer.py # Training logic
│ ├── evaluator.py # Evaluation metrics
│ ├── preprocess.py # Data preprocessing
│ ├── split_data.py # Dataset splitting
│ └── main.py # Entry point
│
├── requirements.txt # Python dependencies
├── LICENSE
└── README.md # This documentation
```


@article{chiu2021vulberta,
  title={VulBERTa: A Source Code Based Pre-Trained Language Model for Vulnerability Detection},
  author={Chiu, Kelvin and Li, Yi and He, Yujia and Zhang, Meng and Rubin, Julia},
  journal={arXiv preprint arXiv:2104.12408},
  year={2021}
}
# ALQAC 2025 - Legal Question Answering System

Welcome to the ALQAC 2025 (Automated Legal Question Answering Challenge) repository by Team Hallucineers. This project features a comprehensive legal question answering system with advanced retrieval and reasoning capabilities.

## Project Overview

Our solution consists of two main components:
1. **Retrieval System**: A sophisticated document retrieval pipeline that efficiently finds relevant legal documents
2. **Question Answering System**: A language model-based system that generates accurate answers with detailed legal reasoning

## System Architecture

### 1. Retrieval System

Our retrieval system implements a two-stage approach:
- **First Stage**: BM25 for initial document retrieval
- **Second Stage**: Sentence transformers and rerankers for precise document ranking

#### Key Features:
- Hybrid BM25 + dense retrieval
- Custom-trained legal document embeddings
- Advanced reranking with specialized legal models

### 2. Question Answering System

Our QA system uses Chain-of-Thought (CoT) reasoning to provide well-justified answers across multiple question types:
- True/False questions
- Multiple-choice questions
- Free-text questions

#### Key Features:
- Supervised Fine-Tuning (SFT) on legal domain data
- Generalized Preference Ranking Optimization (GPRO)
- Ensemble methods for improved accuracy
- Detailed step-by-step legal reasoning

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Required Python packages (install via `pip install -r requirements.txt`)

### Data Structure

```
ALQAC_2025_data/
├── main_data/           # Main competition data
│   ├── alqac25_law.json
│   ├── alqac25_train.json
│   └── ...
└── additional_data/
    └── zalo/           # Additional Zalo data
```

## Getting Started

### 1. Retrieval System

#### Data Processing
```bash
cd retrieval
python process_data.py
```

#### BM25 Training
```bash
python bm25_train.py
```

#### Fine-tuning Rerankers
```bash
# Install FlagEmbedding
pip install -U FlagEmbedding[finetune]

# Train reranker
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    --model_name_or_path AITeamVN/Vietnamese_Reranker \
    --train_data ./pair_data/rerank_data_top20.jsonl \
    --output_dir ./legal-reranker \
    --learning_rate 6e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4
```

### 2. Question Answering System

#### Data Preparation
Run the notebook to generate training data with Chain-of-Thought reasoning:
```bash
jupyter notebook alqac-gen-data-train.ipynb
```

#### Model Training
1. **SFT Training**:
```bash
jupyter notebook alqac-train-data-train-qwen-fulldata.ipynb
```

2. **GPRO Training**:
```bash
jupyter notebook alqac-train-grpo-config1.ipynb
jupyter notebook alqac-train-grpo-config2.ipynb
jupyter notebook alqac-train-grpo-config3.ipynb
```

#### Inference
Generate predictions using the trained models:
```bash
# Using SFT model
jupyter notebook alqac-inference-test-qwen-fulldata.ipynb

# Using GPRO models
jupyter notebook alqac-inference-test-qwen-grpo-fulldata.ipynb
```

## Evaluation

### Retrieval Evaluation
```bash
cd retrieval
python eval.py \
    --raw_data ../ALQAC_2025_data \
    --hybrid \
    --combine_type weighted_sum \
    --rerank_range_score 0.5 \
    --retrieve_range_scores 0.05 \
    --alpha 0.3 \
    --eval_size 0.2
```

### QA Evaluation
Run the submission notebook to generate final predictions:
```bash
jupyter notebook submit.ipynb
```

## Acknowledgments

- ALQAC 2025 organizers for the challenge and dataset
- Hugging Face for the Transformers library
- The open-source community for valuable tools and models

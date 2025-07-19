#!/bin/bash
# filepath: run_eval.sh

# Make script executable
chmod +x run_eval.sh

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Basic evaluation with traditional cosine similarity
echo "=== Running Basic Evaluation ==="
python eval.py \
    --raw_data "ALQAC_2025_data" \
    --saved_model "saved_model" \
    --bm25_path "saved_model/bm25_Plus_04_06_model_full_manual_stopword" \
    --legal_data "saved_model/doc_refers_saved" \
    --range-score 2.6 \
    --fixed-score 2.6 \
    --eval_size 0.2 \
    --combine-type "weighted_sum" \
    --model_1_weight 0.5 \
    --model_2_weight 0.5 \
    --model_3_weight 0.0 \
    --hybrid \
    --alpha 0.5

echo -e "\n=== Building FAISS Index ==="
# Build FAISS index (run once)
python eval.py \
    --raw_data "ALQAC_2025_data" \
    --saved_model "saved_model" \
    --legal_data "saved_model/doc_refers_saved" \
    --build_faiss \
    --faiss_index_type "Flat" \
    --faiss_path "faiss_indices"

echo -e "\n=== Running FAISS Evaluation ==="
# Evaluation with FAISS
python eval.py \
    --raw_data "ALQAC_2025_data" \
    --saved_model "saved_model" \
    --bm25_path "saved_model/bm25_Plus_04_06_model_full_manual_stopword" \
    --legal_data "saved_model/doc_refers_saved" \
    --range-score 2.6 \
    --fixed-score 2.6 \
    --eval_size 0.2 \
    --combine-type "weighted_sum" \
    --model_1_weight 0.5 \
    --model_2_weight 0.5 \
    --model_3_weight 0.0 \
    --hybrid \
    --alpha 0.5 \
    --use_faiss \
    --faiss_path "faiss_indices"

echo -e "\n=== Running FAISS Evaluation with Law Filtering ==="
# Evaluation with FAISS and law filtering
python eval.py \
    --raw_data "ALQAC_2025_data" \
    --saved_model "saved_model" \
    --bm25_path "saved_model/bm25_Plus_04_06_model_full_manual_stopword" \
    --legal_data "saved_model/doc_refers_saved" \
    --range-score 2.6 \
    --fixed-score 2.6 \
    --eval_size 0.2 \
    --combine-type "weighted_sum" \
    --model_1_weight 0.5 \
    --model_2_weight 0.5 \
    --model_3_weight 0.0 \
    --hybrid \
    --alpha 0.5 \
    --use_faiss \
    --faiss_path "faiss_indices" \
    --use_law_filter \
    --use_llm_law_filter \
    --top_laws 3

echo -e "\n=== Running Grid Search ==="
# Grid search for best parameters
python eval.py \
    --raw_data "ALQAC_2025_data" \
    --saved_model "saved_model" \
    --bm25_path "saved_model/bm25_Plus_04_06_model_full_manual_stopword" \
    --legal_data "saved_model/doc_refers_saved" \
    --eval_size 0.1 \
    --hybrid \
    --find-best-score

echo -e "\n=== All evaluations completed! ==="
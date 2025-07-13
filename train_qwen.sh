# python train_sentence_transformer.py --pretrained_model Qwen/Qwen3-Embedding-0.6B \
# --step 400 --round 2 --saved_model qwen_zalo_saved_model --pair_eval_path pair_data/bm_25_pairs_top50 \
# --pair_data_path pair_data_zalo/bm_25_pairs_top50 --batch_size 6 --data_path ALQAC_2025_data/additional_data/zalo --zalo \
# --hub_model_id phonghoccode/ALQAC_2025_Qwen3_Embedding_top50_zalo

python train_sentence_transformer.py --pretrained_model Qwen/Qwen3-Embedding-0.6B \
--step 100 --round 2 --saved_model qwen_saved_model \
--pair_data_path pair_data/bm_25_pairs_top50 --batch_size 16 --data_path ALQAC_2025_data \
--hub_model_id phonghoccode/ALQAC_2025_Qwen3_Embedding_top50
python train_sentence_transformer.py --pretrained_model AITeamVN/Vietnamese_Embedding_v2 \
--step 200 --round 1 \
--pair_data_path /kaggle/working/ALQAC-2025/pair_data/bm_25_pairs_top50 --batch_size 8 --data_path ALQAC_2025_data \
--hub_model_id phonghoccode/ALQAC_2025_Embedding_top50_round1

python train_sentence_transformer.py --pretrained_model huyydangg/DEk21_hcmute_embedding \
--step 200 --round 1 --wseg \
--pair_data_path /kaggle/working/ALQAC-2025/pair_data/bm_25_pairs_top50 --batch_size 8 --data_path ALQAC_2025_data \
--hub_model_id phonghoccode/ALQAC_2025_Embedding_top50_round1_wseg

python hard_negative_mining.py --sentence_bert_path phonghoccode/ALQAC_2025_Embedding_top50_round1 --top_k 50 --encode_legal_data
python hard_negative_mining.py --sentence_bert_path phonghoccode/ALQAC_2025_Embedding_top50_round1_wseg --top_k 50 --encode_legal_data
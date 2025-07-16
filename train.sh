python eval.py --raw_data ALQAC_2025_data --find-best-score --hybrid --reranker AITeamVN/Vietnamese_Reranker

python train_sentence_transformer.py --pretrained_model AITeamVN/Vietnamese_Embedding_v2 \
--step 200 --round 1 --saved_model sbert_saved_model --eval_size 0.0\
--pair_data_path pair_data/bm_25_pairs_top50 --batch_size 8 --data_path ALQAC_2025_data \
--hub_model_id phonghoccode/ALQAC_2025_Embedding_final

python train_sentence_transformer.py --pretrained_model huyydangg/DEk21_hcmute_embedding \
--step 200 --round 1 --wseg --saved_model sbert_wseg_saved_model --eval_size 0.0\
--pair_data_path pair_data/bm_25_pairs_top50 --batch_size 8 --data_path ALQAC_2025_data \
--hub_model_id phonghoccode/ALQAC_2025_Embedding_final_wseg
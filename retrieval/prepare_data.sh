python process_data.py
python process_data.py --data-path ../ALQAC_2025_data/additional_data/zalo --zalo
python bm25_train.py
python bm25_train.py --data-path ../ALQAC_2025_data/additional_data/zalo --zalo --save_path saved_model_zalo
python bm25_create_pairs.py --rerank
python bm25_create_pairs.py --rerank --data_path ../ALQAC_2025_data/additional_data/zalo --zalo --save_pair_path pair_data_zalo --model_path saved_model_zalo/bm25_Plus_04_06_model_full_manual_stopword --saved_model_path saved_model_zalo --eval_size 0.0
python bm25_train.py --data-path ../ALQAC_2025_data/additional_data/zalo --zalo --save_path saved_model_zalo
python bm25_create_pairs.py --data_path ../ALQAC_2025_data/additional_data/zalo --zalo --save_pair_path pair_data_zalo --model_path saved_model_zalo/bm25_Plus_04_06_model_full_manual_stopword --saved_model_path saved_model_zalo
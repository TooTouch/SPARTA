# ====================
# sqlova_data: wikisql raw
# ====================
python3 train.py --do_test --do_dev --seed 1 --bS 8 \
                 --datadir ./data/sqlova_data --logdir ./logs/sqlova_data \
                 --bert_name bert-base-uncased \
                 --max_seq_length 222 --EG \
                 --trained

# ====================
# ko_data v1: history 없음 ㅠㅠ
# ====================
python3 train.py --do_test --do_dev --seed 1 --bS 8 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token_v1 \
                 --bert_name bert-base-multilingual-cased \
                 --max_seq_length 222 --EG \
                 --trained

python3 train.py --do_test --do_dev --seed 1 --bS 8 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token_v1_copy \
                 --bert_name bert-base-multilingual-cased \
                 --max_seq_length 222 --EG \
                 --trained

# ====================
# ko_data v2
# ====================
python3 train.py --do_test --do_dev --seed 1 --bS 8 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token \
                 --bert_name bert-base-multilingual-cased \
                 --max_seq_length 222 --EG \
                 --trained


# ====================
# ko_data_not_h
# ====================
python3 train.py --do_test --do_dev --seed 223 --bS 8 \
                 --datadir ./data/ko_token_not_h --logdir ./logs/ko_token_not_h \
                 --bert_name bert-base-multilingual-cased \
                 --max_seq_length 222 --EG \
                 --trained

# ====================
# ko_from_table
# ====================
python3 train.py --do_test --do_dev --seed 223 --bS 8 \
                 --datadir ./data/ko_from_table --logdir ./logs/ko_from_table \
                 --bert_name bert-base-multilingual-cased \
                 --max_seq_length 222 --EG \
                 --trained




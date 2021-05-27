# ====================
# wikisql raw
# ====================
python3 train.py --do_train --seed 1 --bS 8 --tepoch 10 \
                 --datadir ../data/data --logdir ./logs/sqlova_data \
                 --accumulate_gradients 2 --bert_name bert-base-uncased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1

# ====================
# ko_token 
# ====================
python3 train.py --do_train --seed 1 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG

python3 train.py --do_train --seed 1 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --trained


# ====================
# ko_token_not_h
# ====================
python3 train.py --do_train --seed 223 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_token_not_h --logdir ./logs/ko_token_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG

python3 train.py --do_train --seed 223 --bS 8 --tepoch 20 \
                 --datadir ./data/ko_token_not_h --logdir ./logs/ko_token_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --trained

python3 train.py --do_train --seed 223 --bS 8 --tepoch 20 \
                 --datadir ./data/ko_token_not_h --logdir ./logs/ko_token_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --trained


# ====================
# ko_from_table
# ====================
python3 train.py --do_train --seed 223 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_from_table --logdir ./logs/ko_from_table \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1

python3 train.py --do_train --seed 223 --bS 8 --tepoch 20 \
                 --datadir ./data/ko_from_table --logdir ./logs/ko_from_table \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1 --trained

# ====================
# ko_from_table_not_h
# ====================
python3 train.py --do_train --seed 223 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_from_table_not_h --logdir ./logs/ko_from_table_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1

python3 train.py --do_train --seed 223 --bS 8 --tepoch 20 \
                 --datadir ./data/ko_from_table_not_h --logdir ./logs/ko_from_table_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1 --trained

python3 train.py --do_train --seed 223 --bS 8 --tepoch 20 \
                 --datadir ./data/ko_from_table_not_h --logdir ./logs/ko_from_table_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1 --trained


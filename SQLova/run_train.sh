# ====================
# wikisql raw
# ====================
python3 train.py --do_train --seed 1 --bS 8 --tepoch 10 \
                 --datadir ./data/sqlova_data --logdir ./logs/sqlova_data \
                 --accumulate_gradients 2 --bert_name bert-base-uncased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1

# ====================
# ko_data v1 history 없음 ㅠㅠ
# ====================
python3 train.py --do_train --seed 1 --bS 8 --tepoch 100 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG

# ====================
# ko_data v2
# ====================
python3 train.py --do_train --seed 1 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG

# 그래픽카드 소음 때문에 이어서 학습 # 37 epoch부터 다시
python3 train.py --do_train --seed 1 --bS 8 --tepoch 13 \
                 --datadir ./data/ko_token --logdir ./logs/ko_token \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --trained 

# ====================
# ko_data_not_h
# ====================
python3 train.py --do_train --seed 223 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_token_not_h --logdir ./logs/ko_token_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG

# ====================
# ko_from_table
# ====================
python3 train.py --do_train --seed 223 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_from_table --logdir ./logs/ko_from_table \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1

# 그래픽카드 소음 때문에 이어서 학습 # 22 epoch 부터 다시
python3 train.py --do_train --seed 223 --bS 8 --tepoch 28 \
                 --datadir ./data/ko_from_table --logdir ./logs/ko_from_table \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --trained --gpu 1


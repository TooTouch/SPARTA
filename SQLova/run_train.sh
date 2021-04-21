# ====================
# wikisql raw
# ====================
python3 train.py --do_train --seed 1 --bS 8 --tepoch 10 \
                 --datadir ./data/sqlova_data --logdir ./logs/sqlova_data \
                 --accumulate_gradients 2 --bert_name bert-base-uncased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG \
                 --gpu 1

# ====================
# ko_data
# ====================
# python3 train.py --do_train --seed 1 --bS 8 \
#                  --datadir ./data/ko_token --logdir ./logs/ko_token \
#                  --accumulate_gradients 2 --bert_type_abb uS \
#                  --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG

# ====================
# ko_data_not_h
# ====================
python3 train.py --do_train --seed 223 --bS 8 --tepoch 50 \
                 --datadir ./data/ko_token_not_h --logdir ./logs/ko_token_not_h \
                 --accumulate_gradients 2 --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 --EG
beam_size="4 8 10 16 32"
topk_list='1,2,3,4,5,6,7,8,9,10'

# ====================
# ko_token
# ====================
for beam in $beam_size
do
    python3 train.py --do_test --do_dev --seed 1 --bS 8 \
                    --datadir ../data/ko_token --logdir ./logs/ko_token --savedir ./results/ko_token \
                    --bert_name bert-base-multilingual-cased \
                    --max_seq_length 222 --EG \
                    --trained --beam_size $beam --topk_list $topk_list
done


# ====================
# ko_token_not_h
# ====================
for beam in $beam_size
do
    python3 train.py --do_test --do_dev --seed 223 --bS 8 \
                    --datadir ../data/ko_token_not_h --logdir ./logs/ko_token_not_h --savedir ./results/ko_token_not_h \
                    --bert_name bert-base-multilingual-cased \
                    --max_seq_length 222 --EG \
                    --trained --beam_size $beam --topk_list $topk_list
done


# ====================
# ko_from_table
# ====================
for beam in $beam_size
do
    python3 train.py --do_test --do_dev --seed 223 --bS 8 \
                    --datadir ../data/ko_from_table --logdir ./logs/ko_from_table --savedir ./results/ko_from_table \
                    --bert_name bert-base-multilingual-cased \
                    --max_seq_length 222 --EG \
                    --trained --beam_size $beam --topk_list $topk_list
done


# ====================
# ko_from_table_not_h
# ====================
for beam in $beam_size
do
    python3 train.py --do_test --do_dev --seed 223 --bS 8 \
                    --datadir ../data/ko_from_table_not_h --logdir ./logs/ko_from_table_not_h --savedir ./results/ko_from_table_not_h \
                    --bert_name bert-base-multilingual-cased \
                    --max_seq_length 222 --EG \
                    --trained --beam_size $beam --topk_list $topk_list
done

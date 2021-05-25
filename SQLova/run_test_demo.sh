
# ====================
# demo
# ====================

beam_size='4 10'
logdir=('ko_from_table_not_h' 'ko_from_table' 'ko_token_not_h' 'ko_token')
datadir='./data/demo/'
savedir=('demo_from_table_not_h' 'demo_from_table' 'demo_token_not_h' 'demo_token')
topk_list='1,2,3,4,5,6,7,8,9,10'

for beam in $beam_size
do
    for i in {0..3}
    do
    echo "beam size: $beam, topk list: $topk_list, logdir: ${logdir[i]}, savedir: ${savedir[i]}"
    python3 train.py --do_test --demo --seed 1 --bS 8 \
                    --datadir $datadir --logdir ./logs/${logdir[i]} --savedir ./results/${savedir[i]} \
                    --bert_name bert-base-multilingual-cased \
                    --max_seq_length 222 --EG \
                    --trained --beam_size $beam --topk_list $topk_list
    done
done
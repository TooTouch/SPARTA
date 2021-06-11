
# ====================
# demo
# ====================

beam_size='4 10'
logdir=('ko_from_table_not_h' 'ko_from_table' 'ko_token_not_h' 'ko_token')
datadir='../demo/kaggle_demo/data'
savedir=('kaggle_from_table_not_h' 'kaggle_from_table' 'kaggle_token_not_h' 'kaggle_token')
topk_list='1,2,3,4,5,6,7,8,9,10'


if [ ! -f "$datadir/test_tok.jsonl" ];
then
    python annotate_tootouch.py --din $datadir --dout $datadir --split test
	echo "[PRE-PROCESSING] Complete"
else
	echo "[PRE-PROCESSING] File exists"
fi

for beam in $beam_size
do
    for i in {0..3}
    do
    echo "beam size: $beam, topk list: $topk_list, logdir: ${logdir[i]}, savedir: ${savedir[i]}"
    python3 train.py --do_test --seed 1 --bS 8 \
                    --datadir $datadir --logdir ./logs/${logdir[i]} --savedir ./results/${savedir[i]} \
                    --bert_name bert-base-multilingual-cased \
                    --max_seq_length 222 --EG \
                    --trained --beam_size $beam --topk_list $topk_list
    done
done
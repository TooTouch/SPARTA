# SQLova
- SQLova is a neural semantic parser translating natural language utterance to SQL query. 
- **Official Github**: https://github.com/naver/sqlova
- **Paper**: [A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization](https://arxiv.org/pdf/1902.01069.pdf)

## Architecture

<div align='center'>
    <strong>1. BERT Encoder</strong><br>
    <img src='https://user-images.githubusercontent.com/37654013/119766968-f95d4580-bef0-11eb-858e-50a9cd8d2f6d.png'>
</div>
<br>
<div align='center'>
    <strong>2. Sub-Modules</strong><br>
    <img src='https://user-images.githubusercontent.com/37654013/119767064-2e699800-bef1-11eb-81f1-1a5d7cb6992d.png'>
</div>

# Dataset 
- [WikiSQL](https://github.com/salesforce/WikiSQL)


# Pre-preprocess

In SQLova, the pre-processing of data is carried out to create two results below

- Question tokenization
- The start and end index of Where-value 

To tokenize English questions, `stanza` which is an open source for NLP published from Stanford University was used. However, we used the `Mecab` tokenizer which is tokenization for Korean language.

```bash
python annotate_tootouch.py --din $datadir --dout $savedir --split 'train,dev,test'
```

# Training and Test Details

**Hyperparameters**
- Learning rate : 0.001(Sub-Task model) 0.00001(BERT model)
- Batch size : 8
- Batch accumulation : 2

**Opimization**
- BERT model : Adam optimizer
- Sub-Task model : Adam optimizer

**BERT Model Parameters**
- Max sequence length : 222
- The number of target layer : 2

**Model Parameters**
- Hidden size of LSTM : 100
- The number of LSTM layers : 2
- Drop out: 0.3


# Train and Test

**Train**

```bash
python3 train.py --do_train --seed 1 --bS $batch_size --tepoch $epochs \
                 --datadir $data_directory --logdir $log_directory --savedir $save_directory \
                 --accumulate_gradients $accumulation_size --bert_name bert-base-multilingual-cased \
                 --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 \
                 --EG --trained
```

**Test**

```bash
python3 train.py --do_test --seed 1 --bS $batch_size \
                 --datadir $data_directory --logdir $log_directory --savedir $save_directory \
                 --bert_name bert-base-multilingual-cased --max_seq_length 222 \
                 --beam_size $beam --topk_list $topk_list \
                 --EG --trained 
```

# Demo Inference

```bash
bash run_test_demo.sh
```
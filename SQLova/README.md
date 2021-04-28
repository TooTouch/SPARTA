# SQLova
- SQLova is a neural semantic parser translating natural language utterance to SQL query. 
- **Official Github**: https://github.com/naver/sqlova
- **Paper**: [A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization](https://arxiv.org/pdf/1902.01069.pdf)


# Dataset 
- [WikiSQL](https://github.com/salesforce/WikiSQL)

## Create Korean dataset

We translated English question into Korean question in three ways as follows. 

[Download dataset](https://drive.google.com/drive/u/0/folders/1PnC_JU_QqCVEbyH2WaOETd51dW8Ssc3P)

1. **ko_token** : Keep `where values` in label and `header` of table among the words in English question
2. **ko_token_not_h** : Keep `header` of table among the words in English question
3. **ko_from_table** : Keep `values` in table among the words in English question

<div align='center'>
    <strong>Translation Process [ko_token]</strong><br>
    <img width="500" src='https://user-images.githubusercontent.com/37654013/115502415-ddda9b80-a2af-11eb-9892-029d914aa2f0.png'>
</div>


### run example

You can find translate commends in [translate.sh](https://github.com/TooTouch/SPARTA/blob/main/SQLova/run_translate.sh)

We translated English question into Korea question in three steps as follow

1. Create a question dataframe to translate English to Korean.

```bash
python translate.py --replace value --savedir $savedir
```

2. Translate English to Korean by using Google Tanslator ([click here!](https://translate.google.com/?hl=ko&sl=en&tl=ko&op=docs)) and copy a text file in ko_data directory such as 'ko_train_question.txt'

3. Insert Korean question 

```bash
python translate.py --replace token --savedir $savedir
```


# Pre-preprocess

In SQLova, the pre-processing of data is carried out to create two results below

- Question tokenization
- The start and end index of Where-value 

To tokenize English questions, `stanza` which is an open source for NLP published from Stanford University was used. However, we used the `Mecab` tokenizer to tokenize Korean questions.

```bash
python annotate_tootouch.py --din $datadir --dout $savedir
```

# Training Details

**Hyperparameters**
- Epochs : 50
- Learning rate : 0.001(model) 0.00001(BERT model)
- Batch size : 16

**Opimization**
- Model : Adam optimizer
- BERT model : Adam optimizer

**Model Parameters**
- Hidden size of LSTM : 100
- The number of LSTM layers : 2
- Drop out: 0.3

**BERT Model Parameters**
- Max sequence length : 222
- The number of target layer : 2


# Result

**Reproduction Test**

Model | Train<br> Logical Form<br> Accuracy (%) | Train<br> Execution<br> Accuracy (%) | Dev<br> Logical Form<br> Accuracy (%) | Dev<br> Execution<br> Accuracy (%) | Test<br> Logical Form<br> Accuracy (%) | Test<br> Execution<br> Accuracy (%)
---|---|---|---|---|---|---
SQLova + EG<br>official | x | x | 84.2 | 90.2 | 83.6 | 89.6 
SQLova + EG<br>implementation<br>(Epoch=10) | 83.6 | 88.7 | 82.8 | 88.8 | x | x
SQLova + EG<br>hugging face<br>(Epoch=10) | 69.7 | 79.3 | 76.7 | 83.7 | 76.1 | 83.8

**Dev Set**

Name            |SC Accuracy	    |SA Accuracy	    |WN Accuracy	    |WC Accuracy	    |WO Accuracy	    |WV Accuracy	    |Logical Form<br>Accuracy	    |Execution<br>Accuracy
---|---|---|---|---|---|---|---|---
**ko_token**	    |0.878399	|0.853462	|0.896924	|0.824724	|0.852155	|0.848355	|0.670348	|0.752998
**ko_token_not_h**	|0.760361	|0.838499	|0.876499	|0.781024	|0.834461	|0.824724	|0.572141	|0.660254
**ko_from_table**	|0.535091	|0.641492	|0.660610	|0.562404	|0.618216	|0.609073	|0.383327	|0.458497

**Test Set**

Name	        |SC Accuracy	    |SA Accuracy	    |WN Accuracy	    |WC Accuracy	    |WO Accuracy	    |acc_wv	    |Logical Form<br>Accuracy	    |Execution<br>Accuracy
---|---|---|---|---|---|---|---|---
**ko_token**	    |0.875173	|0.858357	|0.896712	|0.818050	|0.853823	|0.845572	|0.667275	|0.747827
**ko_token_not_h**	|0.751354	|0.844502	|0.880338	|0.781144	|0.835118	|0.825860	|0.564681	|0.656821
**ko_from_table**	|0.526956	|0.641643	|0.658521	|0.558509	|0.614750	|0.606059	|0.376685	|0.452828







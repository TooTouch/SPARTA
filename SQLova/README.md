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


# Result

Model | Train<br> Logical Form<br> Accuracy (%) | Train<br> Execution<br> Accuracy (%) | Dev<br> Logical Form<br> Accuracy (%) | Dev<br> Execution<br> Accuracy (%) | Test<br> Logical Form<br> Accuracy (%) | Test<br> Execution<br> Accuracy (%)
---|---|---|---|---|---|---
SQLova + EG<br>official | x | x | 84.2 | 90.2 | 83.6 | 89.6 
SQLova + EG<br>implementation<br>(Epoch=10) | 83.6 | 88.7 | 82.8 | 88.8 | x | x
SQLova + EG<br>`Korean^`<br> | 66.4 | 72.7 | 68.2 | 76.3 | x | x

**`Korea^`**
- the performance of Korean dataset
- training epochs was 100
- pretrained multilingual cased BERT model was used for embedding inputs
- using 

**Test Set**

Name   	            | acc_sc  	| acc_sa  	| acc_wn  	| acc_wc  	| acc_wo  	|  acc_wv  | 	acc_lx  | 	acc_x
---|---|---|---|---|---|---|---|---
sqlova_data		    | 0.922219	| 0.893249	| 0.930470	| 0.892934	| 0.905026	| 0.893689	| 0.750031	| 0.826175
ko_token		    | 0.451820	| 0.440106	| 0.461771	| 0.421716	| 0.438342	| 0.435256	| 0.342423	| 0.385124
ko_token_v1		    | 0.454780	| 0.440925	| 0.461960	| 0.429966	| 0.440547	| 0.437650	| 0.351745	| 0.393060
ko_token_not_h		| 0.327182	| 0.365474	| 0.380967	| 0.339778	| 0.362199	| 0.358861	| 0.247512	| 0.285741
ko_from_table		| 0.005164	| 0.007747	| 0.008124	| 0.007369	| 0.008061	| 0.008061	| 0.004535	| 0.005542

- ko_token : epoch 50
- ko_token_v1 : epoch 100
- ko_token_not_h : epoch 50
- ko_from_table : epoch 50




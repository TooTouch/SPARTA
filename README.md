# SPARTA (Semantic Parsing And Relational Table Aware)

We implement the deep learning model for converting Korean language to SQL query. We got the motivation for our project by using ["Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing"](https://github.com/salesforce/TabularSemanticParsing) as a homage


# Dataset 
- [WikiSQL](https://github.com/salesforce/WikiSQL)

## Create Korean dataset

We translated English question into Korean question in three ways as follows. 

[Download dataset](https://drive.google.com/drive/u/0/folders/1PnC_JU_QqCVEbyH2WaOETd51dW8Ssc3P)

1. **multi_wikisql1.1** : Keep values of table using `fuzzy string matching` among the words in English question
1. **ko_token** : Keep `where values` in label and `header` of table among the words in English question
2. **ko_token_not_h** : Keep `header` of table among the words in English question
3. **ko_from_table** : Keep `values` in table among the words in English question

<div align='center'>
    <strong>Translation Process [ko_token]</strong><br>
    <img width="500" src='https://user-images.githubusercontent.com/37654013/115502415-ddda9b80-a2af-11eb-9892-029d914aa2f0.png'>
</div>

### Run

1. Create a question dataframe to translate English to Korean.

```bash
python translate.py --extract --datadir ./data --savedir ./ko_data
```

2. Translate English to Korean by using Google Tanslator ([click here!](https://translate.google.com/?hl=ko&sl=en&tl=ko&op=docs)) and copy a text file in ko_data directory such as 'ko_train_question.txt'

3. Insert Korean question 

```bash
python translate.py --insert --datadir ./data --savedir ./ko_data
```

# Model
- [SQLova](): 

> Performance (WIKI-SQL)

English and Korean performance [[Here](https://github.com/TooTouch/SPARTA/tree/main/SQLova)]

- [BRIDGE](https://github.com/salesforce/TabularSemanticParsing): Use the command from this link for reproduction

> Performance (WIKI-SQL)

| BERT                   | Top-1 EM | TOP-3 EM | TOP-10 EM | TOP-1 EXE | TOP-3 EXE | TOP-10 EXE |
| ---------------------- | -------- | -------- | --------- | --------- | --------- | ---------- |
| **Multilingual Cased** | 0.378    | 0.503    | 0.612     | 0.453     | 0.578     | 0.676      |
| Multilingual Uncased   | 0.02     |          |           |           |           |            |
| Base Uncased           | 0        |          |           |           |           |            |

> Prediction Results (Best Since)

```
{"sql": {"sel": 2, "conds": [[0, 0, "Terrence Ross"]], "agg": 0}, "table_id": "1-10015132-16"}
{"sql": {"sel": 5, "conds": [[4, 0, "1995-VALUE"]], "agg": 0}, "table_id": "1-10015132-16"}
{"sql": {"sel": 5, "conds": [[4, 0, "2003-06"]], "agg": 0}, "table_id": "1-10015132-16"}
{"sql": {"sel": 5, "conds": [[0, 0, "Jalen Rose"]], "agg": 3}, "table_id": "1-10015132-16"}
{"sql": {"sel": 4, "conds": [[3, 0, "Assen"]], "agg": 0}, "table_id": "1-10083598-1"}
{"sql": {"sel": 6, "conds": [[4, 0, "Kevin Curtain"]], "agg": 3}, "table_id": "1-10083598-1"}
{"sql": {"sel": 1, "conds": [[3, 0, "Misano"]], "agg": 0}, "table_id": "1-10083598-1"}
```

* "VALUE" value corrupts the prediction and makes our accuracy lower than expected.
* Seems like "VALUE" value comes from the input sentence which is used for encoding, and pointer generator somehow takes that value to decoding.

> Prediction Results (on Multilingual BERT Uncased)

```
SELECT * FROM L
SELECT * FROM L
SELECT * FROM L
SELECT * FROM J
{'sel': 5, 'conds': [[0, 0, 'mir johnson']], 'agg': 0}
SELECT * FROM J
SELECT * FROM J
{'sel': 2, 'conds': [[5, 0, 'fresno state']], 'agg': 0}
{'sel': 5, 'conds': [[0, 0, 'rey johnson']], 'agg': 0}
{'sel': 2, 'conds': [[3, 0, 'jacques chirac']], 'agg': 0}
SELECT * FROM Chronology of longest serving G8 Leaders
```

* WHERE Value normally starts with Capital Letter, which uncased version of BERT couldn't generate tokens with. By used cased version, this phenomenons mostly vanish.


# Reference

[1] Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning.

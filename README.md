# SPARTA (Semantic Parsing And Relational Table Aware)

This is a term project in `Unstructured Text Analysis` class. We implement the deep learning model for converting Korean language to SQL query. 

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/119700897-bec2c100-be8e-11eb-9d61-36de1ca66d5a.png'>
</div>

**Team Members**
- Hoonsang Yoon 
- Jaehyuk Heo 
- Jungwoo Choi
- Jeongseob Kim

**Information**
- Korea University [DSBA Lab](http://dsba.korea.ac.kr/)
- Advisor: [Pilsung Kang](http://dsba.korea.ac.kr/professor/)


# Demo 

Check about Demo in [here](https://github.com/TooTouch/SPARTA/tree/main/demo).

# Video

**Text2SQL Result Video**

[![](https://user-images.githubusercontent.com/37654013/123519659-aa6d2080-d6e7-11eb-867c-05f3e47756f5.png)](https://youtu.be/qzeufyuyrEE)


# Dataset 
- [WikiSQL](https://github.com/salesforce/WikiSQL)

```bash
tar xvjf data/data.tar.bz2
```

## Korean WikiSQL dataset

```bash
unzip data/ko_token.zip
unzip data/ko_token_not_h.zip
unzip data/ko_from_table.zip
unzip data/ko_from_table_not_h.zip
```

# Translation

We translated English question into Korean question in four ways as follows. 

[Download dataset](https://drive.google.com/drive/u/0/folders/1PnC_JU_QqCVEbyH2WaOETd51dW8Ssc3P)

No | Method | Data Name | Description
---|---|---|---
1 | Where+Select | ko_token |  Keep `where values` in label and `column` used in select clause among the words in English question
2 | Where | ko_token_not_h |  Keep `header` of table among the words in English question
3 | Table+Header | ko_from_table | Keep `values` and `header` in table among the words in English question 
4 | Table | ko_from_table_not_h |  Keep `values` in table among the words in English question

<div align='center'>
    <strong>Method 1 (Where+Select)</strong><br>
    <img width="1000" src='https://user-images.githubusercontent.com/37654013/119702737-c1beb100-be90-11eb-9c71-00498dafec0d.png'>
</div>

<div align='center'>
    <strong>Method 2 (Where)</strong><br>
    <img width="1000" src='https://user-images.githubusercontent.com/37654013/119702997-0d715a80-be91-11eb-955d-bafd7e0912b4.png'>
</div>

<div align='center'>
    <strong>Method 3 (Table+Header)</strong><br>
    <img width="1000" src='https://user-images.githubusercontent.com/37654013/119703614-aef8ac00-be91-11eb-947d-6da2086ffeb7.png'>
</div>

<div align='center'>
    <strong>Method 4 (Table)</strong><br>
    <img width="1000" src='https://user-images.githubusercontent.com/37654013/119703354-6d680100-be91-11eb-87ab-a0d07bdf9df6.png'>
</div>



## Run translation

1. Create a question dataframe to translate English to Korean.

```bash
bash run_translate.sh value
```

2. Translate English to Korean by using Google Tanslator ([click here!](https://translate.google.com/?hl=ko&sl=en&tl=ko&op=docs)) and copy a text file in ko_data directory such as 'ko_train_question.txt'

3. Insert Korean question

```bash
bash run_translate.sh token
```

# SPARTA Model

We use pretrained multilingual BERT as encoder.

**Sub Task**

1. SQLova [ [paper](https://arxiv.org/pdf/1902.01069.pdf) | [github](https://github.com/naver/sqlova) ]
2. HydraNet [ [paper](https://arxiv.org/pdf/2008.04759.pdf) | [github](https://github.com/lyuqin/HydraNet-WikiSQL) ]

**Seq2Seq**

1. BRIDGE(TabularSemanticParsing)[ [paper](https://arxiv.org/pdf/2012.12627.pdf) | [github](https://github.com/salesforce/TabularSemanticParsing) ]


# Evaluation

- Logical Form Accuracy
- Execution Accuracy

<div align='center'>
    <img width="1000" src='https://user-images.githubusercontent.com/37654013/119704032-229ab900-be92-11eb-9687-acdc64ab117a.png'>
</div>



# Experiments


Model | Task | Test<br>Logical Form<br>Accuracy(%) | Test<br>Execution<br>Accuracy(%)
---|---|:---:|:---:
SQLova   | Subtask    | 65.8 | 74.3
HydraNet | Subtask    | 40.4 | 40.7
Bridge   | Generation | 54.6 | 62.1


# Download Trained Models

Method | SQlova | Bridge
---|---|---
Where+Select | [Download](https://drive.google.com/file/d/1f1E53QdPAboF3cWdbET4gjAX5Y5Li_Tv/view?usp=sharing) | -
Where | [Download](https://drive.google.com/file/d/1fgmhq50YbHHapKWOityuoO_zXVOvB3Sh/view?usp=sharing) | -
Table+Header | [Download](https://drive.google.com/file/d/1a8vF5mgV1lYDDbFlu7riu0tySFjF08e3/view?usp=sharing) | -
Table | [Download](https://drive.google.com/file/d/1GBQeQ_DuG1YAnaNSkk6u26CawnfqEikP/view?usp=sharing) | -


# Presentation 

**Proposal**

[![](https://user-images.githubusercontent.com/37654013/119706901-82df2a00-be95-11eb-881a-9a6960bb205a.png)](https://youtu.be/Chy0W8W5ck8)


**Interim Findings**

[![](https://user-images.githubusercontent.com/37654013/119706604-2a0f9180-be95-11eb-94ad-7d118e05ebcd.png)](https://www.youtube.com/watch?v=vMJcWZ6Sn5s)

**Final**

[![](https://user-images.githubusercontent.com/37654013/123519632-87db0780-d6e7-11eb-9aa2-4d3b009f582b.png)](https://www.youtube.com/watch?v=3upT_dFziC8)

# Reference

- [1] Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning.
- [2] Hwang, W., Yim, J., Park, S., & Seo, M. (2019). A comprehensive exploration on wikisql with table-aware word contextualization. KR2ML Workship at NeurIPS 2019
- [3] Lyu, Q., Chakrabarti, K., Hathi, S., Kundu, S., Zhang, J., & Chen, Z. (2020). Hybrid ranking network for text-to-sql. arXiv preprint arXiv:2008.04759.
- [4] Xi Victoria Lin, Richard Socher and Caiming Xiong. Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing. Findings of EMNLP 2020.

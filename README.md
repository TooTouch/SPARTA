# SPARTA (Semantic Parsing And Relational Table Aware)

We implement the deep learning model for converting Korean language to SQL query. We got the motivation for our project by using ["Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing"](https://github.com/salesforce/TabularSemanticParsing) as a homage


# Dataset 
- [WikiSQL](https://github.com/salesforce/WikiSQL)

## Create Korean dataset

1. Create a question dataframe to translate English to Korean.

```bash
python translate.py --extract --datadir ./data --savedir ./ko_data
```

2. Translate English to Korean by using Google Tanslator ([click here!](https://translate.google.com/?hl=ko&sl=en&tl=ko&op=docs)) and copy a text file in ko_data directory such as 'ko_train_question.txt'

3. Insert Korean question 

```bash
python translate.py --insert --datadir ./data --savedir ./ko_data
```

# Reference

[1] Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning.

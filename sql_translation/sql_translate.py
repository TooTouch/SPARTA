import os
import json
from tqdm import tqdm
import pandas as pd
import fuzzy_string_match
from itertools import chain

import argparse 

def upper_transform(sentence: str, word: str):
    sentence = sentence.replace(word, f'[\{word.upper()}\]')
    return sentence

def search_table(table_list: list, table_id: str):
    for table in table_list:
        if table['id'] == table_id:
            return table

def upper_transform_question(question: str, table_list, table_id):
    """
    Transform part of the question into upper character to enhance the quality of translation

    Argument
    --------
    question: str, question for database
    table_list: list, list of dictionary of tables
    table_id: str, id for target table
    """
    question = question.lower()
    target_table = search_table(table_list, table_id)

    header = target_table['header']
    rows = list(chain(*target_table['rows']))
    rows.extend(header)

    # find link btw question and values
    matches = fuzzy_string_match.get_matched_entries(question, rows)
    # if match exists, transform the matches to UPPER
    if matches != None:
        matches = [m[0].lower() for m in matches]

        for match in matches:
            if match in question:
                question = upper_transform(question, match)
        return question
    else:
        return question  

def upper_transform_bulk(questions, table_list, table_ids):
    upper_questions = []
    for question, table_id in tqdm(zip(questions, table_ids), desc='Converting questions to upper letter', total=len(questions)):
        upper_q = upper_transform_question(question, table_list, table_id)
        upper_questions.append(upper_q)
    return upper_questions

def extract_data(name: str, datadir: str):
    """
    Data extraction from a jsonl file

    Argument
    --------
    name: str, dataset name ['train','dev','test']

    Return
    ------
    data_lst: data list
    """
    data_lst = []
    filepath = os.path.join(datadir, f'{name}.jsonl')
    with open(filepath) as f:
        for l in tqdm(f, desc=name.upper()):
            data_lst.append(json.loads(l))
    return data_lst
    
def extract_question(name: str, data: list, savedir: str): 
    """
    Extract question from data and save to excel file

    Argument
    --------
    name: str, dataset name ['train','dev','test']
    data: list, data list 
    savedir: str, directory to save file
    """
    # define path to save file
    filepath = os.path.join(savedir,f'{name}_question.xlsx')

    # remove '\xa0'
    questions = [" ".join(d['question'].split()) for d in data]
    table_ids =[d['table_id'] for d in data]
    tables = extract_data(f'{name}.tables', './data')

    extracted_q = upper_transform_bulk(questions, tables, table_ids)

    # save question dataframe 
    question_df = pd.DataFrame({'question':extracted_q})
    question_df.to_excel(filepath,index=False)

def cleanse_lines(korean_dir: str):
    """
    Cleanse korean file lines

    Argument
    ---
    korean_dir: str, directory of korean txt file which has been translated with Google Translate
    """
    file = open(korean_dir, 'r')
    lines = file.readlines()

    lines = [line.replace('[\\', '') for line in lines]
    lines = [line.replace('\\]', '') for line in lines]
    lines = [line.replace('\n', '').strip()  for line in lines]

    lines = [' '.join(line.split()) for line in lines]
    lines = [line.title() for line in lines]

    return lines


def insert_ko_question(data: list, name: str, savedir: str):
    """
    Insert Korean questions with English questions

    Argument
    --------
    name: str, dataset name ['train','dev','test']
    data: list, data list 
    savedir: str, directory to save file
    """
    # define path to save file
    ko_filepath = os.path.join(savedir,f'ko_{name}_question.txt')
    filepath = os.path.join(savedir,f'ko_{name}.jsonl')

    assert os.path.isfile(ko_filepath), f'ko_{name}_question.txt does not exist.'

    # read Korean questions
    # with open(ko_filepath,'r') as f:
    #     ko_question = [d.replace('\n','') for d in f.readlines()]
    ko_question = cleanse_lines(ko_filepath)
    
    # replace Korean questions with English questions
    for i in range(len(data)):
        data[i]['question'] = ko_question[i]

    # save Korean data
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
        print('Write {} records to {}'.format(len(data), filepath))

    
   
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',type=str,default='data', )
    parser.add_argument('--savedir',type=str,default='./ko_data',help='save directory')
    parser.add_argument('--extract',action='store_true',help='extract question')
    parser.add_argument('--insert',action='store_true',help='replace Korean question with English question ')
    args = parser.parse_args()
    
    # make directory
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    
    for f in ['train', 'dev', 'test']:
        print('[WikiSQL DATA]')
        data = extract_data(name=f, datadir=args.datadir)
        
        if args.extract:
            print('[EXTRACT QUESTION]')
            extract_question(name=f, data=data, savedir=args.savedir)

        if args.insert:
            print('[INSERT KOREAN QUESTION]')
            insert_ko_question(name=f, data=data, savedir=args.savedir)

        print('Done.\n')
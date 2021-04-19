from sqlova.utils.utils_wikisql import load_wikisql_data

import pandas as pd 
import os 
import json
from copy import deepcopy
import argparse

def replace_value_with_h_wv(data: list, tables: dict):
    """
    Replace header and where-values in question into token value

    Arguments
    ---------
    data: list, data list 
    tables: dict, tables infomation

    Returns
    -------
    data: list, questions with token of header and where-values
    total_wv_in_q_lst: list, where-values used in question of data point
    h_in_q_lst: list, header used in question of data point
    """
    total_wv_in_q_lst = [] # total where value in question
    h_in_q_lst = [] # header in question
    for idx, d in enumerate(data):
        # where value
        wv_in_q_lst = []
        for i, c in enumerate(d['sql']['conds']):
            wv = str(c[2]) # where value

            # find start index
            start_idx = d['question'].lower().find(wv.lower())

            # wv_in_q is where-value in question
            wv_in_q = d['question'][start_idx:start_idx+len(wv)]
            
            # replace where value in question with [V{idx}]
            number_dict = {
                0: '영',
                1: '일',
                2: '이',
                3: '삼'
            } 
            data[idx]['question'] = data[idx]['question'][:start_idx] + f'[값{number_dict[i]}]' + data[idx]['question'][start_idx+len(wv):]
        
            wv_in_q_lst.append(wv_in_q)
        
        total_wv_in_q_lst.append(wv_in_q_lst)
            
        # header
        h = tables[d['table_id']]['header'][d['sql']['sel']].lower()
        if h in d['question'].lower():
            # find start index 
            start_idx = d['question'].lower().find(h.lower())
            
            # header in question
            h_in_q = d['question'][start_idx:start_idx+len(h)]
            
            # replace header in question with [H]
            data[idx]['question'] = data[idx]['question'][:start_idx] + '[이름]' + data[idx]['question'][start_idx+len(h):]
            
            h_in_q_lst.append(h_in_q)
        else:
            h_in_q_lst.append(None)

    return data, total_wv_in_q_lst, h_in_q_lst


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

    # save question dataframe 
    question_df = pd.DataFrame({'question':questions})
    question_df.to_excel(filepath,index=False)


def save_token_question_and_info(name: str, datadir: str, savedir: str):
    """
    Save question with token and information

    Argumnets
    ---------
    name: str, dataset name ['train','dev','test']
    datadir: str, saved data directory
    savedir: str, directory to save data
    """
    
    token_info = {name:{}}
    # load data
    data, tables = load_wikisql_data(path_wikisql=datadir, mode=name, no_tok=True, no_hs_tok=True)
    # transform
    data_with_token, total_wv_in_q_lst, h_in_q_lst = replace_value_with_h_wv(deepcopy(data), tables)
    # save
    extract_question(name=name, data=data_with_token, savedir=savedir)
    
    token_info[name]['wv'] = total_wv_in_q_lst
    token_info[name]['h'] = h_in_q_lst
    
    json.dump(token_info, open(os.path.join(savedir,f'{name}_token_info.json'),'w'), indent=4)
    print(f'Complete {name.upper()}')



def insert_replace_h_wv_with_value(name: str, datadir:str, savedir: str):
    """
    Insert Korean questions in data

    Argument
    --------
    name: str, dataset name ['train','dev','test']
    datadir: str, saved data directory
    savedir: str, directory to save file
    """
    # define path to save file
    ko_filepath = os.path.join(savedir,f'ko_{name}_question.txt')
    filepath = os.path.join(savedir,f'{name}.jsonl')

    assert os.path.isfile(ko_filepath), f'ko_{name}_question.txt does not exist.'

    # load English data
    data, _ = load_wikisql_data(path_wikisql=datadir, mode=name, no_tok=True, no_hs_tok=True)

    # read Korean questions
    with open(ko_filepath,'r') as f:
        ko_question = f.readlines()
        ko_question = list(map(lambda x: x.replace('\n',''), ko_question))

    # load token information
    token_info = json.load(open(os.path.join(savedir, f'{name}_token_info.json'),'r'))

    # replace token with values
    for idx, d in enumerate(ko_question):
        if '[이름]' in d:
            ko_question[idx] = ko_question[idx].replace('[이름]', token_info[name]['h'][idx])

        if token_info[name]['wv'][idx] != []:
            for i, wv in enumerate(token_info[name]['wv'][idx]):
                number_dict = {
                    0: '영',
                    1: '일',
                    2: '이',
                    3: '삼'
                }
                ko_question[idx] = ko_question[idx].replace(f'[값{number_dict[i]}]', wv)
                 
    # insert Korean questions in data
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
    parser.add_argument('--dataset', type=str, default='train,dev,test', help='dataset names')
    parser.add_argument('--replace', type=str, choices=['value','token'], help='value: replace value with token, token: replace token with value')
    parser.add_argument('--datadir', type=str, default='./data/raw', help='wikisql data directory')
    parser.add_argument('--savedir', type=str, default='./data/ko_token', help='data directory to save file')
    args = parser.parse_args()

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    dataset_names = args.dataset.split(',')

    if args.replace == 'value':
        for name in dataset_names:
            save_token_question_and_info(name=name, datadir=args.datadir, savedir=args.savedir)

    elif args.replace == 'token':
        for name in dataset_names:
            insert_replace_h_wv_with_value(name=name, datadir=args.datadir, savedir=args.savedir)


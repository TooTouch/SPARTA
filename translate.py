import os
import sys
import urllib.request

import json
from tqdm import tqdm
import time

import argparse 

def extract_data(name: str):
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
    with open(f'data/{name}.jsonl') as f:
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
    filepath = os.path.join(args.savedir,f'{name}_question.xlsx')

    # remove '\xa0'
    question = [" ".join(data['question'].split()) for data in train_data]

    # save question dataframe 
    question_df = pd.DataFrame({'question':question})
    question_df.to_excel(filepath,index=False)

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
    ko_filepath = os.path.join(args.savedir,f'ko_{name}_question.txt')
    filepath = os.path.join(args.savedir,f'ko_{name}.jsonl')

    assert os.path.isfile(ko_filepath), f'ko_{name}_question.txt does not exist.'

    # read Korean questions
    with open(ko_filepath,'r') as f:
        ko_question = [d.replace('\n','') for d in f.readlines()]
    
    # replace Korean questions with English questions
    for i in range(len(data)):
        data[i]['question'] = ko_question[i]

    # save Korean data
    json.dump(data, open(filepath,'w'))

    
   
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir',type=str,default='./data',help='save directory')
    parser.add_argument('--extract',action='store_true',help='extract question')
    parser.add_argument('--ko_insert',action='store_true',help='replace Korean question with English question ')
    args = parser.parse_args()

    
    for f in ['train','dev','test']:
        data = extract_data(name=f)
        
        if args.extract:
            print('[EXTRACT QUESTION]')
            extract_question(name=f, data=data, savedir=args.savedir)

        if args.ko_insert:
            print('[INSERT KOREAN QUESTION]')
            insert_ko_question(name=f, data=data, savedir=args.savedir)

            

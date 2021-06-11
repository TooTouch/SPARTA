import pandas as pd
import sqlite3
import os 
import json

import argparse

def create_db(data, table_id, conn):

    # read table
    table_id = 'table_' + table_id
    table_id = table_id.replace('-','_')

    # schema
    schema = []
    for idx, dtype in enumerate(data.dtypes):
        t = 'text' if dtype=='object' else 'real'
        schema.append(f'col{idx} {t}')

    schema = ', '.join(schema)

    # create table
    conn.execute(f'CREATE TABLE {table_id}({schema})')

    # insert data into db
    values = ['?' for i in range(len(data.columns))]

    for str_col in data.columns[data.dtypes=='object']:
        data[str_col] = data[str_col].str.lower()

    cur = conn.cursor()
    cur.executemany(
        f'INSERT INTO {table_id} VALUES (' + ', '.join(values) + ')',
        data.values
    )

    conn.commit()

def create_table(data, table_id):
    table = {}

    # headers
    header = [c.upper() for c in data.columns]

    # types
    types_lst = []
    for dtype in data.dtypes:
        t = 'text' if dtype=='object' else 'real'
        types_lst.append(t)

    # rows
    data = data.astype(str)
    data = data.apply(lambda x: x.str.lower(), axis=1)

    # insert
    table['id'] = table_id
    table['header'] = header
    table['types'] = types_lst
    table['rows'] = data.values.tolist()

    return table


def preprocessing(data, table_id):
    # data preprocessing
    if table_id == 'CustomerAcqusition':
        data['Limit'] = data['Limit'].astype(int)
        data = data.dropna()
        
    elif table_id == 'CustomerRepayment':
        del data['Unnamed: 4']
        data = data.rename(columns={'SL No:':'SL No'})
        data.iloc[0,0] = 1
        data = data.dropna()
        data['SL No'] = data['SL No'].astype(int)

    elif table_id == 'CustomerSpend':
        data = data.rename(columns={'SL No:':'SL No'})
        data = data.dropna()

    elif table_id == 'ApplicationRecord':
        data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'].astype(int)
        data['AMT_INCOME_TOTAL'] = data['AMT_INCOME_TOTAL'].astype(int)
        data = data.dropna()

    elif table_id == 'PersonalTransaction':
        data['Date'] = pd.to_datetime(data.Date).astype(str)
    
    else:
        data = data.dropna()

    return data.head(100) # select 100 samples from top

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',type=str,help='data directory')
    args = parser.parse_args()

    table_ids = [table_id for table_id in os.listdir(args.datadir) if '.csv' in table_id]
    conn = sqlite3.connect(f"./{args.datadir}/test.db")
    
    table_lst = []
    for table_id in table_ids:
        # read data
        data = pd.read_csv(f'{args.datadir}/{table_id}')
        # define table_id
        table_id = table_id.replace('.csv','')

        # preprocessing data
        data = preprocessing(data, table_id)
        
        # create database
        create_db(data, table_id, conn)
        print(f'CREATE TABLE {table_id} / TABLE SIZE: ROW-{data.shape[0]} COL-{data.shape[1]}')

        # create table
        table = create_table(data, table_id)
        table_lst.append(table)

    # end database connection
    conn.close()

    # save tables
    n_written = 0
    with open(f'./{args.datadir}/test.tables.jsonl','w',encoding='utf-8') as fo:
        for line in table_lst:
            fo.write(json.dumps(line, ensure_ascii=False) + '\n')
            n_written += 1
        print('wrote {} examples'.format(n_written))

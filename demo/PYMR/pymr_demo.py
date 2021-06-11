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
    table['name'] = table_id.replace('1-','table_').replace('-','_')

    return table


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',type=str,help='data directory')
    args = parser.parse_args()

    conn = sqlite3.connect(f"./{args.datadir}/test.db")
    
    # read data
    data = pd.read_csv(f'{args.datadir}/PYMR.csv')
    # define table_id
    table_id = 'PYMR'

    # preprocessing data
    data['Year'] = data['Year'].astype(int)
    
    # create database
    create_db(data, table_id, conn)
    print(f'CREATE TABLE {table_id} / TABLE SIZE: ROW-{data.shape[0]} COL-{data.shape[1]}')

    # create table
    table = create_table(data, table_id)
    table_lst = [table]

    # end database connection
    conn.close()

    # save tables
    n_written = 0
    with open(f'./{args.datadir}/test.tables.jsonl','w',encoding='utf-8') as fo:
        for line in table_lst:
            fo.write(json.dumps(line, ensure_ascii=False) + '\n')
            n_written += 1
        print('wrote {} examples'.format(n_written))

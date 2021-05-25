#!/usr/bin/env python
import json
from argparse import ArgumentParser
from tqdm import tqdm
from lib.dbengine import DBEngine
from lib.query import Query
from lib.common import count_lines

import os 
import time
import pickle
import json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source_file', help='source file for the prediction')
    parser.add_argument('db_file', help='source database for the prediction')
    parser.add_argument('pred_file', help='predictions by the model')
    parser.add_argument('--ordered', action='store_true', help='whether the exact match should consider the order of conditions')
    args = parser.parse_args()

    engine = DBEngine(args.db_file)
    exact_match = []
    
    # debug
    temp = []
    idx = 0
    
    with open(args.source_file) as fs, open(args.pred_file) as fp:
        grades = []
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(args.source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'], ordered=args.ordered)
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            pred = ep.get('error', None)
            qp = None
            if not ep.get('error', None):
                try:
                    qp = Query.from_dict(ep['query'], ordered=args.ordered)
                    pred = engine.execute_query(eg['table_id'], qp, lower=True)
                    
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)
            
            # debug
            idx+=1
            temp.append({"idx" : idx, "pred" : pred, "gold" : gold})
#             print("idx : ", idx)
#             print("query pred : ", qp)
#             print("query gold : ", qg)
            
#             if idx == 10:
#                 break
            
        ### 수정
        result_ = {
            'ex_accuracy': sum(grades) / len(grades),
            'lf_accuracy': sum(exact_match) / len(exact_match),
            }
        
        path_new = f"{args.pred_file}_{time.strftime('%c', time.localtime(time.time()))}_lf_ea_result.json"
        path_temp = f"{args.pred_file}_{time.strftime('%c', time.localtime(time.time()))}_lf_ea_result_value.json"
        result_json = json.dumps(result_)
        temp_json = json.dumps(temp)

        with open(path_new, 'w') as f:
            f.write(result_json)
        with open(path_temp, 'w') as f:
            f.write(temp_json)     
            
        ### 수정
        
        print(json.dumps({
            'ex_accuracy': sum(grades) / len(grades),
            'lf_accuracy': sum(exact_match) / len(exact_match),
            }, indent=2))

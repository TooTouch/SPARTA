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

    parser.add_argument('--topk', type=int, default=3, help='k of top_k')
    
    args = parser.parse_args()

    engine = DBEngine(args.db_file)
   
    temp = []
    
    with open(args.source_file) as fs, open(args.pred_file) as fp:
        grades = []
        exact_match = []
        
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(args.source_file)):            
            eg = json.loads(ls)
            qg = Query.from_dict(eg['sql'], ordered=args.ordered)
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            
            pred_topk = []
            qp_topk = []
            
            ep = json.loads(lp)
            pred = ep.get('error', None)
            qp = None
            for i in range(args.topk):
                if not ep.get('error', None):
                    try:
                        
                        if ep['query'][str(i)]['conds'] == [[]]:
                            ep['query'][str(i)]['conds'] = []
            
                        qp = Query.from_dict(ep['query'][str(i)], ordered=args.ordered)
                        pred = engine.execute_query(eg['table_id'], qp, lower=True)

                    except Exception as e:

                        pred = repr(e)

                qp_topk.append(qp)
                pred_topk.append(pred)
            
            if gold in pred_topk:
                correct = True
            else:
                correct = False
                
            grades.append(correct)

            if qg in qp_topk:
                match = True
            else:
                match = False
                
            exact_match.append(match)
            
            temp.append({"gold" : gold, "pred_topk" : pred_topk, "correct" : correct, "match" : match})

            
        result_ = {
        'ex_accuracy': sum(grades) / len(grades),
        'lf_accuracy': sum(exact_match) / len(exact_match),
        }

        path_new = f"{args.pred_file}_{time.strftime('%c', time.localtime(time.time()))}_lf_ea_result_topk.json"
        path_temp = f"{args.pred_file}_{time.strftime('%c', time.localtime(time.time()))}_lf_ea_result_value_topk.json"
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
            }, indent=2), "len grades : ", len(grades), "len exact match : ", len(exact_match))


import os
import json
import numpy as np
import pickle
import utils
from argparse import ArgumentParser
from modeling.model_factory import create_model
from featurizer import HydraFeaturizer, SQLDataset
from wikisql_lib.dbengine import DBEngine

# class for json serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        

import time

def print_metric(label_file, pred_file):
    sp = [(json.loads(ls)["sql"], json.loads(lp)["query"]) for ls, lp in zip(open(label_file), open(pred_file))]

    sel_acc = sum(p["sel"] == s["sel"] for s, p in sp) / len(sp)
    agg_acc = sum(p["agg"] == s["agg"] for s, p in sp) / len(sp)
    wcn_acc = sum(len(p["conds"]) == len(s["conds"]) for s, p in sp) / len(sp)

    def wcc_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[0] for c in a] == [c[0] for c in b]

    def wco_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[1] for c in a] == [c[1] for c in b]

    def wcv_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [str(c[2]).lower() for c in a] == [str(c[2]).lower() for c in b]

    wcc_acc = sum(wcc_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wco_acc = sum(wco_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wcv_acc = sum(wcv_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)

    print('sel_acc: {}\nagg_acc: {}\nwcn_acc: {}\nwcc_acc: {}\nwco_acc: {}\nwcv_acc: {}\n' \
          .format(sel_acc, agg_acc, wcn_acc, wcc_acc, wco_acc, wcv_acc))
    
    result_component = {"sel_acc" : sel_acc,
                        "agg_acc" : agg_acc,
                        "wcn_acc" : wcn_acc,
                        "wcc_acc" : wcc_acc,
                        "wco_acc" : wco_acc,
                        "wcv_acc" : wcv_acc}
    
    result_component_json = json.dumps(result_component)
    path_component = f"{out_file}_{time.strftime('%c', time.localtime(time.time()))}_component_result.json"
    
    with open(path_component, 'w') as f:
        f.write(result_component_json)
    
    


if __name__ == "__main__":
    
###================================================================================================###

    parser = ArgumentParser()

    parser.add_argument('--topk', type=int, default=3, help='k of top_k')
    parser.add_argument('--beam_size', type=int, default=5, help='k of top_k')
    
    args = parser.parse_args()



    # case1 : ko_token_1
    ### dev
#     in_file = "data/wikidevko_token_1.jsonl"
#     out_file = f"output/dev_out_ko_token_1_beam-{args.beam_size}_top-{args.topk}.jsonl"
#     label_file = "WikiSQL/data/dev.jsonl"
#     db_file = "WikiSQL/data/dev.db"
#     model_out_file = f"output/dev_model_out_ko_token_1_beam-{args.beam_size}_top-{args.topk}.pkl"    
    

#     ### test (20210504_203857)
#     in_file = "data/wikitestko_token_1.jsonl"
#     out_file = f"output/test_out_ko_token_1_beam-{args.beam_size}_top-{args.topk}.jsonl"
#     label_file = "WikiSQL/data/test.jsonl"
#     db_file = "WikiSQL/data/test.db"
#     model_out_file = f"output/test_model_out_ko_token_1_beam-{args.beam_size}_top-{args.topk}.pkl"
    
    
    # case2 : ko_token_not_h_2
#     ### dev
#     in_file = "data/wikidevko_token_not_h_2.jsonl"
#     out_file = "output/dev_out_ko_token_not_h_2.jsonl"
#     label_file = "WikiSQL/data/dev.jsonl"
#     db_file = "WikiSQL/data/dev.db"
#     model_out_file = "output/dev_model_out_ko_token_not_h_2.pkl"
    
    
    ### test (20210505_124438)
#     in_file = "data/wikitestko_token_not_h_2.jsonl"
#     out_file = f"output/test_out_ko_token_not_h_2_beam-{args.beam_size}_top-{args.topk}.jsonl"
#     label_file = "WikiSQL/data/test.jsonl"
#     db_file = "WikiSQL/data/test.db"
#     model_out_file = f"output/test_model_out_ko_token_not_h_2_beam-{args.beam_size}_top-{args.topk}.pkl"


    # case3 : ko_from_table_3
    ### dev
#     in_file = "data/wikidevko_from_table_3.jsonl"
#     out_file = "output/dev_out_ko_from_table_3.jsonl"
#     label_file = "WikiSQL/data/dev.jsonl"
#     db_file = "WikiSQL/data/dev.db"
#     model_out_file = "output/dev_model_out_ko_from_table_3.pkl"
    
    
    ### test (20210505_182728)
#     in_file = "data/wikitestko_from_table_3.jsonl"
#     out_file = f"output/test_out_ko_from_table_3_beam-{args.beam_size}_top-{args.topk}.jsonl"
#     label_file = "WikiSQL/data/test.jsonl"
#     db_file = "WikiSQL/data/test.db"
#     model_out_file = f"output/test_model_out_ko_from_table_3_beam-{args.beam_size}_top-{args.topk}.pkl"
    
    
    # case4 : ko_from_table_not_h_4
    ## dev
#     in_file = "data/wikidevko_from_table_not_h_4.jsonl"
#     out_file = "output/dev_out_ko_from_table_not_h_4.jsonl"
#     label_file = "WikiSQL/data/dev.jsonl"
#     db_file = "WikiSQL/data/dev.db"
#     model_out_file = "output/dev_model_out_ko_from_table_not_h_4.pkl"
    
    
    ### test (20210505_235209)
    in_file = "data/wikitestko_from_table_not_h_4.jsonl"
    out_file = f"output/test_out_ko_from_table_not_h_4_beam-{args.beam_size}_top-{args.topk}.jsonl"
    label_file = "WikiSQL/data/test.jsonl"
    db_file = "WikiSQL/data/test.db"
    model_out_file = f"output/test_model_out_ko_from_table_not_h_4_beam-{args.beam_size}_top-{args.topk}.pkl"
    
###================================================================================================###
    
    # All Best
    model_path = "output/20210505_235209"
    epoch = 4

    engine = DBEngine(db_file)
    config = utils.read_conf(os.path.join(model_path, "model.conf"))
    # config["DEBUG"] = 1
    featurizer = HydraFeaturizer(config)
    pred_data = SQLDataset(in_file, config, featurizer, False)
    print("num of samples: {0}".format(len(pred_data.input_features)))
    
    
    ##======================EG + TOP_k=============================##

    model = create_model(config, is_train=False)
    model.load(model_path, epoch)

    if "DEBUG" in config:
        model_out_file = model_out_file + ".partial"

    if os.path.exists(model_out_file):
        model_outputs = pickle.load(open(model_out_file, "rb"))
    else:
        model_outputs = model.dataset_inference(pred_data)
        pickle.dump(model_outputs, open(model_out_file, "wb"))


    beam_size = args.beam_size
    top_k = args.topk

    print("===HydraNet+EG===")
    print(f"beam_size : {beam_size}, top_k : {top_k}")
    
    pred_sqls = model.predict_SQL_with_EG(engine, pred_data, model_outputs=model_outputs, beam_size=beam_size, top_k=top_k)
    
    with open(out_file + ".eg", "w") as g:
                
        idx = 0
        tmp = []
        for pred_sql in pred_sqls:
            
            idx += 1
            
            sub_query = dict()
            result_k = {"query" : dict()}
            
            sub_query['agg'] = int(pred_sql[0])
            sub_query['sel'] = int(pred_sql[1])
            sub_query["conds"] = [cond for cond in pred_sql[2]]
            tmp.append(sub_query)
            
            if idx == (top_k):
                
                for i in range(top_k):
                    # sub_query[i] = {"0": {"agg": 0, "sel": 3, "conds": [5, 0, "butler cc (ks)"]}
                    result_k['query'][i] = tmp[i]
                    
                idx = 0
                tmp = []
            
                g.write(json.dumps(result_k, cls=NpEncoder) + "\n")
            
    print(f"{out_file+'.eg'} is saved for all pred_sqls. wikisql_prediction.py step is finished")
    
    ##======================EG + TOP_k=============================##


# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018


import os, sys, argparse, re, json

from matplotlib.pylab import *


import torch.nn as nn
import torch
import torch.nn.functional as F
# import torchvision.datasets as dsets

from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl

from sqlnet.dbengine import DBEngine

from torch.utils.tensorboard import SummaryWriter

# BERT
# import bert.tokenization as tokenization
# from bert.modeling import BertConfig, BertModel
from transformers import BertConfig, BertModel, BertTokenizer
from sqlova.model.nl2sql.wikisql_models import *

from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')



def construct_hyper_param(parser, notebook=False):
    parser.add_argument('--datadir', default='./data/ko_token/', type=str, help='data directory')
    parser.add_argument('--logdir', default='./logs/ko_token/', type=str, help='log directory')
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_infer', default=False, action='store_true')
    parser.add_argument('--infer_loop', default=False, action='store_true')

    parser.add_argument("--trained", default=False, action='store_true')
    parser.add_argument('--tepoch', default=100, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--bert_name", 
                        type=str, 
                        default='bert-base-uncased', 
                        choices=['bert-base-multilingual-cased','bert-base-uncased'], 
                        help='bert pretrained model name')
    parser.add_argument("--max_seq_length",
                        default=222, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")
    # gpu
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')

    args = parser.parse_args(args=[]) if notebook else parser.parse_args()

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args

def get_bert(name, device):
    
    # # huggingface 
    bert_config = BertConfig.from_pretrained(name)
    model_bert = BertModel.from_pretrained(name)
    model_bert.config.output_hidden_states = True # return all hidden states
    tokenizer = BertTokenizer.from_pretrained(name)
    
    model_bert.to(device)

    return model_bert, tokenizer, bert_config



def get_models(args, trained=False, path_model_bert=None, path_model=None, device='cpu'):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f'BERT: pretrained {args.bert_name}')
    print(f"BERT: learning rate: {args.lr_bert}")
    print(f"BERT: Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(name=args.bert_name, device=device)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops, device=device)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None
        
        res = torch.load(path_model_bert, map_location=device)
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        res = torch.load(path_model, map_location=device)
        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config


def get_opt(args, model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert




def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train', device='cpu'):
    model.train()
    model_bert.train()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_wv = 0  # of where-value
    cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0  # of execution acc

    # Engine for SQL querying.
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(tqdm(train_loader, desc='TRAIN')):
        cnt += len(t)

        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers, device=device)

        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            #
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, device=device)

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out


def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
          f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
          f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
    print(f'===============================')


def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test', device='cpu'):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers, device=device)
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)

            # get loss & step
            loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size,
                                                                                            device=device)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None
            loss = torch.tensor([0])

        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list1, current_cnt)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


def tokenize_corenlp(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1):
        for tok in sentence:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def infer(nlu1,
          table_name, data_table, path_db, db_name,
          model, model_bert, bert_config, max_seq_length, num_target_layers,
          beam_size=4, show_table=False, show_answer_only=False):
    # I know it is of against the DRY principle but to minimize the risk of introducing bug w, the infer function introuced.
    model.eval()
    model_bert.eval()
    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))

    # Get inputs
    nlu = [nlu1]
    # nlu_t1 = tokenize_corenlp(client, nlu1)
    nlu_t1 = tokenize_corenlp_direct_version(client, nlu1)
    nlu_t = [nlu_t1]

    tb1 = data_table[0]
    hds1 = tb1['header']
    tb = [tb1]
    hds = [hds1]
    hs_t = [[]]

    wemb_n, wemb_h, l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx \
        = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                        num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                    l_hs, engine, tb,
                                                                                    nlu_t, nlu_tt,
                                                                                    tt_to_t_idx, nlu,
                                                                                    beam_size=beam_size,
                                                                                    device=device)

    # sort and generate
    pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
    if len(pr_sql_i) != 1:
        raise EnvironmentError
    pr_sql_q1 = generate_sql_q(pr_sql_i, [tb1])
    pr_sql_q = [pr_sql_q1]

    try:
        pr_ans, _ = engine.execute_return_query(tb[0]['id'], pr_sc[0], pr_sa[0], pr_sql_i[0]['conds'])
    except:
        pr_ans = ['Answer not found.']
        pr_sql_q = ['Answer not found.']

    if show_answer_only:
        print(f'Q: {nlu[0]}')
        print(f'A: {pr_ans[0]}')
        print(f'SQL: {pr_sql_q}')

    else:
        print(f'START ============================================================= ')
        print(f'{hds}')
        if show_table:
            print(engine.show_table(table_name))
        print(f'nlu: {nlu}')
        print(f'pr_sql_i : {pr_sql_i}')
        print(f'pr_sql_q : {pr_sql_q}')
        print(f'pr_ans: {pr_ans}')
        print(f'---------------------------------------------------------------------')

    return pr_sql_i, pr_ans


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    torch_seed(args.seed)

    # gpu
    GPU_NUM = args.gpu # select gpu number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    ## 2. Paths
    path_h = args.datadir # './data'  # '/home/wonseok'
    path_wikisql = args.datadir  #  './data' # os.path.join(path_h, 'data', 'wikisql_tok')

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    path_save_for_evaluation = args.logdir

    ## 3. Load data

    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)

    ## 4. Build & Load models
    if not args.trained:
        model, model_bert, tokenizer, bert_config = get_models(args, 
                                                              device=device)
    else:
        # start epoch
        train_info = json.load(open(os.path.join(args.logdir, 'train_info.json'),'r'))
        args.start_epoch = train_info['last_epoch'] + 1
        args.tepoch = args.start_epoch + args.tepoch 

        # To start from the pre-trained models, un-comment following lines.
        path_model_bert = os.path.join(args.logdir, 'model_bert_best.pt')
        path_model = os.path.join(args.logdir, 'model_best.pt')
        model, model_bert, tokenizer, bert_config = get_models(args, 
                                                               trained=True,
                                                               path_model_bert=path_model_bert, 
                                                               path_model=path_model, 
                                                               device=device)

    ## 5. Get optimizers
    if args.do_train:
        opt, opt_bert = get_opt(args, model, model_bert, args.fine_tune)

        # History
        writer = SummaryWriter(args.logdir)

        ## 6. Train
        acc_lx_t_best = -1
        epoch_best = -1
        for epoch in range(args.start_epoch, args.tepoch):
            # train
            print('epochs: ',epoch)
            acc_train, aux_out_train = train(train_loader,
                                             train_table,
                                             model,
                                             model_bert,
                                             opt,
                                             bert_config,
                                             tokenizer,
                                             args.max_seq_length,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0,
                                             path_db=path_wikisql,
                                             dset_name='train',
                                             device=device)

            # check DEV
            with torch.no_grad():
                acc_dev, results_dev, cnt_list = test(dev_loader,
                                                      dev_table,
                                                      model,
                                                      model_bert,
                                                      bert_config,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      args.num_target_layers,
                                                      detail=False,
                                                      path_db=path_wikisql,
                                                      st_pos=0,
                                                      dset_name='dev', 
                                                      EG=args.EG,
                                                      device=device)

            print_result(epoch, acc_train, 'train')
            print_result(epoch, acc_dev, 'dev')

            # tensorboard
            writer.add_scalar('Train/Loss', acc_train[0], epoch)
            writer.add_scalar('Train/Acc SC', acc_train[1], epoch)
            writer.add_scalar('Train/Acc SA', acc_train[2], epoch)
            writer.add_scalar('Train/Acc WN', acc_train[3], epoch)
            writer.add_scalar('Train/Acc WC', acc_train[4], epoch)
            writer.add_scalar('Train/Acc WO', acc_train[5], epoch)
            writer.add_scalar('Train/Acc WV_idx', acc_train[6], epoch)
            writer.add_scalar('Train/Acc WV', acc_train[7], epoch)
            writer.add_scalar('Train/Acc Logical', acc_train[8], epoch)
            writer.add_scalar('Train/Acc Execute', acc_train[9], epoch)
            
            writer.add_scalar('Dev/Loss', acc_dev[0], epoch)
            writer.add_scalar('Dev/Acc SC', acc_dev[1], epoch)
            writer.add_scalar('Dev/Acc SA', acc_dev[2], epoch)
            writer.add_scalar('Dev/Acc WN', acc_dev[3], epoch)
            writer.add_scalar('Dev/Acc WC', acc_dev[4], epoch)
            writer.add_scalar('Dev/Acc WO', acc_dev[5], epoch)
            writer.add_scalar('Dev/Acc WV_idx', acc_dev[6], epoch)
            writer.add_scalar('Dev/Acc WV', acc_dev[7], epoch)
            writer.add_scalar('Dev/Acc Logical', acc_dev[8], epoch)
            writer.add_scalar('Dev/Acc Execute', acc_dev[9], epoch)

            # save results for the official evaluation
            save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

            # save best model and train information
            # Based on Dev Set logical accuracy lx
            acc_lx_t = acc_dev[-2]
            train_info = {'last_epoch':epoch}

            if acc_lx_t > acc_lx_t_best:
                acc_lx_t_best = acc_lx_t
                epoch_best = epoch

                train_info['best_acc_lx'] = acc_lx_t_best
                train_info['best_epoch'] = epoch_best
                
                # save best model
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join(args.logdir, f'model_best.pt'))

                state = {'model_bert': model_bert.state_dict()}
                torch.save(state, os.path.join(args.logdir, f'model_bert_best.pt'))

            # save train info
            json.dump(train_info, open(os.path.join(args.logdir, 'train_info.json'), 'w'))

            print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")

    if args.do_infer:
        # To use recent corenlp: https://github.com/stanfordnlp/python-stanford-corenlp
        # 1. pip install stanford-corenlp
        # 2. download java crsion
        # 3. export CORENLP_HOME=/Users/wonseok/utils/stanford-corenlp-full-2018-10-05

        # from stanza.nlp.corenlp import CoreNLPClient
        # client = CoreNLPClient(server='http://localhost:9000', default_annotators='ssplit,tokenize'.split(','))

        import corenlp

        client = corenlp.CoreNLPClient(annotators='ssplit,tokenize'.split(','))

        nlu1 = "Which company have more than 100 employees?"
        path_db = './data_and_model'
        db_name = 'ctable'
        data_table = load_jsonl('./data_and_model/ctable.tables.jsonl')
        table_name = 'ftable1'
        n_Q = 100000 if args.infer_loop else 1
        for i in range(n_Q):
            if n_Q > 1:
                nlu1 = input('Type question: ')
            pr_sql_i, pr_ans = infer(
                nlu1,
                table_name, data_table, path_db, db_name,
                model, model_bert, bert_config, max_seq_length=args.max_seq_length,
                num_target_layers=args.num_target_layers,
                beam_size=1, show_table=False, show_answer_only=False
            )

#!/usr/bin/env python3
# docker run --name corenlp -d -p 9000:9000 vzhong/corenlp-server
# Wonseok Hwang. Jan 6 2019, Comment added
# Jaehyuk Heo. May 25 2021, Edited
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import ujson as json

from tqdm import tqdm
import copy
from wikisql.lib.common import count_lines, detokenize
from wikisql.lib.query import Query
from konlpy.tag import Mecab


def find_sub_list(sl, l):
    # from stack overflow.
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results

def check_wv_tok_in_nlu_tok(wv_tok1, nlu_t1):
    """
    Jan.2019: Wonseok
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.

    return:
    st_idx of where-value string token in nlu under CoreNLP tokenization scheme.
    """
    g_wvi1_corenlp = []
    nlu_t1_low = [tok.lower() for tok in nlu_t1]
    for i_wn, wv_tok11 in enumerate(wv_tok1):
        wv_tok11_low = [tok.lower() for tok in wv_tok11]
        results = find_sub_list(wv_tok11_low, nlu_t1_low)
        st_idx, ed_idx = results[0]

        g_wvi1_corenlp.append( [st_idx, ed_idx] )

    return g_wvi1_corenlp



def annotate_example_tootouch(example, table):
    """
    Apr. 2021: Jaehyuk
    Annotate only the information that will be used in our model.
    """

    # tokenizer
    tokenizer = Mecab()

    ann = {'table_id': example['table_id'],'phase': example['phase']}
    ann['question'] = example['question']
    ann['question_tok'] = [str(q).lower() for q in tokenizer.morphs(example['question'])]
    # ann['table'] = {
    #     'header': [annotate(h) for h in table['header']],
    # }
    ann['sql'] = example['sql']
    ann['query'] = copy.deepcopy(example['sql'])

    conds1 = ann['sql']['conds']
    wv_ann1 = []
    for conds11 in conds1:
        wv_ann11 = tokenizer.morphs(str(conds11[2]))
        wv_ann1.append( wv_ann11 )

        # Check whether wv_ann exsits inside question_tok

    try:
        wvi1_corenlp = check_wv_tok_in_nlu_tok(wv_ann1, ann['question_tok'])
        ann['wvi_corenlp'] = wvi1_corenlp
    except:
        ann['wvi_corenlp'] = None
        ann['tok_error'] = 'SQuAD style st, ed are not found under CoreNLP.'

    return ann


def is_valid_example(e):
    if not all([h['words'] for h in e['table']['header']]):
        return False
    headers = [detokenize(h).lower() for h in e['table']['header']]
    if len(headers) != len(set(headers)):
        return False
    input_vocab = set(e['seq_input']['words'])
    for w in e['seq_output']['words']:
        if w not in input_vocab:
            print('query word "{}" is not in input vocabulary.\n{}'.format(w, e['seq_input']['words']))
            return False
    input_vocab = set(e['question']['words'])
    for col, op, cond in e['query']['conds']:
        for w in cond['words']:
            if w not in input_vocab:
                print('cond word "{}" is not in input vocabulary.\n{}'.format(w, e['question']['words']))
                return False
    return True


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', default='/Users/wonseok/data/WikiSQL-1.1/data', help='data directory')
    parser.add_argument('--dout', default='/Users/wonseok/data/wikisql_tok', help='output directory')
    parser.add_argument('--split', default='train,dev,test', help='comma=separated list of splits to process')
    args = parser.parse_args()

    answer_toy = not True
    toy_size = 10

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # for split in ['train', 'dev', 'test']:
    for split in args.split.split(','):
        fsplit = os.path.join(args.din, split) + '.jsonl'
        ftable = os.path.join(args.din, split) + '.tables.jsonl'
        fout = os.path.join(args.dout, split) + '_tok.jsonl'

        print('annotating {}'.format(fsplit))
        with open(fsplit) as fs, open(ftable) as ft, open(fout, 'w', encoding='utf-8') as fo:
            print('loading tables')

            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                d = json.loads(line)
                tables[d['id']] = d
            print('loading examples')

            n_written = 0
            cnt = -1
            for line in tqdm(fs, total=count_lines(fsplit)):
                cnt += 1
                d = json.loads(line)
                
                a = annotate_example_tootouch(d, tables[d['table_id']])
                fo.write(json.dumps(a, ensure_ascii=False) + '\n')
                n_written += 1

                if answer_toy:
                    if cnt > toy_size:
                        break
            print('wrote {} examples'.format(n_written))

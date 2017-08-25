#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# File: /Users/hain/ai/InsuranceQA-Machine-Learning/deep_qa_1/network.py
# Author: Hai Liang Wang
# Date: 2017-08-08:18:32:05
#
#===============================================================================

"""
   A data API for learning QA.
   
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 Hai Liang Wang. All Rights Reserved"
__author__    = "Hai Liang Wang"
__modify__    = "Xuming Lin"
__date__      = "2017-08-08:18:32:05"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

import random
import insuranceqa_data as insuranceqa

import numpy as np

_train_data = insuranceqa.load_pairs_train()
_test_data = insuranceqa.load_pairs_test()
_valid_data = insuranceqa.load_pairs_valid()



'''
build vocab data with more placeholder
'''
vocab_data = insuranceqa.load_pairs_vocab()
vocab_size = len(vocab_data['word2id'].keys())
VOCAB_PAD_ID = vocab_size+1
VOCAB_GO_ID = vocab_size+2
vocab_data['word2id']['<PAD>'] = VOCAB_PAD_ID
vocab_data['word2id']['<GO>'] = VOCAB_GO_ID
vocab_data['id2word'][VOCAB_PAD_ID] = '<PAD>'
vocab_data['id2word'][VOCAB_GO_ID] = '<GO>'


def combine_pos_and_neg_sample(data):
    '''
    combine the positive answers and negative samples with the same problem
    '''
    qa = dict()
    for x in data:
        qa.setdefault(x['qid'], ["", [], []])
        qa[x['qid']][0] = x['question']
        if x['label'] == [0, 1]:
            qa[x['qid']][1].append(x['utterance'])
        else:
            qa[x['qid']][2].append(x['utterance'])
    result = list()
    for qid in qa:
        question = qa[qid][0]
        for pos_a in qa[qid][1]:
            for neg_a in qa[qid][2]:
                result.append({'qid': qid, 'question': question, 'pos_utterance': pos_a, 'neg_utterance': neg_a})
    return result

_train_data = combine_pos_and_neg_sample(_train_data)

def _get_corpus_metrics():
    '''
    max length of questions
    '''
    for cat, data in zip(["valid", "test", "train"], [_valid_data, _test_data, _train_data]):
        max_len_question = 0
        total_len_question = 0
        max_len_utterance = 0
        total_len_utterance = 0
        for x in data:
            total_len_question += len(x['question']) 
            total_len_utterance += len(x['utterance'])
            if len(x['question']) > max_len_question: 
                max_len_question = len(x['question'])
            if len(x['utterance']) > max_len_utterance: 
                max_len_utterance = len(x['utterance'])
        print('max len of %s question : %d, average: %d' % (cat, max_len_question, total_len_question/len(data)))
        print('max len of %s utterance: %d, average: %d' % (cat, max_len_utterance, total_len_utterance/len(data)))
    # max length of answers


class BatchIter():
    '''
    Load data with mini-batch
    '''
    def __init__(self, data = None, batch_size = 100):
        assert data is not None, "data should not be None."
        self.batch_size = batch_size
        self.data = data

    def next(self):
        random.shuffle(self.data)
        index = 0
        total_num = len(self.data)
        while index <= total_num:
            yield self.data[index:index + self.batch_size]
            index += self.batch_size

def padding(lis, pad, size):
    '''
    right adjust a list object
    '''
    if size > len(lis):
        lis += [pad] * (size - len(lis))
    else:
        lis = lis[0:size]
    return lis

def pack_question_n_utterance(q, p_u, n_u=None, q_length = 20, u_length = 99):
    '''
    combine question and utterance as input data for feed-forward network
    '''
    assert len(q) > 0 and len(p_u) > 0, "question and utterance must not be empty"
    q = padding(q, VOCAB_PAD_ID, q_length)
    p_u = padding(p_u, VOCAB_PAD_ID, u_length)
    assert len(q) == q_length, "question should be pad to q_length"
    assert len(p_u) == u_length, "utterance should be pad to u_length"
    if n_u is not None:
        assert len(n_u) > 0, "negative utterance must not be empty"
        n_u = padding(n_u, VOCAB_PAD_ID, u_length)
        assert len(n_u) == u_length, "negative utterance should be pad to u_length"
        return q, p_u, n_u
    return q, p_u

def __resolve_train_data(data, batch_size, question_max_length = 20, utterance_max_length = 99):
    '''
    resolve train data
    '''
    batch_iter = BatchIter(data = data, batch_size = batch_size)
    
    for mini_batch in batch_iter.next():
        qids = []
        questions = []
        pos_answers = []
        neg_answers = []
        for o in mini_batch:
            q, pu, nu = pack_question_n_utterance(o['question'], o['pos_utterance'], o['neg_utterance'], question_max_length, utterance_max_length)
            qids.append(o['qid'])
            questions.append(q)
            pos_answers.append(pu)
            neg_answers.append(nu)
        if len(questions) > 0:
            yield qids, questions, pos_answers, neg_answers
        else:
            raise StopIteration

def __resolve_valid_data(data, batch_size, question_max_length = 20, utterance_max_length = 99):
    '''
    resolve valid data
    '''
    batch_iter = BatchIter(data = data, batch_size = batch_size)
    
    for mini_batch in batch_iter.next():
        qids = []
        questions = []
        answers = []
        labels = []
        for o in mini_batch:
            q, pu = pack_question_n_utterance(o['question'], o['utterance'], None, question_max_length, utterance_max_length)
            qids.append(o['qid'])
            questions.append(q)
            answers.append(pu)
            labels.append(np.argmax(o['label']))
        if len(questions) > 0:
            # print('data in batch:%d' % len(mini_batch))
            yield qids, questions, answers, labels
        else:
            raise StopIteration

# export data

def load_train(batch_size = 100, question_max_length = 20, utterance_max_length = 99):
    '''
    load train data
    '''
    return __resolve_train_data(_train_data, batch_size, question_max_length, utterance_max_length)

def load_test(question_max_length = 20, utterance_max_length = 99):
    '''
    load test data
    '''
    questions = []
    answers = []
    labels = []
    qids = []
    for o in _test_data:
        qid = o['qid']
        q, pu = pack_question_n_utterance(o['question'], o['utterance'], None, question_max_length, utterance_max_length)
        qids.append(qid)
        questions.append(q)
        answers.append(pu)
        labels.append(np.argmax(o['label']))
    return qids, questions, answers, labels

def load_valid(batch_size = 100, question_max_length = 20, utterance_max_length = 99):
    '''
    load valid data
    '''
    return __resolve_valid_data(_valid_data, batch_size, question_max_length, utterance_max_length)

def test_batch():
    '''
    retrieve data with mini batch
    '''
    for mini_batch in zip(load_train()):
        for qid, q, pos_a, neg_a in mini_batch:
            print(q[0])
            print(pos_a[0])
            print(neg_a[0])
            break
        break
    for mini_batch in zip(load_valid()):
        for qid, q, pos_a, labels in mini_batch:
            print(q[0])
            print(pos_a[0])
            print(labels[0])
            break
        break
    for (qid, q, pos_a, label) in zip(*load_test()):
        print(q)
        print(pos_a)
        print(label)
        break

    print("VOCAB_PAD_ID", VOCAB_PAD_ID)
    print("VOCAB_GO_ID", VOCAB_GO_ID)

if __name__ == '__main__':
    test_batch()


#! /usr/bin/env python3

import sys
import os
import argparse
from collections import defaultdict
from math import log

"""this function is for multi predictions with one true label. 
 includes precision, recall, recall_at_k, Discounted Cumulative Gain and top_k_accuracy(by aggregating recall@k)
 input prediction file should contains k predictions in one line"""

def get_lines(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f]
    return lines

def compute_DCG(gold_label, system_label):  # system label must be a list of predictions and contain gold_label
    correct_index = system_label.index(gold_label) + 1
    return 1/(log(correct_index + 1, 2))

def compute_counts(gold, system):
    assert len(gold) == len(system)

    recall_tp, recall_fn, recall_at_k_tp, recall_at_k_fn = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    DCG = defaultdict(int)

    for gold_label, system_label in zip(gold, system):
        if gold_label == system_label[0]:   # if matches first prediction
            recall_tp[gold_label] += 1
        else:
            recall_fn[gold_label] += 1
        if gold_label in system_label:  # if any of the k predictions matches
            recall_at_k_tp[gold_label] += 1
            DCG[gold_label] += compute_DCG(gold_label, system_label)    # sum all DCG for each tag, find avg later
        else:
            recall_at_k_fn[gold_label] += 1

    precision_tp, precision_fp = defaultdict(int), defaultdict(int)
    
    for system_label, gold_label in zip(system, gold):
        if system_label[0] == gold_label:   # if matches first prediction
            precision_tp[system_label[0]] += 1
        else:
            precision_fp[system_label[0]] += 1

    assert sum(recall_tp.values()) == sum(precision_tp.values())

    return recall_tp, precision_fp, recall_fn, recall_at_k_tp, recall_at_k_fn, DCG

def compute_performance(tp, fp, fn):
    if tp == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = 100 * (tp / (tp + fp))
        recall = 100 * (tp / (tp + fn))
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def compute_recall_at_k(cm_tp, cm_fn):
    if cm_tp == 0:
        recall_at_k = 0.0
    else:
        recall_at_k = 100 * (cm_tp / (cm_tp + cm_fn))
    return recall_at_k

def compute_stats(gold, system):
    recall_tp, precision_fp, recall_fn, recall_at_k_tp, recall_at_k_fn, DCG = compute_counts(gold, system)

    gold_tags = sorted(set(gold))
    stats = {}
    for tag in gold_tags:
        tp = recall_tp[tag]
        fp = precision_fp[tag]
        fn = recall_fn[tag]
        precision, recall, _ = compute_performance(tp, fp, fn)
        cm_tp = recall_at_k_tp[tag]
        cm_fn = recall_at_k_fn[tag]
        recall_at_k = compute_recall_at_k(cm_tp, cm_fn)
        DCG_sum = DCG[tag]
        DCG_avg = DCG_sum/(tp + fn)
        stats[tag] = (precision, recall, recall_at_k, DCG_avg, tp, fp, cm_tp, cm_fn, cm_tp + cm_fn)

    tp = sum(value for tag, value in recall_tp.items() if tag in gold_tags)
    fp = sum(value for tag, value in precision_fp.items() if tag in gold_tags)
    fn = sum(value for tag, value in recall_fn.items() if tag in gold_tags)
    precision, recall, f1 = compute_performance(tp, fp, fn)
    recall_at_k_tp = sum(value for tag, value in recall_at_k_tp.items() if tag in gold_tags)
    recall_at_k_fn = sum(value for tag, value in recall_at_k_fn.items() if tag in gold_tags)
    recall_at_k = compute_recall_at_k(recall_at_k_tp, recall_at_k_fn)
    recall_at_k_list = [x[2] for x in stats.values()]
    recall_at_k_macro = sum(recall_at_k_list)/len(recall_at_k_list)
    DCG_avg = sum(value for tag, value in DCG.items() if tag in gold_tags)/(tp + fn)
    stats['overall'] = (precision, recall, recall_at_k, DCG_avg, tp, fp, recall_at_k_tp, recall_at_k_fn, recall_at_k_tp + recall_at_k_fn)

    return stats, recall_at_k, recall_at_k_macro

def get_result_output(stats):
    output = []

    output.append(144 * '-')
    output.append('| {0:^23} | {1:^10} | {2:^10} | {3:^10} | {4:^10} | {5:^10} | {6:^9} | {7:^12} | {8:^12} | {9:^10} |'.format('Label', 'Prec', 'Rec', 'Recall@k', 'DCG','TP', 'FP', 'recall@k_tp', 'recall@k_fn', 'sample_num'))
    labels = dict(stats)
    del labels['overall']
    for i in sorted(labels.items(), key=lambda x: x[1][2], reverse=True):   # sort by recall_at_k
        tag = i[0]
        precision, recall, recall_at_k, dcg, tp, fp, recall_at_k_tp, recall_at_k_fn, sample_num = i[1]
        output.append(144 * '-')
        output.append('| {0:^23} | {1:^10.5f} | {2:^10.5f} | {3:^10.5f} | {4:^10.5f} | {5:^10} | {6:^9} | {7:^12} | {8:^12} | {9:^10} |'.format(tag, precision, recall, recall_at_k, dcg, tp, fp, recall_at_k_tp, recall_at_k_fn, sample_num))
    output.append(144 * '-')
    tag = 'overall'
    precision, recall, recall_at_k, dcg, tp, fp, recall_at_k_tp, recall_at_k_fn, sample_num = stats[tag]
    output.append('| {0:^23} | {1:^10.5f} | {2:^10.5f} | {3:^10.5f} | {4:^10.5f} | {5:^10} | {6:^9} | {7:^12} | {8:^12} | {9:^10} |'.format(tag, precision, recall, recall_at_k, dcg, tp, fp, recall_at_k_tp, recall_at_k_fn, sample_num))
    output.append(144 * '-')

    return '\n'.join(output)

def get_accuracy(stats):
    _, _, _, _, _, tp, _, _, _, tp_fn = stats['overall']
    return 100 * tp / tp_fn

def get_macro_f1(stats, num_tags):
    labels = dict(stats)
    del labels['overall']
    total_f1 = sum(elems[2] for elems in labels.values())
    return total_f1 / num_tags

def score(args):
    gold = [line.replace('__label__', '') for line in get_lines(args.gold)]
    system = [line.replace('__label__', '').split(' ') for line in get_lines(args.system)]

    if len(gold) != len(system):
        sys.stderr.write('Number of lines are different: {}, {}\n'.format(len(gold), len(system)))
        return

    stats, top_k_accuracy, recall_at_k_macro = compute_stats(gold, system)

    result_output = get_result_output(stats)
    sys.stdout.write('{}\n'.format(result_output))

    sys.stdout.write('Top_k_accuracy(micro): {:.5f}\n'.format(top_k_accuracy)) # same as recall_at_k
    sys.stdout.write('Top_k_accuracy(macro): {:.5f}\n'.format(recall_at_k_macro)) # same as recall_at_k_macro

    # accuracy = get_accuracy(stats)
    # sys.stdout.write('accuracy: {:.5f}\n'.format(accuracy))
    # macro_f1 = get_macro_f1(stats, len(set(gold)))
    # sys.stdout.write('Macro F-Score: {:.5f}\n'.format(macro_f1))

def main():
    parser = argparse.ArgumentParser(description='Score system output.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gold', help='gold file')
    parser.add_argument('system', help='system file')
    args = parser.parse_args()
    score(args)
    return 0

if __name__ == '__main__':
    sys.exit(main())

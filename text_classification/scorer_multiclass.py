# -*- coding: utf-8 -*-

import sys
import os
import argparse
from collections import defaultdict, Counter
from math import log
import pandas as pd
from math import ceil

from sklearn.metrics import roc_auc_score
"""
output prediction metrics to csv. 
for multi predictions with one true label.

if threshold:  Only compute for classes with fisrt probability > threshold
if tuning_precision: tune threshold for each class for desired precision

columns: ['label', 'precision@1', 'recall@1', 'recall@3', 'recall@5', 'DCG', 'tp@1', 'fp@1', 'tp@5', 'fn@5', 'predicted_test_num', 'total_test_num']
"""

class DataLoader(object):
    def __init__(self, gold, system):
        self.gold_path = gold
        self.system_path = system

    def load_data(self):
        pass

    def load_data_fasttext(self):   # gold and fasttext predictions
        with open(self.gold_path, 'r', encoding='utf-8-sig') as f:
            gold_tmp = [line.strip() for line in f.readlines()]
        with open(self.system_path, 'r', encoding='utf-8-sig') as f:
            system_tmp = [line.strip() for line in f.readlines()]
            system_tmp = list(map(lambda x: x.split(' '), system_tmp))
            system_tmp = list(map(lambda x: [[x[i].replace('__label__', ''), float(x[i + 1])] for i in range(0, len(x), 2)], system_tmp))
        if len(gold_tmp) != len(system_tmp):
            print('Number of lines are different: {}, {}\n'.format(len(gold_tmp), len(system_tmp)))
            return
        gold, system = [], []
        # remove '无法确定' and change str to float
        for i, v in enumerate(system_tmp):
            if gold_tmp[i] != '无法确定':  # if 预测为无法确定或是label为无法确定，ignore
                gold.append(gold_tmp[i])
                system.append([[x[0]] + [x[1]] for x in v if x[0] != '无法确定'])
        return gold, system
    # end def

class ThresholdFinder(object):  # find threshold for desired precision@1
    def __init__(self, gold, system, total_gold_counts, precision, tuned_thres_path):
        self.gold = gold
        self.system = system
        self.total_gold_counts = total_gold_counts
        self.precision = precision
        self.tuned_thres_path = tuned_thres_path

    def process(self):
        for label in list(set(self.gold)):
            right_pred = []
            wrong_pred = []
            for i, v in enumerate(self.system):
                if v[0] == label:
                    if self.gold[i] == label:
                        right_pred.append(v)
                    else:
                        wrong_pred.append(v)
            if not right_pred:  # no right_pred
                thres = 0.99
            elif not wrong_pred:  # all right_pred
                if len(right_pred) > 10:  # must check whether test_size if enough
                    thres = 0.0
                else:
                    thres = 0.2
            else:
                total_count = self.total_gold_counts[label]
                thres = self.find_threshold(right_pred, wrong_pred, total_count)
            with open(self.tuned_thres_path, 'a', encoding='utf-8-sig') as f:
                thres = str(ceil(thres * 10000) / 10000.0)  # round up and keep 4 decimal
                f.write(label + '\t' + thres + '\n')
        print('output threshold list to %s' % self.tuned_thres_path)

    def find_threshold(self, right_pred, wrong_pred, total_count):
        thresholds = sorted(set([a[1] for a in right_pred] + [b[1] for b in wrong_pred])) # all probabilities
        if len(thresholds) > 1000:
            # reduce threshold list to 1000 values
            index_range = ceil(len(thresholds) / 1000)
            thresholds = [thresholds[0]] + thresholds[1:-1:index_range] + [thresholds[-1]]
        metrics_dict = {}
        for i, thres in enumerate(thresholds):
            metrics = self.compute_metrics(right_pred, wrong_pred, total_count, thres)
            metrics_dict[thres] = metrics

        # TODO: DRAW

        # find threshold that satisfy given precision:
        precision_list = []
        for thres in list(sorted(thresholds)):
            if thres not in metrics_dict:
                continue
            precision_list.append([thres, metrics_dict[thres]['precision']])
        res = next((x[0] for x in precision_list if x[1] >= self.precision), 0.99)  # if cannot reach desired precision, use thres=0.99
        return res

    def compute_metrics(self, right_pred, wrong_pred, total_count, thres):
        passed_pred_num = len([a for a in right_pred if a[1] >= thres])
        precision = passed_pred_num / (passed_pred_num + len([a for a in wrong_pred if a[1] >= thres]))
        recall = passed_pred_num / total_count
        metrics = {'precision': precision, 'recall': recall}
        return metrics


class Scorer(object):
    def __init__(self, gold, system, total_gold_counts, outputpath, system_score = None):
        self.gold = gold
        self.system = system
        self.system_score = system_score
        self.total_gold_counts = total_gold_counts
        self.outputpath = outputpath    # output of score

    def score(self):
        stats = self.compute_stats()
        aucs = self.compute_auc()
        df_out = self.get_result_output(stats, aucs)
        df_out.to_csv(self.outputpath, encoding='utf-8-sig', index=False)

    def compute_DCG(self, gold_label, system_label):  # system label must be a list of predictions and contain gold_label
        correct_index = system_label.index(gold_label) + 1
        return 1 / (log(correct_index + 1, 2))

    def compute_counts(self):
        assert len(self.gold) == len(self.system)

        recall_tp, recall_at_3_tp, recall_at_5_tp = defaultdict(int), defaultdict(int), defaultdict(int)
        DCG = defaultdict(int)

        for gold_label, system_label in zip(self.gold, self.system):
            if gold_label == system_label[0]:  # if matches first prediction
                recall_tp[gold_label] += 1
            if gold_label in system_label[:3]:  # if any of the 3 predictions matches
                recall_at_3_tp[gold_label] += 1
            if gold_label in system_label[:5]:  # if any of the 5 predictions matches
                recall_at_5_tp[gold_label] += 1
                DCG[gold_label] += self.compute_DCG(gold_label, system_label)  # sum all DCG for each tag, find avg later

        precision_tp, precision_fp = defaultdict(int), defaultdict(int)

        for system_label, gold_label in zip(self.system, self.gold):
            if system_label[0] == gold_label:  # if matches first prediction
                precision_tp[system_label[0]] += 1
            else:
                precision_fp[system_label[0]] += 1

        assert sum(recall_tp.values()) == sum(precision_tp.values())

        return recall_tp, precision_fp, recall_at_3_tp, recall_at_5_tp, DCG

    def auc_roc(self, tag):
        label_scores = [dict(zip(*label_score)) for label_score in zip(self.system, self.system_score)]
        gold_binary = [1 if label==tag else 0 for label in self.gold]
        pred_score = [0.0 if tag not in label_score else label_score[tag] for label_score in label_scores]
        auc = roc_auc_score(gold_binary, pred_score)
        return auc

    def compute_performance(self, tp, fp, fn):
        if tp == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def compute_recall_at_k(self, cm_tp, cm_fn):
        if cm_tp == 0:
            recall_at_k = 0.0
        else:
            recall_at_k = cm_tp / (cm_tp + cm_fn)
        return recall_at_k

    def compute_stats(self):
        recall_tp, precision_fp, recall_at_3_tp, recall_at_5_tp, DCG = self.compute_counts()

        gold_tags = sorted(set(self.gold))
        n = len(gold_tags)
        stats = {}
        for tag in gold_tags:
            tp = recall_tp[tag]
            fp = precision_fp[tag]
            tag_count = self.total_gold_counts[tag]
            fn = tag_count - tp
            precision, recall, f1 = self.compute_performance(tp, fp, fn)
            tp_at_3 = recall_at_3_tp[tag]
            fn_at_3 = tag_count - tp_at_3
            recall_at_3 = self.compute_recall_at_k(tp_at_3, fn_at_3)
            tp_at_5 = recall_at_5_tp[tag]
            fn_at_5 = tag_count - tp_at_5
            recall_at_5 = self.compute_recall_at_k(tp_at_5, fn_at_5)
            DCG_sum = DCG[tag]
            DCG_avg = DCG_sum / (tp + fn)
            stats[tag] = (
            precision, recall, f1, recall_at_3, recall_at_5, DCG_avg, tp, fp, fn, tp_at_5, fn_at_5, tag_count)

        # overall stats
        tag_stats_list = list(stats.values())
        tp = sum(value for tag, value in recall_tp.items() if tag in gold_tags)
        overall_count = self.total_gold_counts['overall']
        fp = overall_count - tp
        fn = fp
        recall_at_3_tp = sum(value for tag, value in recall_at_3_tp.items() if tag in gold_tags)
        recall_at_3_fn = overall_count - recall_at_3_tp
        recall_at_5_tp = sum(value for tag, value in recall_at_5_tp.items() if tag in gold_tags)
        recall_at_5_fn = overall_count - recall_at_5_tp

        # micro average
        precision, recall, f1 = self.compute_performance(tp, fp, fn)
        recall_at_3 = self.compute_recall_at_k(recall_at_3_tp, recall_at_3_fn)
        recall_at_5 = self.compute_recall_at_k(recall_at_5_tp, recall_at_5_fn)
        DCG_avg = sum(value for tag, value in DCG.items() if tag in gold_tags) / overall_count
        stats['overall_micro'] = (
        precision, recall, f1, recall_at_3, recall_at_5, DCG_avg, tp, fp, fn, recall_at_5_tp, recall_at_5_fn,
        overall_count)

        # macro average
        precision = sum([i[0] for i in tag_stats_list]) / n
        recall = sum([i[1] for i in tag_stats_list]) / n
        f1 = sum([i[2] for i in tag_stats_list]) / n
        recall_at_3 = sum([i[3] for i in tag_stats_list]) / n
        recall_at_5 = sum([i[4] for i in tag_stats_list]) / n
        DCG_avg = sum([i[5] for i in tag_stats_list]) / n
        stats['overall_macro'] = (
            precision, recall, f1, recall_at_3, recall_at_5, DCG_avg, tp, fp, fn, recall_at_5_tp, recall_at_5_fn,
            overall_count)

        # weighted average
        precision = sum([i[0]*i[-1] for i in tag_stats_list]) / overall_count
        recall = sum([i[1] * i[-1] for i in tag_stats_list]) / overall_count
        f1 = sum([i[2] * i[-1] for i in tag_stats_list]) / overall_count
        recall_at_3 = sum([i[3]*i[-1] for i in tag_stats_list]) / overall_count
        recall_at_5 = sum([i[4]*i[-1] for i in tag_stats_list]) / overall_count
        DCG_avg = sum([i[5]*i[-1] for i in tag_stats_list]) / overall_count
        stats['overall_weighted'] = (
            precision, recall, f1, recall_at_3, recall_at_5, DCG_avg, tp, fp, fn, recall_at_5_tp, recall_at_5_fn,
            overall_count)

        return stats

    def compute_auc(self):
        gold_tags = sorted(set(self.gold))
        aucs = {}
        
        for tag in gold_tags:
          aucs[tag] = self.auc_roc(tag)
        tag_auc_list = list(aucs.items())
        aucs['overall_micro'] = 0.0
        aucs['overall_macro'] = sum(aucs.values())/len(aucs)
        aucs['overall_weighted'] = sum([self.total_gold_counts[i[0]] * i[1] for i in tag_auc_list]) / self.total_gold_counts['overall']
        return aucs

    def get_result_output(self, stats, aucs):
        df = pd.DataFrame.from_dict(stats, orient='index', columns=['precision@1', 'recall@1', 'f1', 'recall@3', 'recall@5',
                                                                    'DCG_avg', 'tp', 'fp', 'fn', 'recall@5_tp',
                                                                    'recall@5_fn', 'test_num'])
        df['label'] = df.index
        df['auc_roc'] = df['label'].apply(lambda x: aucs[x] if x in aucs else 0.0)
        df[['precision@1', 'recall@1', 'f1', 'recall@3', 'recall@5', 'DCG_avg', 'auc_roc']] = df[[
            'precision@1', 'recall@1', 'f1', 'recall@3', 'recall@5', 'DCG_avg', 'auc_roc']].applymap(lambda x: float("%0.4f" % (x)))

        df = df[['label', 'auc_roc', 'f1', 'precision@1', 'recall@1', 'recall@3', 'recall@5', 'DCG_avg', 'tp', 'fp', 'fn',
                 'recall@5_tp', 'recall@5_fn', 'test_num']]
        df = df.sort_values(by=['test_num'], ascending=False)
        return df


def process(args):
    print(args)
    gold, system = DataLoader(args.gold, args.system).load_data_fasttext()
    total_gold_counts = Counter(gold)
    total_gold_counts['overall'] = sum(total_gold_counts.values())

    if args.tune_thres_by_prec != 0.0:
        system = [predictions[0] for predictions in system]   # consider 1st prediction
        ThresholdFinder(gold, system, total_gold_counts, args.tune_thres_by_prec, args.tuned_thres_path).process()
    elif args.tuned_thres == 1:   # use tuned_thresholds
            if os.path.isfile(args.tuned_thres_path):
                with open(args.tuned_thres_path, 'r', encoding='utf-8-sig') as f:
                    threshold_dict = dict([line.strip().split('\t') for line in f.readlines()])
                gold_new, system_new, system_score = [], [], []
                for i, v in enumerate(system):
                    if v[0][1] >= float(threshold_dict.get(v[0][0], 0.99)):  # if not found, set max thres = 0.99
                        system_new.append([a[0] for a in v])
                        system_score.append([a[1] for a in v])
                        gold_new.append(gold[i])
                Scorer(gold_new, system_new, total_gold_counts, args.outputpath, system_score).score()
            else:
                print('Threshold file does not exist!')
    else:   # use common thres for all classes
        gold_new, system_new, system_score = [], [], []
        for i, v in enumerate(system):
            if v[0][1] > args.thres:  # if first prediction > threshold
                system_new.append([a[0] for a in v])  # append all predictions
                system_score.append([a[1] for a in v])
                gold_new.append(gold[i])
        Scorer(gold_new, system_new, total_gold_counts, args.outputpath, system_score).score()

def main():
    parser = argparse.ArgumentParser(description='Score multiclass predictions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gold', help='gold file')
    parser.add_argument('system', help='system file')
    parser.add_argument('--thres', type=float, default=0.0, help='use a common threshold')
    parser.add_argument('--tune_thres_by_prec', type=float, default=0.0, help='tune threshold for every class to achieve a particular precision')
    parser.add_argument('--tuned_thres_path', type=str, default='thres/class_threshold.txt', help='path of tuned threshold for each class')
    parser.add_argument('--tuned_thres', type=int, default=0, help='0: use common thres; 1: use tuned thres')
    parser.add_argument('--outputpath', type=str, default='score/india_2020/model1.csv', help='score outputpath')
    args = parser.parse_args()
    process(args)
    return 0

if __name__ == '__main__':
    sys.exit(main())

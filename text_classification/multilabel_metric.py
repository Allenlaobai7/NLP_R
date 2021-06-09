from math import log
from collections import defaultdict

"""this script is for multi predictions with 1 or more true labels. 
 both input files should contain 1-k labels in 1 line.

 for individual label: precision@1, recall_1@k(whether this label is being predicted in k predictions)
 for overall performace: precision@k, recall@k, accuracy and Normalized Discounted Cumulative Gain(NDCG)

 precision@1: precision of each label at prediction 1
 recall_1@k: recall of each label in all predictions
 precision@k: number of correct predictions / number of predictions at k
 recall@k: number of correct predictions / number of true labels
 accuracy: 1 if at least one correct label predicted, 0 otherwise
 NDCG: discounted gain according to the ranks of predicted truths

 """


def compute_global_metrics(gold_label, system_label):
    predicted_truths = list(set(gold_label).intersection(set(system_label)))
    if predicted_truths:
        predicted_truths_rank = [system_label.index(a) + 1 for a in predicted_truths]

        DCG = sum([1 / (log(i + 1, 2)) for i in predicted_truths_rank]) # all predictions with same relevance since we do not include probabilities
        IDCG = sum([1 / (log(i + 1, 2)) for i in range(1, len(gold_label)+1)])
        NDCG = DCG/IDCG

        precision_at_k = len(predicted_truths) / len(system_label)
        recall_at_k = len(predicted_truths) / len(gold_label)
        accuracy = 1.0
    else:
        NDCG, precision_at_k, recall_at_k, accuracy = 0.0, 0.0, 0.0, 0.0

    return NDCG, precision_at_k, recall_at_k, accuracy


def compute_counts(gold, system):
    assert len(gold) == len(system)

    precision_at_1_tp, precision_at_1_fp = defaultdict(int), defaultdict(int)
    recall_1_at_k_tp, recall_1_at_k_fn = defaultdict(int), defaultdict(int)
    NDCG_list, precision_at_k_list, recall_at_k_list, accuracy_list = [], [], [], []

    for gold_label, system_label in zip(gold, system):
        for label in gold_label:
            if label in system_label:    # tp of this label in k predictions
                recall_1_at_k_tp[label] += 1
            else:
                recall_1_at_k_fn[label] += 1
        first_prediction = system_label[0]
        if first_prediction in gold_label:  # tp@1
            precision_at_1_tp[first_prediction] += 1
        else:
            precision_at_1_fp[first_prediction] += 1

        NDCG, precision_at_k, recall_at_k, accuracy = compute_global_metrics(gold_label, system_label)
        NDCG_list.append(NDCG)
        precision_at_k_list.append(precision_at_k)
        recall_at_k_list.append(recall_at_k)
        accuracy_list.append(accuracy)

    return precision_at_1_tp, precision_at_1_fp, recall_1_at_k_tp, recall_1_at_k_fn, \
           NDCG_list, precision_at_k_list, recall_at_k_list, accuracy_list
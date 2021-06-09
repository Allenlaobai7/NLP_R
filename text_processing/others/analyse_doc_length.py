import json
from scipy import stats
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt

# length = []
for line in open('./output/doc_length.txt', 'r', encoding='utf8'):
    doc_length = json.loads(line)

'''summaries of all labels'''
non_empty_count, empty_count = 0, 0
len_summary, doc_length_136 = {},{}
for key, value in doc_length.items():
    try:
        len_summary[key] = stats.describe(value)
        doc_length_136[key] = value
        non_empty_count += 1
    except:
        empty_count += 1

'''for the 136 labels with examples, plot normal plots of all > 1000 obs, group by 20'''
# doc_length = {k: v for k, v in doc_length.items() if v is not []}
print(len(doc_length_136))
doc_summary = OrderedDict(sorted(doc_length_136.items(), key=lambda t: len(t[1]), reverse=True))

top20 = list(doc_summary.items())[0:20]
top20to40 = list(doc_summary.items())[20:40]
top40to60 = list(doc_summary.items())[40:60]
top60to80 = list(doc_summary.items())[60:80]
top80to100 = list(doc_summary.items())[80:100]
top100to120 = list(doc_summary.items())[100:120]   # last one has 7 obs

print(top100to120[-1])
for i in top20:
    sns.kdeplot(i[1], bw=2, label=i[0])
plt.legend()
plt.show()



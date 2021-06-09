'''preprocess'''

from collections import Counter
a = Counter()
a.update({"happy嗬me"})

print(a)

# 箜 分割词      澍 分割词的feature
def convert_str_to_map(string):
    word_dict_list = []
    words = string.split('箜')
    for word in words:
        features = word.split('澍')
        word_dict_list.append({features[0]:features[1]})
    return word_dict_list

string = "dog澍noun澍happy箜unhappy澍adj澍haha"

a = convert_str_to_map(string)
print(a)

def keep_noun_and_adj(word_dict):            # keep only words with pos "noun" or "adj"     # this one for big dict
    res = [k for k, v in word_dict.items() if v in ["noun", "adj"]]
    return res

def keep_noun_and_adj(word_dict_list):    # this one for list of dicts
    res = [list(x.keys())[0] for x in word_dict_list if list(x.values())[0] in ['noun', 'adj']]
    return res

b = keep_noun_and_adj(a)
print(b)


import pandas as pd
from collections import Counter
import math

df= pd.DataFrame({"a": ["one", "two", "three", "four", "two", "three"], "b": ["一", "二", "三", "四", "二", "三"],
                  "words": [["hello", 'ok','not'], ["no",'bad','sad'], ["hap",'wer','shi'],["shi",'smi','hxi']
                      ,["no", 'sad','not'],["ok", 'shi','shi']]})

print(df.head(6))
# def group_set_column(col):
#     res = list(set(col))
#     return res

grouped_df = df.groupby(['a','b']).agg({'words':'sum'}).reset_index()
print(grouped_df)

y_count = df[['a','words']].groupby('a').count().reset_index()
y_count.columns= ['a','count']
y_count['tag_p'] = y_count['count'] / (len(y_count)*[len(df)])

print(y_count)

joined_df = grouped_df.merge(y_count, on = "a")
print(joined_df)

# create words count overall, store in a Counter
words_total = joined_df['words'].sum()
words_total_counter = Counter(words_total)
print(words_total_counter)

# create words count for each tag
joined_df['words_count'] = list(map(lambda x: Counter(x), joined_df['words']))

# calculate pmi
def calculate_pmi(words_count, tag_p):
    # create a dict to store pmi"
    res = {}
    for key, value in words_count.items():
        pmi = math.log(value/(words_total_counter[key]*tag_p))   # p(x,y)/(p(x)*p(y))
        res[key] = pmi
    return res

joined_df['pmi_score'] = joined_df.apply(lambda x: calculate_pmi(x.words_count, x.tag_p), axis=1)

joined_df.to_csv('./test.csv')
print(joined_df)

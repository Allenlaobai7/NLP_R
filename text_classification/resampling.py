# utf-8
import pandas as pd
import numpy as np
from sklearn.utils import resample
from heapq import nsmallest

class BalanceClass(object):
    def __init__(self, df, label_col, text_col, final_sample_no, exceptions=[], mode=1):     # 1: downsample, 2: upsample, 3: combined
        self.df = df
        self.label_col = label_col
        self.text_col = text_col
        self.final_sample_no = final_sample_no
        self.exceptions = exceptions    # these labels will be ignored
        self.mode = mode

        label_count = self.df[self.label_col].value_counts(sort=True).rename_axis(self.label_col).reset_index(name='counts')
        print(label_count)
        self.labels_to_upsample = [item for item in label_count[label_count['counts'] < self.final_sample_no][self.label_col].tolist()
                              if item not in self.exceptions]
        self.labels_to_downsample = [item for item in label_count[self.label_col].tolist()
                              if item not in self.labels_to_upsample and item not in self.exceptions]
        print(self.labels_to_upsample)
        print(self.labels_to_downsample)

    def process(self):
        if self.mode == 1:
            self.downsample()
        elif self.mode == 2:
            self.upsample()
        elif self.mode == 3:
            self.upsample()
            self.downsample()
        return self.df

    def downsample(self):
        df_good = self.df[~self.df[self.label_col].isin(self.labels_to_downsample)]
        for label in self.labels_to_downsample:
            df_tmp = self.df[self.df[self.label_col] == label]
            df_downsampled = resample(df_tmp, replace=True, n_samples=self.final_sample_no, random_state=42)
            df_good = pd.concat([df_good, df_downsampled])
        self.df = df_good
        del df_good
        label_count = self.df[self.label_col].value_counts(sort=True).rename_axis(self.label_col).reset_index(
            name='counts')
        print(label_count)
    # end def

    def create_sample(self, samples, final_sample_no):  # minority class samples
        def generate_sample(n_list, k, samples_with_lengths, m):
            new_samples = []
            for i in n_list:  # for all lengths of samples to generate
                similar_len_samples = nsmallest(m, samples_with_lengths, key=lambda x: abs(x[0] - i))
                all_tokens = [token for item in similar_len_samples for token in item[1]]
                for j in range(k):  # create k samples from these tokens
                    new_samples.append(' '.join(np.random.choice(all_tokens, i, replace=False)))
            return new_samples

        sample_number = len(samples)
        samples = [sample.split(' ') for sample in samples]
        samples_with_lengths = [[len(sample), sample] for sample in samples]
        sample_lengths = [sample[0] for sample in samples_with_lengths]

        new_samples = []
        samples_to_generate = final_sample_no - sample_number
        m = 10  # number of samples to choose from while creating new samples
        k = samples_to_generate // sample_number
        if k != 0:
            n_list = sample_lengths
            new_samples.extend(generate_sample(n_list, k, samples_with_lengths, m=m * k))

        n_num = samples_to_generate % sample_number
        n_list = np.random.choice(sample_lengths, n_num, replace=False)  # randomly select n_num numbers of n
        new_samples.extend(generate_sample(n_list, 1, samples_with_lengths, m=m))
        return new_samples
    # end def

    def upsample(self):  # df must have only two columns
        for label in self.labels_to_upsample:
            samples = self.df[self.df[self.label_col] == label][self.text_col].tolist()
            new_samples = self.create_sample(samples, self.final_sample_no)
            self.df = pd.concat([self.df, pd.DataFrame({self.label_col: [label] * len(new_samples), self.text_col: new_samples})])
        label_count = self.df[self.label_col].value_counts(sort=True).rename_axis(self.label_col).reset_index(name='counts')
        print(label_count)
    # end def
# end class


if __name__ == '__main__':
    final_sample_no = 3000
    train = BalanceClass(train, 'labels', 'text', final_sample_no, [], mode=3).process()
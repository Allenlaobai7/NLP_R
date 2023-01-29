import json
import pandas as pd
import sklearn
from utils import utilwords


class prepare_train():
    def __init__(self):
        self.pos_words = utilwords.skillwords
        print('import %d numbers of positive words' % len(self.pos_words))
        self.path = './NeiEntro_wordinfo.txt'
        file = open(self.path, 'r', encoding='utf8')
        self.train_2 = []
        self.train_3 = []
        self.train_4 = []

        self.pos_count = 0
        self.pos_count_2 = 0
        self.pos_count_3 = 0
        self.pos_count_4 = 0
        self.word_count = 0
        counter = 0
        for line in file:
            if line.startswith('['):
                line = line[1:]
            elif len(line) == 1:
                break
            strline = str(line).replace(',\n', '')
            wordinfo = json.loads(strline)
            self.add_label(wordinfo)
            counter += 1
            if counter % 100000 == 0:
                print('Processing %s' % counter)
        file.close()
        print('%d numbers of positive words found in traindata out of %d words' % (self.pos_count, self.word_count - 1))
        print(self.pos_count_2, self.pos_count_3, self.pos_count_4)
    # def

    def add_label(self, wordinfo):
        key = list(wordinfo.keys())[0]
        values = [item for value in list(wordinfo.values()) for item in value]
        if values[-1] > 1 and values[-1] < 5:
            self.word_count += 1
            oneline = list([key])
            oneline.extend([value for value in values])
            if key in self.pos_words:
                self.pos_count += 1
                oneline.append(1)
                if oneline[-2] > 2:
                    if oneline[-2] > 3:
                        self.pos_count_4 += 1
                    else:
                        self.pos_count_3 += 1
                else:
                    self.pos_count_2 += 1
            else:
                oneline.append(0)

            # add to respective datasets
            if oneline[-2] > 2:
                if oneline[-2] > 3:
                    self.train_4.append(oneline)
                else:
                    self.train_3.append(oneline)
            else:
                self.train_2.append(oneline)
    # end def

    def create_matrix(self):
        col_names = ['word', 'count', 'probability', 'PPMI', 'right', 'left', 'length', 'label']
        self.train_2 = pd.DataFrame.from_records(self.train_2, columns=col_names)
        self.train_3 = pd.DataFrame.from_records(self.train_3, columns=col_names)
        self.train_4 = pd.DataFrame.from_records(self.train_4, columns=col_names)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=True, with_std=True)
        self.train_2[['PPMI', 'right', 'left']] = \
            scaler.fit_transform(self.train_2[['PPMI', 'right', 'left']])
        self.train_3[['PPMI', 'right', 'left']] = \
            scaler.fit_transform(self.train_3[['PPMI', 'right', 'left']])
        self.train_4[['PPMI', 'right', 'left']] = \
            scaler.fit_transform(self.train_4[['PPMI', 'right', 'left']])
        return self.train_2, self.train_3, self.train_4


def balanced_train(data, features):
    X = data[features]
    y = data['label']
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_sample(X, y)
    return X_resampled, y_resampled
# end def

def create_train_test(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                        shuffle=True, stratify=y)  # select balanced sample
    return X_train, X_test, y_train, y_test
# end def

class classifier:
    def __init__(self, X_train, X_test, y_train, y_test, N):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.gram = N

    def performance_metrics(self):
        print("Performance of model %s" % str(self.model))
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        pred = self.model.predict(self.X_test)
        if not str(self.model)[:3] == "SGD":
            pred_proba = self.model.predict_proba(self.X_test)
            pred_proba_c1 = pred_proba[:, 1]
            print("AUC Score : %f" % sklearn.metrics.roc_auc_score(self.y_test, pred_proba_c1))

        print("prediciton Accuracy : %f" % accuracy_score(self.y_test, pred))
        print("Confusion_matrix : ")
        print(confusion_matrix(self.y_test, pred))
        print("classification report : ")
        print(classification_report(self.y_test, pred, labels=['0', '1']))

    def decisiontree(self):
        from sklearn import tree
        self.model = tree.DecisionTreeClassifier(criterion='gini', random_state=42, class_weight='balanced')
        self.model.fit(self.X_train, self.y_train)
        self.performance_metrics()
        import pickle
        with open('./model/DT' + str(self.gram) + 'Gram' +
                  '.txt', 'wb') as f:
            pickle.dump(self.model, f)
    # end def
# end class


class classifier_cv:
    def __init__(self, train, X, y, features, N):
        self.X = X
        self.y = y
        self.train = train
        self.original_X = self.train[features]
        self.model = None
        self.gram = N

    def decisiontree(self):
        from sklearn import tree
        self.model = tree.DecisionTreeClassifier(criterion='gini', random_state=42, class_weight='balanced')

        from sklearn.model_selection import cross_validate
        scoring = ['precision_macro', 'recall_macro']
        scores = cross_validate(self.model, self.X, self.y, scoring=scoring, cv=5, return_train_score=False)
        print(scores['test_recall_macro'])

        self.model.fit(self.X, self.y)
        pred = self.model.predict(self.original_X)
        word = self.train['word']
        res = dict(zip(word, pred))
        predicted_pos = []
        for key, value in res.items():
            if 1 == value:
                predicted_pos.append(key)
        import pickle
        with open('./model/DT_final' + str(self.gram) + 'Gram' +
                  '.txt', 'wb') as f:
            pickle.dump(self.model, f)
        return predicted_pos
    # end def
# end class


def frequency_filter(output_pos, train, frequency, n):
    from utils import utilwords
    skillwords = utilwords.skillwords

    word = train['word']
    freq = train['count']
    wordfreq = dict(zip(word, freq))
    newword = list(filter(lambda x: x not in skillwords, output_pos))
    print('%d new words from %d total words' % (len(newword), len(output_pos)))

    output_dt = list(filter(lambda x: wordfreq[x] > frequency, newword))
    with open('./output/predicted_DT' + str(n) + 'Gram' +
              '.txt', 'w', encoding='utf8') as f:
        for word in output_dt:
            f.write(str(word) + '\n')
    print('%d words from DT with frequency > %d' % (len(output_dt), frequency))
    return output_dt
# end def

def main():
    train_2, train_3, train_4 = prepare_train().create_matrix()
    features = ['PPMI', 'right', 'left']

    X_2, y_2 = balanced_train(train_2, features)
    X_3, y_3 = balanced_train(train_3, features)
    X_4, y_4 = balanced_train(train_4, features)

    # final
    print('#' * 20 + '\t' + '2_gram' + '\t' + '#' * 20)
    output_dt = classifier_cv(train_2, X_2, y_2, features, 2).decisiontree()
    print('%d 2-gram words from DT' % len(output_dt))
    frequency_filter(output_dt, train_2, frequency=5, n=2)

    # # final
    print('#' * 20 + '\t' + '3_gram' + '\t' + '#' * 20)
    output_dt = classifier_cv(train_3, X_3, y_3, features, 3).decisiontree()
    print('%d 3-gram words from DT' % len(output_dt))
    frequency_filter(output_dt, train_3, frequency=3, n=3)

    # # final
    print('#' * 20 + '\t' + '4_gram' + '\t' + '#' * 20)
    output_dt = classifier_cv(train_4, X_4, y_4, features, 4).decisiontree()
    print('%d 4-gram words from DT' % len(output_dt))
    frequency_filter(output_dt, train_4, frequency=2, n=4)

if __name__ == '__main__':
    main()

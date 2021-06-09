import fasttext


model = fasttext.train_supervised('data/train.txt')

print(model.words)
print(model.labels)

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test('data/test.txt'))

a =model.predict("Requirement already satisfied  in Library Frameworks Python framework Versions lib python3.6 site-packages from fasttext ", k =5)
print(a)
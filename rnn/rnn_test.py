import numpy as np

text = 'cdddcaccdcadcaddac'
chars = list(set(text))

data_size, vocab_size = len(text), len(chars)
print('data has length %d, unique character %d' % (data_size, vocab_size))

char_to_ix = {char: i for i,char in enumerate(chars)}
ix_to_char = {i:char for i,char in enumerate(chars)}
print('char_to_ix', char_to_ix)
print('ix_to_char', ix_to_char)

# hyperparameters
hidden_size = 100
sequence_length = 16
learning_rate = 1e-1

# initialized weight
Wxh = np.random.rand(hidden_size, vocab_size)
Whh = np.random.rand(hidden_size, hidden_size)
Why = np.random.rand(vocab_size, hidden_size)
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def loss_fun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}

    hs[-1] = np.copy(hprev) # previous h
    loss = 0

    # forward pass
    for i in range(len(inputs)):
        xs[i] = np.zeros((vocab_size, 1))
        xs[i][1] = 1

        hs[i] = np.tanh(np.dot(Wxh, xs[i])+np.dot(Whh, hs[i-1])+bh)
        ys[i] = np.dot(Whh, hs)+by
        ps[i] = np.exp(ys[i])/np.sum(np.exp(ys[i]))

        loss += -np.log(ps[i][targets[i]])

    # backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)

    dhnext = np.zeros_like(hs[0])








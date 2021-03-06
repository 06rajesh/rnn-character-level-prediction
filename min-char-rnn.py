"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""

import numpy as np
from pathlib import Path
import pickle


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[
            t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backpropagation into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


def get_weights():
    _count = [0, 0]
    _loss = [-np.log(1.0 / vocab_size) * seq_length,  -np.log(1.0 / vocab_size) * seq_length]
    _weights = [Wxh, Whh, Why, bh, by]
    _hprev = np.zeros((hidden_size, 1))
    _mem_weights = [mWxh, mWhh, mWhy, mbh, mby]

    weight_file = Path("weights/0.pkl")
    if weight_file.is_file():
        # Getting back the Weights:
        with open('weights/0.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            _count, _loss, _hprev, _weights, _mem_weights = pickle.load(f)
            # print("Resuming From %d iter with loss %f" % (_count[0], _loss[0]))

    return _count, _loss, _hprev,  _weights, _mem_weights


# def save_model():


def train(count, all_loss, hprev):
    n, p = count
    smooth_loss, loss = all_loss

    for i in range(5):
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))  # reset RNN memory
            p = 0  # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
        print(hprev)
        # sample from the model now and then
        if n % 500 == 0:
            sample_ix = sample(hprev, inputs[0], 500)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('iter %d, loss: %f' % (n, smooth_loss), end='\r')
            with open("output/" + str(n // 50000) + ".txt", 'a+') as f:
                identifier = 'iter %d, loss: %f' % (n, smooth_loss)
                f.write(identifier)
                f.write('\n==================================\n')
                f.write(txt)
                f.write('\n\n')

        # forward seq_length characters through the net and fetch gradient
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 500 == 0:
            # print("Saving Weights on Iter: %f" % n)
            # Saving the Weights:
            with open('weights/0.pkl', 'wb') as f:
                _count = [n, p]
                _all_loss = [smooth_loss, loss]
                _weights = [Wxh, Whh, Why, bh, by]
                _mem_weights = [mWxh, mWhh, mWhy, mbh, mby]
                pickle.dump([_count, _all_loss, hprev, _weights, _mem_weights], f)

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
        p += seq_length  # move data pointer
        n += 1  # iteration counter


def generate(hprev, total, _count, _all_loss):
    iprev = np.array([
        [0.72961938],
        [-0.35330982],
        [-0.50739483],
        [-0.90847369],
        [0.69079595],
        [0.05311218],
        [-0.434315],
        [-0.91756149],
        [-0.44593906],
        [0.77491903]
    ])

    n, p = _count
    smooth_loss, loss = _all_loss
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    sample_ix = sample(iprev, inputs[0], total)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    with open("output/" + str(n // 50000) + ".txt", 'a+') as f:
        identifier = 'iter %d, loss: %f' % (n, smooth_loss)
        f.write(identifier)
        f.write('\n==================================\n')
        f.write(txt)
        f.write('\n\n')


if __name__ == '__main__':
    # data i/o
    data = open('input.txt', 'r').read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('Data has {} characters, {} unique.'.format(data_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # hyper parameters
    hidden_size = 10  # size of hidden layer of neurons
    seq_length = 100  # number of steps to unroll the RNN for
    learning_rate = 1e-1

    # model parameters
    Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
    bh = np.zeros((hidden_size, 1))  # hidden_bias
    by = np.zeros((vocab_size, 1))  # output_bias
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  # memory variables for Adagrad
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad

# count, all_loss, h_prev, weights, mem_weights = get_weights()
# Wxh, Whh, Why, bh, by = weights
# mWxh, mWhh, mWhy, mbh, mby = mem_weights
# train(count, all_loss, h_prev)
# print(Wxh, Whh)
# generate(h_prev, 500, count, all_loss)





"""
Minimal character-level Vanilla RNN model. Written by Rajesh Baidya
BSD License
"""

import numpy as np
from pathlib import Path


def get_data():
    # data i/o
    dt = open('zafor-sir.txt', 'r').read()

    weight_file = Path("weights/data.npz")
    if weight_file.is_file():
        loaded = np.load("weights/data.npz")
        print('Previous LoadedData has {} characters, {} unique.'.format(len(dt), len(loaded['chars'])))
        char_ix = {ch: i for i, ch in enumerate(loaded['chars'])}
        ix_char = {i: ch for i, ch in enumerate(loaded['chars'])}
        return dt, loaded['chars'], char_ix, ix_char

    characters = list(set(dt))
    d_size, v_size = len(dt), len(characters)
    print('Data has {} characters, {} unique.'.format(d_size, v_size))
    char_ix = {ch: i for i, ch in enumerate(characters)}
    ix_char = {i: ch for i, ch in enumerate(characters)}
    np.savez("weights/data.npz", chars=characters)
    return dt, characters, char_ix, ix_char


def get_weight_bias(h_size, v_size):
    weight_file = Path("weights/weights.npz")
    if weight_file.is_file():
        loaded = np.load("weights/weights.npz")
        print("Loading Previous Weights")
        return loaded['Wxh'], loaded['Whh'], loaded['Why'], loaded['bh'], loaded['by']
    # model parameters
    wxh = np.random.randn(h_size, v_size) * 0.01  # input to hidden
    whh = np.random.randn(h_size, h_size) * 0.01  # hidden to hidden
    why = np.random.randn(v_size, h_size) * 0.01  # hidden to output
    hb = np.zeros((h_size, 1))  # hidden_bias
    ho = np.zeros((v_size, 1))  # output_bias

    return wxh, whh, why, hb, ho


def load_checkpoints():
    weight_file = Path("weights/checkpoints.npz")
    if weight_file.is_file():
        loaded = np.load("weights/checkpoints.npz")
        print("Resuming From {}".format(loaded['n']))
        return loaded['n'], loaded['p'], loaded['hprev'], loaded['loss'], loaded['smooth_loss']
    _n, _p = [0, 0]
    _hprev = np.zeros((hidden_size, 1))
    _loss = -np.log(1.0 / vocab_size) * seq_length
    return _n, _p, _hprev, _loss, _loss


def get_mem_vars():
    weight_file = Path("weights/weights.npz")
    if weight_file.is_file():
        loaded = np.load("weights/weights.npz")
        print("Loading Previous Memory Variables")
        return loaded['mWxh'], loaded['mWhh'], loaded['mWhy'], loaded['mbh'], loaded['mby']
    mxh, mhh, mhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  # memory variables for Adagrad
    mhb, mho = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
    return mxh, mhh, mhy, mhb, mho


def save_model(n, p, hprev, loss, smooth_loss):
    # print("Saving Weights on Iter: %f" % n)
    np.savez("weights/checkpoints.npz", n=n, p=p, hprev=hprev, loss=loss, smooth_loss=smooth_loss)
    np.savez("weights/weights.npz",
             Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by,
             mWxh=mWxh, mWhh=mWhh, mWhy=mWhy, mbh=mbh, mby=mby
             )


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


def loss_func(inputs, targets, hprev):
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
    dwxh, dwhh, dwhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[
            t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dwhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backpropagation into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh non linearity
        dbh += dhraw
        dwxh += np.dot(dhraw, xs[t].T)
        dwhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dwxh, dwhh, dwhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dwxh, dwhh, dwhy, dbh, dby, hs[len(inputs) - 1]


def train(n, p, smooth_loss, hprev):
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))  # reset RNN memory
            p = 0  # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
        # sample from the model now and then
        if n % 5000 == 0:
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
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_func(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 5000 == 0:
            # Saving the Weights:
            save_model(n, p, hprev, loss, smooth_loss)

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
        p += seq_length  # move data pointer
        n += 1  # iteration counter


def generate(hprev, total, n, p, smooth_loss):
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    sample_ix = sample(hprev, inputs[0], total)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    with open("output/" + str(n // 50000) + ".txt", 'a+') as f:
        identifier = 'iter %d, loss: %f' % (n, smooth_loss)
        f.write(identifier)
        f.write('\n==================================\n')
        f.write(txt)
        f.write('\n\n')

if __name__ == '__main__':
    data, chars, char_to_ix, ix_to_char = get_data()
    data_size, vocab_size = len(data), len(chars)

    # hyper parameters
    hidden_size = 10  # size of hidden layer of neurons
    seq_length = 200  # number of steps to unroll the RNN for
    learning_rate = 1e-1

    Wxh, Whh, Why, bh, by = get_weight_bias(hidden_size, vocab_size)
    mWxh, mWhh, mWhy, mbh, mby = get_mem_vars()
    cnt, completed, hPrev, l, smooth_l = load_checkpoints()
    # train(cnt, completed, smooth_l, hPrev)
    generate(hPrev, 500, cnt, completed, smooth_l)


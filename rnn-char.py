"""
Minimal character-level Vanilla RNN model. Written by Rajesh Baidya
BSD License
"""

import numpy as np
from pathlib import Path


def get_data():
    # data i/o
    data = open('input.txt', 'r').read()

    weight_file = Path("weights/data.npz")
    if weight_file.is_file():
        loaded = np.load("weights/data.npz")
        print('Previous LoadedData has {} characters, {} unique.'.format(len(data), len(loaded['chars'])))
        return data, loaded['chars'], loaded['char_to_ix'], loaded['ix_to_char']

    characters = list(set(data))
    d_size, v_size = len(data), len(characters)
    print('Data has {} characters, {} unique.'.format(d_size, v_size))
    char_ix = {ch: i for i, ch in enumerate(characters)}
    ix_char = {i: ch for i, ch in enumerate(characters)}
    np.savez("weights/data.npz", chars=characters, char_to_ix=char_ix, ix_to_char=ix_char)
    return data, characters, char_ix, ix_char


def get_weight_bias(h_size, v_size):
    # model parameters
    wxh = np.random.randn(h_size, v_size) * 0.01  # input to hidden
    whh = np.random.randn(h_size, h_size) * 0.01  # hidden to hidden
    why = np.random.randn(v_size, h_size) * 0.01  # hidden to output
    hb = np.zeros((h_size, 1))  # hidden_bias
    ho = np.zeros((v_size, 1))  # output_bias

    return wxh, whh, why, hb, ho


if __name__ == '__main__':
    data, chars, char_to_ix, ix_to_char = get_data()
    data_size, vocab_size = len(data), len(chars)

    # hyper parameters
    hidden_size = 10  # size of hidden layer of neurons
    seq_length = 100  # number of steps to unroll the RNN for
    learning_rate = 1e-1

    Wxh, Whh, Why, bh, by = get_weight_bias(hidden_size, vocab_size)

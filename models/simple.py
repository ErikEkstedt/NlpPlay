import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleEncoder(nn.Module):
    def __init__(self,
                 vocab,
                 word_vector_dim=512,
                 rnn_hidden_state=512,
                 rnn_layers=2,
                 rnn_dropout=0.1,
                 bidirectional=False,
                 device='cpu'):
        super(SimpleEncoder, self).__init__()

        self.word_vector_dim = word_vector_dim
        self.rnn_hidden_state = rnn_hidden_state
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=vocab.num_words,
                                      embedding_dim=word_vector_dim,
                                      padding_idx=0)

        self.gru = nn.GRU(input_size=word_vector_dim,
                          hidden_size=rnn_hidden_state,
                          num_layers=rnn_layers,
                          dropout=self.rnn_dropout,
                          bidirectional=bidirectional)

    def forward(self, x):
        padded_tensor = x['padded'].to(self.device)
        list_of_lengths = x['lengths']

        padded_word_vectors = self.embedding(padded_tensor)
        packed_padded = pack_padded_sequence(padded_word_vectors,
                                             list_of_lengths,
                                             batch_first=True)

        # packed_padded in -> packed_padded out
        packed_padded_out, hidden = self.gru(packed_padded)
        padded_out, lengths = pad_packed_sequence(packed_padded_out,
                                                  batch_first=True)

        return {'padded': padded_out, 'lengths': lengths}, hidden


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()

    def forward(self, x):
        return x


class SimpleChatBot(nn.Module):
    def __init__(self, vocab):
        super(SimpleChatBot, self).__init__()

        self.vocab = vocab
        self.word_dim = word_dim
        self.embedding = nn.Embedding(vocab.num_words, )
        self.encoder = SimpleEncoder
        self.decoder = SimpleDecoder

    def forward(self, x):
        return x


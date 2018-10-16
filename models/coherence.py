import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CoherenceRNNPackedPadded(nn.Module):
    '''
    Arguments:
        vocab_size,             int, words in vocab+pad_tokens
        word_vector_dim int,    size of word embeddings 
        rnn_hidden_state
        rnn_layers
        bidirectional
        rnn_dropout
    '''
    def __init__(self, vocab_size, word_vector_dim,
                 rnn_hidden_state, rnn_layers=2,
                 rnn_dropout=0.1, bidirectional=False, device='cpu'):
        super(CoherenceRNNPackedPadded, self).__init__()
        self.vocab_size = vocab_size
        self.word_vector_dim = word_vector_dim
        self.rnn_hidden_state = rnn_hidden_state
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=word_vector_dim,
                                      padding_idx=0)

        self.gru = nn.GRU(input_size=word_vector_dim,
                          hidden_size=rnn_hidden_state,
                          num_layers=rnn_layers,
                          dropout=self.rnn_dropout,
                          bidirectional=bidirectional)

        self.head = nn.Linear(2*rnn_hidden_state, 1)   # 2 hidden -> coherence

    def forward(self, src, hidden=None, verbose=False):
        '''
        Arguments:
        src (Dict):

            src_padded:    torch.Tensor of padded sequences
            src_lengths:            list of ints. lengths of sequence
            src_eos_idx:            list of ints. placement of EOS

        Return:
            Coherence:      torch.FloatTensor(), shape (N, 1)

        '''
        padded_tensor = src['src_padded'].to(self.device)
        list_of_lengths = src['src_lengths']
        list_of_eos_idx = src['src_eos_idx']

        padded_word_vectors = self.embedding(padded_tensor)
        packed_padded = pack_padded_sequence(padded_word_vectors,
                                             list_of_lengths,
                                             batch_first=True)

        # packed_padded in -> packed_padded out
        packed_padded_out, hidden = self.gru(packed_padded)
        padded_out, lengths = pad_packed_sequence(packed_padded_out,
                                                  batch_first=True)

        # Get output from EOS token in between response and context as well as
        # last (after response)
        whole_batch = torch.arange(padded_out.shape[0])
        outputs_at_eos = padded_out[whole_batch, list_of_eos_idx]
        outputs_last = padded_out[whole_batch, lengths-1]  # lengths differ from idx by one

        # Coherence
        coherence = torch.cat((outputs_at_eos, outputs_last), dim=1)
        coherence_out = self.head(coherence).squeeze()  # (3,1) -> (3,)

        return torch.sigmoid(coherence_out)


def save_checkpoint(save_dir,
                    model,
                    optimizer,
                    loss,
                    rnn_layers,
                    vocab,
                    epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'en_opt': optimizer.state_dict(),
        'loss': loss,
        'voc_dict': vocab.__dict__,
        'embedding': model.embedding.state_dict()
    }, os.path.join(save_dir, 'layers_{}_ep_{}_{}.tar'.format(rnn_layers, epoch, 'checkpoint')))


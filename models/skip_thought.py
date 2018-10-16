import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUEncoder(nn.Module):
    '''
    sources:
        GRU paper: https://arxiv.org/pdf/1502.02367.pdf
    '''
    def __init__(self,
                 vocab_size,
                 word_vector_dim,
                 rnn_hidden_state,
                 rnn_layers=2,
                 rnn_dropout=0.1,
                 bidirectional=False,
                 device='cpu'):
        super(GRUEncoder, self).__init__()

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

    def forward(self, src, hidden=None, verbose=False):
        '''
        Arguments:
        src (Dict):
            input_padded:    torch.Tensor of padded sequences
            input_lengths:   list of Ints. lengths of sequence
        '''

        # Get word idx
        padded_tensor = src['padded'].to(self.device)
        list_of_lengths = src['lengths']

        # Embedd the tensors and create pack_padded_sequence
        padded_word_vectors = self.embedding(padded_tensor)
        packed_padded = pack_padded_sequence(padded_word_vectors,
                                             list_of_lengths,
                                             batch_first=True)

        # packed_padded in -> packed_padded out
        packed_padded_out, hidden = self.gru(packed_padded)
        padded_out, lengths = pad_packed_sequence(packed_padded_out,
                                                  batch_first=True)
        return padded_out, lengths


# Many bad decoders

class SkipThought(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(SkipThought, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        def recurrence(input, hidden):
            """Recurrence helper."""

            gates = self.input_weights(input) + self.hidden_weights(hx)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)  # o_t
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class SimpleThoughtVectorDecoder(nn.Module):
    '''
    This simpler version of the decoders in the Skip-Thought vector paper uses
    the same technique as standard NMT models. In this approach the hidden state
    of the decoders are set to the output from the encoder. In the paper the
    decoders have their own hidden vectors and uses the encoder state to
    calculate the bias in the gru through a linear function.
    '''
    def __init__(self, input_size, hidden_size, vocab_size,
                 num_layers=1, embedding=None):
        super(MyClass, self).__init__()

        # self.embedding = embedding
        self.gru = nn.GRU(input_size, hidden_size, num_layers batch_first=True)


    def forward(self, x):
        return x


class ThoughtVecDecoder(nn.Module):
    '''
    The decoders are custom for the thought-vector paper. Instead of using the
    regular bias term in the standard GRU they use a matrix multiplication with
    some learnable matrix and the last output from the encoder, the sentence
    vector.

    (*) - elementwise operation

    r_t(x, hidden) = torch.sigmoid(W_r * x + U_r * hidden + C_r * enc_state)
    z_t(x, hidden) = torch.sigmoid(W_z * x + U_z * hidden + C_z * enc_state)
    h(r_t, hidden) = torch.tanh(W * x + U * (r_t (*) hidden) + C * enc_state)

    hidden(hidden, h, z_t) = (1 - z_t) (*) hidden + z_t (*) h
    '''
    def __init__(self, input_size, hidden_size, enc_state_size, num_layers=1):
        super(ThoughtVecDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enc_state_size = enc_state_size
        self.num_layers = num_layers

        # Weights
        self.W = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.C = nn.Linear(enc_state_size, 3 * hidden_size, bias=False)
        self.U_rz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.num_layers, self.hidden_size)
        return hidden


    def forward(self, inputs, enc_state):
        def recurrence(x, hidden, enc_state, targets, verbose=False):
            w = self.W(x)
            c = self.C(enc_state)
            u = self.U_rz(hidden)

            wr, wz, wh = torch.chunk(w, 3)
            cr, cz, ch = torch.chunk(c, 3)
            ur, uz = torch.chunk(u, 2)

            r = torch.sigmoid(wr + ur + cr)
            z = torch.sigmoid(wz, uz, cz)

            uh = self.U_h(r*hidden)
            h = torch.tanh(wu + uh + ch)

            if verbose:
                print('input: ', x.shape)
                print('r: ', r.shape)
                print('z: ', z.shape)
                print('h: ', h.shape)
            return (1 - z)*hidden + z * h

        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        # word embeddings of input
        inputs = self.embedding(inputs)  # word vectors

        output = []
        steps = range(inputs.size(1))  # (B, T, D), B-batch, T-sequence len, D, word dim
        for i in steps:
            hidden = recurrence(inputs[i], hidden, enc_state)
            output.append(hidden)



        return output, hidden


class GreedySearchDecoder(nn.Module):
    '''Pytorch chatbot tutorial'''
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores # Return collections of word tokens and scores



if __name__ == "__main__":
    # input_size = 5
    # hidden_size = 5
    # seq_len = 3
    # batch_size = 8
    # rnn_layers = 1
    # loss_fn = nn.MSELoss()

    # inputs = torch.ones(batch_size, seq_len, input_size)
    # hidden = torch.ones(rnn_layers, batch_size, hidden_size)
    # target = torch.ones(inputs.shape)

    # out, hidden = model(inputs, hidden)


    # Real data
    from dataset import get_data
    dloader = get_data('data/data.pt', batch_size=6)

    vocab_size = dloader.dataset.vocab.num_words
    print('vocabulary size: ', vocab_size)  # 56710

    encoder = GRUEncoder(vocab_size,
                         word_vector_dim=256,
                         rnn_hidden_state=256,
                         rnn_layers=2,
                         rnn_dropout=0.1,
                         bidirectional=False,
                         device='cpu')



    verbose = False
    for current, prepost in dloader:
        if verbose:
            print(current['padded'].shape)
            print(prepost['pre_padded'].shape)
            print(prepost['post_padded'].shape)
        pad, length = encoder(current)
        print(pad.shape)
        break


    # todo
    # dec = ThoughtVecDecoder







    # TorchText



    # Approach 1:
    # set up fields
    TEXT = data.Field(lower=True, batch_first=True)

    # make splits for data
    train, valid, test = datasets.WikiText2.splits(TEXT)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0])['text'][0:10])


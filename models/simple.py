import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleEncoder(nn.Module):
    def __init__(self,
                 vocab_num_words,
                 word_vector_dim=512,
                 rnn_hidden_state=512,
                 rnn_layers=2,
                 rnn_dropout=0.1,
                 bidirectional=False,
                 device='cpu'):
        super(SimpleEncoder, self).__init__()
        self.vocab_num_words = vocab_num_words
        self.word_vector_dim = word_vector_dim
        self.rnn_hidden_state = rnn_hidden_state
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=vocab_num_words,
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
    def __init__(self,
                 vocab,
                 embedding,
                 rnn_hidden_state=512,
                 rnn_layers=2,
                 rnn_dropout=0.1,
                 bidirectional=False,
                 device='cpu'):
        super(SimpleDecoder, self).__init__()
        self.vocab = vocab
        self.SOS_token = vocab.word2index['SOS']
        self.word_vector_dim = word_vector_dim
        self.rnn_hidden_state = rnn_hidden_state
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout if rnn_layers > 1 else 0
        self.device = device

        self.embedding = embedding
        self.gru = nn.GRU(input_size=embedding.embedding_dim,
                          hidden_size=rnn_hidden_state,
                          num_layers=rnn_layers,
                          dropout=self.rnn_dropout,
                          bidirectional=bidirectional)

    def forward(self, x, enc_out, enc_hidden):
        print(x.keys())
        padded_tensor = x['padded'].to(self.device)
        list_of_lengths = x['lengths']
        batch_size = padded_tensor.shape[0]

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        print(decoder_input.shape)
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        print(decoder_hidden.shape)

        # Loop over tokens
        loss, n_totals, print_losses = 0, 0, []
        for t in range(max_target_len):
            # Forward batch of sequences through decoder one time step at a time
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden,
                                                     encoder_outputs)
            print(decoder_hidden.shape)
            print(decoder_input.shape)
            # Teacher forcing works best for each token according to
            # Scheduled Sampling for Sequence Prediction 
            # with Recurrent Neural Networks, (S. Bengio et al 2015)
            if random.random() < teacher_forcing_ratio:
                decoder_input = target_variable[t].view(1, -1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
        return loss, sum(print_losses) / n_totals


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


if __name__ == "__main__":

    from datasets import get_pairs_dataloader # assume ipy from repo-root

    dloader = get_pairs_dataloader(batch_size=8)

    # Get sample data: x, label
    for x, label in dloader:
        print(x.keys())  # padded, lengths
        print(label.keys())  # padded, lengths
        break

    vocab = dloader.dataset.vocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'encoder_decoder_model'
    attn_model = 'dot' # 'general', 'concat'
    hidden_size = 512
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    vocab_size = vocab.num_words
    word_vector_dim = 128

    # SimpleEncoder
    enc = SimpleEncoder(vocab_size,
                        word_vector_dim,
                        rnn_hidden_state=hidden_size,
                        rnn_layers=encoder_n_layers,
                        rnn_dropout=dropout,
                        bidirectional=False,
                        device='cpu')

    enc_out, enc_hidden = enc(x)
    enc_padded, enc_lengths = enc_out['padded'], enc_out['lengths']

    print('max length: ', max(enc_length).item())
    print(enc_padded.shape)  # batch_size, max_length, 512
    print(enc_hidden.shape)  # 2, batch_size, 512

    dec = SimpleDecoder(vocab,
                        enc.embedding,
                        rnn_hidden_state=512,
                        rnn_layers=2,
                        rnn_dropout=0.1,
                        bidirectional=False,
                        device='cpu')

    dec_out, dec_hidden = dec(x=label, enc_out=enc_out, enc_hidden=enc_hidden)

    for key in vocab.word2index.keys():
        if key.startswith('SOS'):
            print(key)



    # Attention Play
    vector_size = 10

    attention = Attn(method='dot', hidden_size=vector_size)

    enc_out = torch.ones(10, vector_size)
    enc_out.shape
    dec_out = torch.randn(1, 10)
    dec_out.shape

    # batchify
    enc_out = enc_out.unsqueeze(0)
    dec_out = dec_out.unsqueeze(0)

    attention(dec_out, enc_out)


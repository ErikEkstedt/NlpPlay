import itertools
from vocabulary import Voc

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

##################################################################
##################################################################
##################################################################
# PyTorch Chatbot - Pairs

class DialogPairDataset(Dataset):
    def __init__(self, pairs, vocab, pad_idx=0):
        self.pairs = pairs
        self.vocab = vocab
        self.pad_token = torch.LongTensor([pad_idx])

    def __len__(self):
        return len(self.pairs)

    def decode(self, data):
        if isinstance(data, torch.Tensor):
            data = [x.item() for x in data]
        sentence = []
        for word_idx in data:
            sentence.append(self.vocab.index2word[word_idx])
        return sentence

    def indexes_from_sentence(self, sentence):
        return [self.vocab.word2index[word] for word in sentence.split(' ')]

    def __getitem__(self, idx):
        '''lazy reimplementation of trigrams data
        uses only the first two entries. (missing some data)
        '''
        pair = self.pairs[idx]

        # Torchify words -> idx -> tensors
        context = self.indexes_from_sentence(pair['pre'])
        response = self.indexes_from_sentence(pair['current'])
        context = torch.LongTensor(context)
        response = torch.LongTensor(response)
        return context, response


def pairs_collate_fn(data):
    '''
    Arguments:
        data:  list of tensors (context, response)
    '''

    def zeroPadding(l, fillvalue=0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    # The data should be sorted by length.
    # Sort by length of contexts
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences.
    context_batch, response_batch = zip(*data)  # returns tuples

    # Context
    context_lengths = [len(d) for d in context_batch]
    context_padded = pad_sequence(list(context_batch), batch_first=True)

    # Response
    response_lengths = [len(d) for d in response_batch]
    response_np = [x.numpy() for x in response_batch]
    response_padded_list = zeroPadding(response_np)
    response_padded = torch.LongTensor(response_padded_list)

    # I like to return dict such that the names and meaning of the data is shown
    # in the training loop.
    context = {'padded': context_padded,
               'lengths': context_lengths}
    response = {'padded': response_padded,
                'lengths': response_lengths}
    return context, response


def get_pairs_dataloader(loadfile='data/dialog/PERSON_mincount_4.pt', **kwargs):
    data = torch.load(loadfile)
    trigrams = data['trigrams']
    vocab = Voc('skip_thought')
    vocab.__dict__ = data['vocab_dict']
    dset = DialogPairDataset(trigrams, vocab)
    return DataLoader(dset, collate_fn=pairs_collate_fn, **kwargs)



##################################################################
##################################################################
##################################################################
# Skip-Thougth vectors - Trigrams
# Dialog dataset sentence trigrams s(s_i-1, s_i, s_i+1).

class DialogTrigramDataset(Dataset):
    def __init__(self, trigrams, vocab):
        self.trigrams = trigrams
        self.vocab = vocab

    def __len__(self):
        return len(self.trigrams)

    def decode(self, data):
        if isinstance(data, torch.Tensor):
            data = [x.item() for x in data]
        sentence = []
        for word_idx in data:
            sentence.append(self.vocab.index2word[word_idx])
        return sentence

    def indexesFromSentence(self, sentence):
        return [self.vocab.word2index[word] for word in sentence.split(' ')]

    def __getitem__(self, idx):
        tri = self.trigrams[idx]

        # Torchify words -> idx -> tensors
        current = self.indexesFromSentence(tri['current'])
        pre = self.indexesFromSentence(tri['pre'])
        post = self.indexesFromSentence(tri['post'])

        current = torch.LongTensor(current)
        pre = torch.LongTensor(pre)
        post = torch.LongTensor(post)

        return current, pre, post


def trigrams_collate_fn(data):
    '''
    Arguments:
        data:  list of tensors (pre, current, post)
    '''

    def zeroPadding(l, fillvalue=0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    # The data should be sorted by length.
    # Sort by length of contexts
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences.
    current, pre, post = zip(*data)  # returns tuples

    # Current, sorted after this data
    current_lengths = [len(d) for d in current]
    current_padded = pad_sequence(list(current), batch_first=True)

    # Pad pre
    pre_lengths = [len(d) for d in pre]
    pre_np = [x.numpy() for x in pre]
    pre_padded_list = zeroPadding(pre)
    pre_padded = torch.LongTensor(pre_padded_list).transpose(1, 0)

    # Pad post
    post_lengths = [len(d) for d in post]
    post_np = [x.numpy() for x in post]
    post_padded_list = zeroPadding(post)
    post_padded = torch.LongTensor(post_padded_list).transpose(1, 0)

    # I like to return dict such that the names and meaning of the data is
    # shown in the training loop.
    current = {'padded': current_padded,
               'lengths': current_lengths}
    prepost = {'pre_padded': pre_padded, 'pre_lengths': pre_lengths,
               'post_padded': post_padded, 'post_lengths': post_lengths}
    return current, prepost


def get_trigrams_trainval_loader(loadfile='data/dialog/PERSON_mincount_4.pt',
                                 **kwargs):
    ''' Split data into train/test sets and return dataloaders'''
    from sklearn.model_selection import train_test_split

    data = torch.load(loadfile)
    trigrams = data['trigrams']
    vocab = Voc('skip_thought')
    vocab.__dict__ = data['vocab_dict']

    train, test = train_test_split(trigrams, test_size=0.1)

    train_dset = DialogTrigramDataset(train, vocab)
    test_dset = DialogTrigramDataset(test, vocab)

    train_loader = DataLoader(train_dset,
                              collate_fn=trigrams_collate_fn, **kwargs)
    test_loader = DataLoader(test_dset,
                             collate_fn=trigrams_collate_fn, **kwargs)
    return train_loader, test_loader


def get_trigrams_dataloader(loadfile='data/dialog/PERSON_mincount_4.pt', **kwargs):
    data = torch.load(loadfile)
    trigrams = data['trigrams']
    vocab = Voc('skip_thought')
    vocab.__dict__ = data['vocab_dict']
    dset = DialogsTrigramDataset(trigrams, vocab)
    return DataLoader(dset, collate_fn=trigrams_collate_fn, **kwargs)


##################################################################
##################################################################
##################################################################
# Coherence. Pairs + Randomly generated data (choose sentence, random sentences)

class CoherenceDataset(Dataset):
    ''' Cornell Movie database '''
    def __init__(self, pairs=None, vocab_dict=None, use_eos_sos_tokens=False,
                 randomize_responses=True, choose_random_responses=False,
                 ratio_pos_neg_samples=0.5):

        self.vocab = Voc('')
        if pairs is None:
            pairs, vocab_dict = self.load_default_data()
            self.pairs = pairs
            self.vocab.__dict__ = vocab_dict
        else:
            self.pairs = pairs
            self.vocab.__dict__ = vocab_dict

        self.use_eos_sos_tokens = use_eos_sos_tokens
        self.pad_token = torch.LongTensor([self.vocab.PAD_token])
        self.sos_token = torch.LongTensor([self.vocab.SOS_token])
        self.eos_token = torch.LongTensor([self.vocab.EOS_token])

    def __len__(self):
        return len(self.pairs)

    def get_random(self):
        return self[random.randint(0, len(self)-1)]

    def decode(self, data):
        if isinstance(data, torch.Tensor):
            data = [x.item() for x in data]
        sentence = []
        for word_idx in data:
            sentence.append(self.vocab.index2word[word_idx])
        return sentence

    def encode(self, data):
        indices = []
        for word in data:
            indices.append(self.vocab.word2index[word])
        return indices

    def load_default_data(self):
        # Load vocabulary and pairs
        filename = join(CORNELL_ROOT, 'data_maxlen_None_trim_4_val_10%.pt')
        data = torch.load(filename)
        return data['train_pairs'], data['vocab_dict']

    def indexesFromSentence(self, sentence):
        return [self.vocab.word2index[word] for word in sentence.split(' ')]

    def __getitem__(self, idx):
        '''Reads in a pairs in self.pairs and creates torch.tensors.'''
        pair = self.pairs[idx]

        # Context is always the same
        context = self.indexesFromSentence(pair[0])
        context = torch.LongTensor(context)
        response = self.indexesFromSentence(pair[1])
        response = torch.LongTensor(response)
        coherence = torch.FloatTensor([pair[2]])
        return  context, response, coherence


def coherence_collate_fn(data):
    '''
    Arguments:
        data:  list of (tensor, eos_idx, trg)
    '''
    # The data should be sorted by length.
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences.
    src_batch, src_eos_idx, trg_batch = zip(*data)  # returns tuples
    trg_batch = torch.tensor(list(trg_batch))

    # 1. Get the lengths of the input sequences
    # 3. Pad the list of tensors (make lists of tuples)
    src_lengths = [len(d) for d in src_batch]
    src_padded = pad_sequence(list(src_batch), batch_first=True)

    # I like to return dict such that the names and meaning of the data is shown
    # in the training loop.
    return {'src_padded': src_padded,
            'src_lengths': src_lengths,
            'src_eos_idx': src_eos_idx}, trg_batch


def get_train_val_loader(data_path=None,
                         use_eos_sos_tokens=False,
                         randomize_responses=True,
                         choose_random_responses=False,
                         ratio_pos_neg_samples=0.5,
                         **kwargs):

    if data_path is None:
        data_path = join(CORNELL_ROOT, 'data_maxlen_None_trim_4_val_10%.pt')


    data = torch.load(data_path)
    train_dset = CoherenceDataset(pairs=data['train_pairs'],
                                  vocab_dict=data['vocab_dict'],
                                  use_eos_sos_tokens=use_eos_sos_tokens,
                                  randomize_responses=randomize_responses,
                                  choose_random_responses=choose_random_responses,
                                  ratio_pos_neg_samples=ratio_pos_neg_samples)

    val_dset = CoherenceDataset(pairs=data['val_pairs'],
                                vocab_dict=data['vocab_dict'],
                                use_eos_sos_tokens=use_eos_sos_tokens,
                                randomize_responses=randomize_responses,
                                choose_random_responses=choose_random_responses,
                                ratio_pos_neg_samples=ratio_pos_neg_samples)

    train_loader = DataLoader(train_dset, collate_fn=coherence_collate_fn, **kwargs)
    val_loader = DataLoader(val_dset, collate_fn=coherence_collate_fn, **kwargs)
    return train_loader, val_loader


def get_coherence_dataloader(use_eos_sos_tokens=False,
                           randomize_responses=True,
                           choose_random_responses=False,
                           ratio_pos_neg_samples=0.5,
                           **kwargs):

    dset = CoherenceDataset(use_eos_sos_tokens=use_eos_sos_tokens,
                        randomize_responses=randomize_responses,
                        choose_random_responses=choose_random_responses,
                        ratio_pos_neg_samples=ratio_pos_neg_samples)

    return DataLoader(dset, collate_fn=coherence_collate_fn, **kwargs)


##################################################################
##################################################################
##################################################################


if __name__ == "__main__":

    dloader = get_dataloader(batch_size=32, shuffle=True)

    for current, prepost in dloader:
        print(current['padded'].shape)
        print(prepost['pre_padded'].shape)
        print(prepost['post_padded'].shape)
        input()



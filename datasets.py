import itertools
from vocabulary import Voc
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



##################################################################
##################################################################
##################################################################

# Dialog dataset sentence trigrams s(s_i-1, s_i, s_i+1).

class DialogsTrigramDataset(Dataset):
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


def get_train_test_loader(loadfile='data.pt', **kwargs):
    ''' Split data into train/test sets and return dataloaders'''
    from sklearn.model_selection import train_test_split

    data = torch.load(loadfile)
    trigrams = data['trigrams']
    vocab = Voc('skip_thought')
    vocab.__dict__ = data['vocab_dict']

    train, test = train_test_split(trigrams, test_size=0.1)

    train_dset = DialogsTrigramDataset(train, vocab)
    test_dset = DialogsTrigramDataset(test, vocab)

    train_loader = DataLoader(train_dset,
                              collate_fn=trigrams_collate_fn, **kwargs)
    test_loader = DataLoader(test_dset,
                             collate_fn=trigrams_collate_fn, **kwargs)
    return train_loader, test_loader


def get_dataloader(loadfile='data/dialog/PERSON_mincount_4.pt', **kwargs):
    data = torch.load(loadfile)
    trigrams = data['trigrams']
    vocab = Voc('skip_thought')
    vocab.__dict__ = data['vocab_dict']
    dset = DialogsTrigramDataset(trigrams, vocab)
    return DataLoader(dset, collate_fn=trigrams_collate_fn, **kwargs)


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



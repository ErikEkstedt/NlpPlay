import tqdm
from os.path import join


CORNELL_ROOT = '../cornell_movie'

# Coherence Preprocessing
def generate_random_response_data(pairs, vocab, min_len=3, max_len=20):
    '''
    Assumes the vocab contains 3 special tokens at the 3 first indices
    Returns:

    list [context_string, response_string, label_int]
    '''
    negative_pairs = []
    for n in tqdm(range(len(pairs))):
        real_context = pairs[n][0]
        fake_length = np.random.randint(min_len, max_len)
        fake_idx = np.random.randint(3, vocab.num_words, fake_length)
        fake_response = ' '.join([vocab.index2word[x] for x in fake_idx])
        negative_pairs.append([real_context, fake_response, 0])
    return negative_pairs


def generate_choose_random_response_data(pairs, vocab):
    negative_pairs = []
    for n in tqdm(range(len(pairs))):
            real_context = pairs[n][0]
            fake_response = random.choice(pairs)[1]
            negative_pairs.append([real_context, fake_response, 0])
    return negative_pairs


def save_train_validation_split(filename=None,
                                out_filename=None,
                                val_split=0.1,
                                gen_neg_data_function=None):

    def add_label_to_pairs(pairs, label=1):
        for p in pairs:
            p.append(label)

    if filename is None:
        filename = join(CORNELL_ROOT,'data_maxlen_None_trim_4_pairs_189173.pt')

    if out_filename is None:
        out_filename = join(CORNELL_ROOT, 'train_val_{}.pt'.format(val_split))

    print('Loading pairs')
    data = torch.load(filename)
    pairs = data['pairs']

    print('Split training and validation')
    val_samples = int(val_split*len(pairs))
    val_pairs = []
    for i in tqdm(range(val_samples)):
        random_idx = random.randint(0, len(pairs)-1)
        val_pairs.append(pairs.pop(random_idx))

    if gen_neg_data_function is not None:
        add_label_to_pairs(pairs, 1)  # 1 is coherent
        add_label_to_pairs(val_pairs, 1)  # 1 is coherent

        print('Generating negative train samples')
        neg_train_pairs = gen_neg_data_function(pairs, vocab)
        pairs = pairs+neg_train_pairs
        random.shuffle(pairs)

        print('Generating negative validation samples')
        neg_val_pairs = gen_neg_data_function(val_pairs, vocab)
        val_pairs = val_pairs+neg_val_pairs
        random.shuffle(val_pairs)

    split = {'train_pairs': pairs,
            'val_pairs': val_pairs,
            'vocab_dict': data['vocab_dict']}

    print('Save split as: ', out_filename)
    torch.save(split, out_filename)


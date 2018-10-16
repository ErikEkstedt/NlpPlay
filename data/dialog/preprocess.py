from io import open
from tqdm import tqdm
import itertools
import numpy as np


# Preprocess txt file
def load_lines(filename='data/dialog/dialogs.txt'):
    print('Loading dialogs')
    return open(filename, encoding='utf-8').read().strip().split('\n')


def extract_conversations(lines):
    print('Get separate conversations...')
    conversations = []
    tmp_conv = []
    for line in tqdm(lines):
        if len(line) > 2:  # "\r\n" - newline
            tmp_conv.append(line[:-1])
        else:
            conversations.append(tmp_conv)
            tmp_conv = []
    print('Number of conversations: ', len(conversations))
    return conversations


def structure_as_trigrams(conversations):
    print('Get trigrams (s_(i-1), s_i, s_(i+1)')
    trigrams = []
    for conv in tqdm(conversations):
        for i in range(1, len(conv)-1):
            trigrams.append({'current': conv[i],
                             'pre': conv[i-1],
                             'post':conv[i+1]})

    print('Number of trigram examples (overlap): ', len(trigrams))
    return trigrams


def save_trigrams_to_disk(filename='dialogs.txt', savename='dialogs_trigrams_dict'):
    '''
    Writes dialogs as trigrams ( s_(i-1),  s_i, s_(i+1) ) to disk as .npy file

    trigrams[i]['pre']
    trigrams[i]['input']
    trigrams[i]['post']

    Arguments:

        filename:name of txt file (default='dialogs.txt')
        save:   name of file written to disk (default='dialogs_trigrams_dict')

    '''
    lines = load_lines(filename)
    conversations = extract_conversations(lines)
    trigrams = structure_as_trigrams(conversations)
    np.save(savename, trigrams)


def replace_names_with_PERSON(conversations, nlp):
    import spacy
    import time
    new_convs = []
    t = time.time()
    for conv in tqdm(conversations):
        # print(conv)
        tmp_conv = []
        doc = nlp(''.join(conv))
        for sentence in conv:
            s = sentence
            for x in doc.ents:
                if x.label_ in ['PERSON', 'ORG']:
                    # print(x.text,': ', x.label_)
                    s = s.replace(x.text, x.label_)
            tmp_conv.append(s)
        new_convs.append(tmp_conv)
    print(len(new_convs), ' conversations')
    print('took ', time.time() - t, ' seconds')
    return new_convs


def trim_rare_words(voc, trigrams, MIN_COUNT):
    '''Trim words used under the MIN_COUNT from the voc'''

    # Filter out trigrams with trimmed words
    voc.trim(MIN_COUNT)

    keep_tri = []
    for tri in trigrams:
        current_sentence = tri['current']
        pre_sentence = tri['pre']
        post_sentence = tri['post']
        keep_trigram = True

        # Check all sentences: If a word is not in word2index. dont keep
        # datapoint:
        c = current_sentence.split(' ')
        pr = pre_sentence.split(' ')
        po = post_sentence.split(' ')
        for x, y, z in zip(c, pr, po):
            if x not in voc.word2index:
                keep_trigram = False
                break
            if y not in voc.word2index:
                keep_trigram = False
                break
            if z not in voc.word2index:
                keep_trigram = False
                break

        # Only keep trigrams that do not contain trimmed word(s) in their input or output sentence
        if keep_trigram:
            keep_tri.append(tri)

    print("Trimmed from {} trigrams to {}, {:.1f}% of total".format(len(trigrams),
                                                                len(keep_tri),
                                                                100*len(keep_tri) / len(trigrams)))
    return keep_tri


def trim_convs(convs, voc):
    keep_convs = []
    for conv in convs:
        keep = True
        for sentence in conv:
            # print(type(sentence)) # str
            for word in sentence.split(' '):
                if not word in voc.word2index:
                    keep = False
        if keep:
            keep_convs.append(conv)
    print('Conversations: {}/{}'.format(len(keep_convs), len(convs)))
    print('Conversations: {}%'.format(len(keep_convs)*100 / len(convs)))
    print('total Conv after trim:', len(keep_convs))
    return keep_convs

if __name__ == "__main__":
    # Replace names (takes around 10 min on macbook pro)
    # lines = load_lines('data/dialogs.txt')
    # conversations = extract_conversations(lines)
    # trigrams = structure_as_trigrams(conversations)
    # new_convs = replace_names_with_PERSON(conversations, nlp)
    # new_trigrams = structure_as_trigrams(new_convs)
    # old_voc = Voc('old')
    # old_voc.add_trigrams(trigrams)
    # new_voc = Voc('new')
    # new_voc.add_trigrams(new_trigrams)

    new_convs = list(np.load('data/dialog/PERSON_convs.npy'))
    new_trigrams = structure_as_trigrams(new_convs)

    min_count = 4
    new_voc = Voc('new')
    new_voc.add_trigrams(new_trigrams)
    new_voc.trim(min_count)
    newest_convs = trim_convs(new_convs, new_voc)
    trigrams = structure_as_trigrams(newest_convs)

    d = torch.load('data/dialog/PERSON_mincount_4.pt')

    vocab = d['vocab']
    trigrams = d['trigrams']


    data = {'vocab_dict': vocab.__dict__ , 'trigrams': trigrams}

    torch.save(data, 'data/dialog/PERSON_mincount_4.pt')

    min_count = 3
    new_voc = Voc('new')
    new_voc.add_trigrams(new_trigrams)
    new_voc.trim(min_count)
    newest_convs = trim_convs(new_convs, new_voc)
    trigrams = structure_as_trigrams(newest_convs)

    data = {'vocab': new_voc, 'trigrams': trigrams}
    torch.save(data, 'PERSON_mincount_3.pt')



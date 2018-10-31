import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from  os.path import join


class ProtoAlexaData(object):
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.path_convs = self.load_path_conversations()
        self.convs = self.load_conversations()
        # self.create_complete_dataset()  # self.conversations

    def create_complete_dataset(self):
        print('reading conversations (14sek)')
        conversations = pd.read_csv(join(self.data_path, 'conversation.csv'))
        print('Done')
        convs = []
        tmp_conv = []
        print('sorting by conversation id')
        sorted_by_conv_id = conversations.sort_values(by='conversation_id')
        print('collecting conversations')
        for j, (i, p) in enumerate(sorted_by_conv_id.iterrows()):
            if j == 0:
                # I hate this
                conv_id = str(p['conversation_id'])
            if not p['conversation_id'] == conv_id:
                convs.append(tmp_conv)
                tmp_conv = []
                conv_id = p['conversation_id']
            tmp_conv.append(str(p['user_utterance']))
            tmp_conv.append(str(p['system_utterance']))
        self.conversations = convs

    def create_path_dataset(self):
        self.node = pd.read_csv(join(self.data_path, 'node.csv'))
        self.node['is_user'] = self.node['_path'].apply(lambda x: len(x.split('.')) % 2 == 1)
        self.node = self.node[(self.node['is_user'] == False) & (self.node['active'] == 't')]
        self.node_utterance = pd.read_csv(join(self.data_path,'node_utterance.csv'))
        self.utterance = pd.read_csv(join(self.data_path,'utterance.csv'))
        conversations = []
        for i, p in tqdm(list(self.node.iterrows())):
            conversation = []
            for e in p['_path'].split('.'):
                utterance_id = self.node_utterance[self.node_utterance['node_id'] == int(e)]['utterance_id']
                sampled_utterance = int(utterance_id.sample().values[0])
                utterance = self.utterance[self.utterance['id'] == sampled_utterance]
                conversation.append(utterance['utterance_text'].values.tolist()[0])
            conversations.append(conversation)
        self.path_conversations = conversations

    def load_path_conversations(self, filepath = 'data/alexa_path_conversations.csv'):
        convs = []
        with open(filepath, newline='\n') as csvfile:
            nodes = csv.reader(csvfile, delimiter=' ', quotechar='"')
            for row in nodes:
                convs.append(row)
        return convs

    def load_conversations(self, filepath = 'data/alexa_all_conversations.csv'):
        convs = []
        with open(filepath, newline='\n') as csvfile:
            nodes = csv.reader(csvfile, delimiter=' ', quotechar='"')
            for row in nodes:
                convs.append(row)
        # self.convs = convs
        return convs

    def path_save_to_disk(self, savename='data/alexa_conversations.csv'):
        delimiter = ' '
        with open(savename, 'w', newline='\n') as csvfile:
            data_writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"' )
            for conv in self.path_conversations:
                data_writer.writerow(pd.Series(conv))

    def complete_save_to_disk(self, savename='data/alexa_conversations.csv'):
        delimiter = ' '
        with open(savename, 'w', newline='\n') as csvfile:
            data_writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"' )
            for conv in self.conversations:
                data_writer.writerow(pd.Series(conv))

    def plot_path_lengths(self, convs):
        def print_statistics(lens, title='Length'):
            print(title)
            print('Max: ', lens.max())
            print('Min: ', lens.min())
            print('Mean: ', lens.mean())
            print('std: ', lens.std())
            print()

        def normal(bins, mu, sigma):
            const = 1/(sigma * np.sqrt(2 * np.pi))
            normal_exp =  np.exp( - (bins - mu)**2 / (2 * sigma**2) )
            return const*normal_exp

        conv_lens = np.array([len(conv) for conv in convs])
        utt_lens = np.hstack([np.array([len(utt) for utt in conv]) for conv
                              in convs])

        print('Conversations: ', len(convs))
        print_statistics(conv_lens, 'Conversation length')
        print_statistics(utt_lens, 'Utterance length')

        fig = plt.figure()
        fig.suptitle('Alexa Conversations: %d' % len(self.convs), fontsize=14, fontweight='bold')

        # Conversation
        mu = conv_lens.mean()
        sigma = conv_lens.std()
        bins = np.linspace(2,10,100)

        ax = fig.add_subplot(211)
        ax.set_title('Conversation')
        ax.set_ylabel('Probability density')

        count, _, ignored = plt.hist(conv_lens, 100, density=True) # (, normed=True)
        ax.plot(bins, normal(bins, mu, sigma), linewidth=2, color='r')

        # Utterances
        mu = utt_lens.mean()
        sigma = utt_lens.std()
        bins = np.linspace(0,utt_lens.max(),100)

        ax = fig.add_subplot(212)
        ax.set_title('Utterences')
        ax.set_xlabel('Length')
        ax.set_ylabel('Probability density')

        count, _, ignored = plt.hist(utt_lens, 100, density=True) # (, normed=True)
        ax.plot(bins, normal(bins, mu, sigma), linewidth=2, color='r')
        plt.title('Utterance')
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.show()


if __name__ == "__main__":

    dset = ProtoAlexaData()

    dset.plot_path_lengths(dset.convs)

    dset.plot_path_lengths(dset.path_convs)


    conversations = pd.read_csv(join('data', 'conversation_mini.csv'))
    conversations = pd.read_csv(join('data', 'conversation_small.csv'))

    import time
    t = time.time()
    conversations = pd.read_csv(join('data', 'conversation.csv'))
    t = time.time() - t
    print('took: ', t)

    for j, (i, p) in tqdm(enumerate(list(sort.iterrows()))):

    convs = []
    tmp_conv = []
    sorted_by_conv_id = conversations.sort_values(by='conversation_id')
    for j, (i, p) in enumerate(sorted_by_conv_id.iterrows()):
        if j == 0:
            # I hate this
            conv_id = str(p['conversation_id'])
        print(p)
        input()
        if not p['conversation_id'] == conv_id:
            convs.append(tmp_conv)
            tmp_conv = []
            conv_id = p['conversation_id']
        tmp_conv.append(str(p['user_utterance']))
        tmp_conv.append(str(p['system_utterance']))

    convs = []

    dset.load_conversations()
    dset.plot_lengths()

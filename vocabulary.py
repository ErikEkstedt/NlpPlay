
class Voc:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    def __init__(self, name):
        self.name = name  # MEMORY: Useful later on if saving the vocab.__dict__
        self.trimmed = False
        self.word2index = {}
        self.word2count = {} 
        self.index2word = {self.PAD_token: "PAD",
                           self.SOS_token: "SOS",
                           self.EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def add_trigrams(self, trigrams):
        print("Adding words to Vocab...")
        for tri in tqdm(trigrams):
            self.add_sentence(tri['current'])
            self.add_sentence(tri['pre'])
            self.add_sentence(tri['post'])

    def add_pairs(self, pairs):
        print("Adding words to Vocab...")
        for pair in tqdm(pairs):
            self.add_sentence(pair[0])
            self.add_sentence(pair[1])

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def sort_words_by_count(self):
        return sorted(self.word2count.items(), key=lambda kv: kv[1])

    def trim(self, min_count):
        '''Remove words below a certain count threshold'''
        if self.trimmed:
            print('Already Trimmed to: ', self.min_count)
            print('...But Trimming again to: ', min_count)
        self.trimmed = True
        self.min_count = min_count

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)



if __name__ == "__main__":


    voc = Voc('test')

    data = 'Hello my name is Mr. Machine Learning and I am stupid'

    # Add sentence

    voc.add_sentence(data)

    print('Unique words: ', voc.num_words)
    print('Word to index: ', voc.word2index)
    print('Index to word: ', voc.index2word)
    print('word count: ', voc.word2count)

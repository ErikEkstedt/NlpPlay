from typing import Dict
import logging
import numpy as np

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger('seq2seq_custom')  # pylint: disable=invalid-name

class Seq2Seq(DatasetReader):
    def __inigt__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path):
        ''' Read data from disk.
        Get strings for source and target.
        call text_to_instance to return iterable[Instance]
        '''
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)

    def _text_to_field(self, s):
        tokenized = self._source_tokenizer.tokenize(s)  # tokenize
        tokenized.insert(0, Token(START_SYMBOL))  # insert START
        tokenized.append(Token(END_SYMBOL))  # insert END
        return TextField(tokenized, self._source_token_indexers)

    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        source_field = self._text_to_field(source_string)

        if target_string is not None:
            target_field = self._text_to_field(source_string)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})

class Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.
    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})


train_path = "ELMo/seq2seq_small_train.tsv"
val_path = "ELMo/seq2seq_small_val.tsv"

reader = Seq2SeqDatasetReader()

# lists of instances. first _read from disk then `text_to_instance`.
train_dataset = reader.read(train_path)
val_dataset = reader.read(val_path)
vocab = Vocabulary.from_instances(train_dataset + val_dataset)

sample_instance = train_dataset[0]
# {'source_tokens': <allennlp.data.fields.text_field.TextField at 0x7ff5882a25f8>,
#  'target_tokens': <allennlp.data.fields.text_field.TextField at 0x7ff5882a2780>}

print(sample_instance.fields['source_tokens'].tokens)
print(sample_instance.fields['target_tokens'].tokens)
# [@start@, have, you, heard, of, the, upcoming, black, panther, movie, @end@]
# [@start@, i, have, and, i, am, so, in, love, with, the, trailer, already, @end@]

print(vocab.get_vocab_size())


# Model
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# embeddings
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmTagger(word_embeddings, lstm, vocab) 
optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BucketIterator(batch_size=2, sorting_keys=[("source_tokens",
                                                       "target_tokens")])
iterator.index_with(vocab)

# Training
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=val_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()

# TODO
# ~/miniconda3/envs/allennlp/lib/python3.7/site-packages/allennlp/data/iterators/bucket_iterator.py in <listcomp>(.0)
#      34             padding_lengths = noisy_lengths
#      35         instance_with_lengths = ([padding_lengths[field_name][padding_key]
# ---> 36                                   for (field_name, padding_key) in sorting_keys],
#      37                                  instance)
#      38         instances_with_lengths.append(instance_with_lengths)

# KeyError: 'target_tokens'

# Prediction
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

sentence = "The dog ate the apple"
tag_logits = predictor.predict(sentence)['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)

print(sentence)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])











# ELMo
def ELMOOO():
    from allennlp.modules.elmo import Elmo, batch_to_ids
    options_file = "https://s3-us-west-2.amazonaws.com/"
    options_file += "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/"
    options_file += "elmo_2x4096_512_2048cnn_2xhighway_options.json"

    weight_file = "https://s3-us-west-2.amazonaws.com/"
    weight_file += "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/"
    weight_file += "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 2, dropout=0)

    for c in dataset:
        character_ids = batch_to_ids(c)  # tensor shape: [2, 3, 50]
        print('characters id: ', character_ids.shape)
        embeddings = elmo(character_ids)
        embs = embeddings['elmo_representations']  # list [layer1, layer2]
        masks = embeddings['mask']
        print('First layer: ', embs[0].shape)
        print('Second layer: ', embs[1].shape)
        input()

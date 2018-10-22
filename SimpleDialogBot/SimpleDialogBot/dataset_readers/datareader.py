from overrides import overrides
from typing import Dict

import logging

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2seqDialogDataReader")
class Seq2SeqDialog(DatasetReader):
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
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                source_target = line.split('\t')
                if not len(source_target) == 2:
                    continue
                source, target = source_target
                yield self.text_to_instance(source, target)

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

    # @overrides
    # def text_to_instance(self, source: str, target: str) -> Instance:  # type: ignore
    #     # pylint: disable=arguments-differ
    #     tokenized_source = self._tokenizer.tokenize(source)
    #     tokenized_target = self._tokenizer.tokenize(target)
    #     source_field = TextField(tokenized_source, self._token_indexers)
    #     target_field = TextField(tokenized_target, self._token_indexers)
    #     fields = {'source_tokens': source_field, 'target_tokens': target_field}
    #     return Instance(fields)


# Tests are awesome!
class TestSeq2SeqDialog(AllenNlpTestCase):
    def test_read_from_file(self, filepath='data/dialog/seq2seq_small_val.tsv'):
        reader = Seq2SeqDialog()
        instances = ensure_list(reader.read(filepath))

        instance1 = {"source_tokens": ["Yea", "he", "'s", "the", "best"],
                     "target_tokens": ["Robert", "Shaw", "played", "Quint", "He", "was", "great"]}
        instance2 = {"source_tokens": ["Robert", "Shaw", "played", "Quint", "He", "was", "great"],
                     "target_tokens": ["Of", "course", "so", "was", "Richard", "Dreyfuss"]}
        instance3 = {"source_tokens": ["Of", "course", "so", "was", "Richard", "Dreyfuss"],
                     "target_tokens": ["Roy", "Schneider", "was", "my", "favorite"]}
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == instance1["source_tokens"]
        assert [t.text for t in fields["target_tokens"].tokens] == instance1["target_tokens"]
        fields = instances[1].fields
        assert [t.text for t in fields["source_tokens"].tokens] == instance2["source_tokens"]
        assert [t.text for t in fields["target_tokens"].tokens] == instance2["target_tokens"]
        fields = instances[2].fields
        assert [t.text for t in fields["source_tokens"].tokens] == instance3["source_tokens"]
        assert [t.text for t in fields["target_tokens"].tokens] == instance3["target_tokens"]
        print('Test Passed!')


if __name__ == "__main__":
    print('Does the reader pass the tests?')
    dataset_reader_test = TestSeq2SeqDialog()
    dataset_reader_test.test_read_from_file()
    # Passed the tests! Yay!

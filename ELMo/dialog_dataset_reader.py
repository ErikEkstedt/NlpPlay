from typing import Dict
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

# Tests
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# This is a sequence to sequence chatbot task trained on sequences of utterences
# The data is located in a .tsv file where each line contains two utterences:
#   source "\t" target
# these are seperated by tab

@DatasetReader.register("seq2seqdialog")
class Seq2Seq_dialog(DatasetReader):
    def __init__(self, lazy: bool = False, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

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
    def text_to_instance(self, source: str, target: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._tokenizer.tokenize(source)
        tokenized_target = self._tokenizer.tokenize(target)
        source_field = TextField(tokenized_source, self._token_indexers)
        target_field = TextField(tokenized_target, self._token_indexers)
        fields = {'source': source_field, 'target': target_field}
        return Instance(fields)


# Tests are awesome!
class TestSeq2Seq_dialog(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = Seq2Seq_dialog()
        instances = ensure_list(reader.read('data/dialog/seq2seq_small_val.tsv'))

        instance1 = {"source": ["Yea", "he", "'s", "the", "best"],
                     "target": ["Robert", "Shaw", "played", "Quint", "He", "was", "great"]}
        instance2 = {"source": ["Robert", "Shaw", "played", "Quint", "He", "was", "great"],
                     "target": ["Of", "course", "so", "was", "Richard", "Dreyfuss" ]}
        instance3 = {"source": ["Of", "course", "so", "was", "Richard", "Dreyfuss" ],
                     "target": ["Roy", "Schneider", "was", "my", "favorite"]}
        fields = instances[0].fields
        assert [t.text for t in fields["source"].tokens] == instance1["source"]
        assert [t.text for t in fields["target"].tokens] == instance1["target"]
        fields = instances[1].fields
        assert [t.text for t in fields["source"].tokens] == instance2["source"]
        assert [t.text for t in fields["target"].tokens] == instance2["target"]
        fields = instances[2].fields
        assert [t.text for t in fields["source"].tokens] == instance3["source"]
        assert [t.text for t in fields["target"].tokens] == instance3["target"]
        print('Test Passed!')


# Does the reader pass the tests?
dataset_reader_test = TestSemanticScholarDatasetReader()
dataset_reader_test.test_read_from_file()

# Passed the tests! Yay!

# Model
from allennlp.common.testing import ModelTestCase

class Seq2SeqDialogModel(ModelTestCase):
    def setUp(self):
        super(Seq2SeqDialogModel, self).setUp()
        self.set_up_model('data/dialog/Seq2Seq_dialog_config.json',
                          'data/dialog/seq2seq_small_val.tsv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

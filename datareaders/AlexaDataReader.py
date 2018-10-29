from typing import Dict
import csv
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("Alexa")
class AlexaDatasetReader(DatasetReader):
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = source_tokenizer or WordTokenizer()
        self._token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._source_add_start_token = source_add_start_token

    @overrides
    def _read(self, filepath):
        # with open(cached_path(file_path), "r") as data_file:
        with open(filepath, "r", newline='\n') as csvfile:
            logger.info("Reading instances from lines in file at: %s", filepath)
            conversations = csv.reader(csvfile, delimiter=' ', quotechar='"')
            for conv in conversations:
                for context, target in zip(conv[:-1], conv[1:]):
                    yield self.text_to_instance(context, target)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # Source tokens
        tokenized_source = self._tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._token_indexers)

        # Target tokens
        tokenized_target = self._tokenizer.tokenize(target_string)
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._token_indexers)
        return Instance({"source_tokens": source_field, "target_tokens": target_field})

if __name__ == "__main__":
    reader = AlexaDatasetReader()
    filepath = '/home/erik/NLP/NlpPlay/data/alexa/test.csv'
    instances = ensure_list(reader.read(filepath))

'''
Following tutorial:
https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/predicting_paper_venues/predicting_paper_venues_pt1.md
'''

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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# In this learning scenario we are going to classify what "venue" research
# papers based on their "abstract" and "title".
# This means that we will use a dataset where each datapoint contains:
#   title
#   abstract
#   venue
# A datapoint is an instance in allennlp world. For this example an instance
# will consist of three Fields.

@DatasetReader.register("s2_papers")
class SemanticScholarDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.
    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``
    where the ``label`` is derived from the venue of the paper.
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
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
                paper_json = json.loads(line)
                title = paper_json['title']
                abstract = paper_json['paperAbstract']
                venue = paper_json['venue']
                yield self.text_to_instance(title, abstract, venue)

    @overrides
    def text_to_instance(self, title: str, abstract: str, venue: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields = {'title': title_field, 'abstract': abstract_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields)


# Build tests to see that the DatasetReader does what its supposed to.
# Inherit useful things form `AllenNlpTestCase`

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list


class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = SemanticScholarDatasetReader()
        instances = ensure_list(reader.read('data/open_research_corpus/s2_papers.jsonl'))

        instance1 = {"title": ["Interferring", "Discourse", "Relations", "in", "Context"],
                     "abstract": ["We", "investigate", "various", "contextual", "effects"],
                     "venue": "ACL"}
        instance2 = {"title": ["GRASPER", ":", "A", "Permissive", "Planning", "Robot"],
                     "abstract": ["Execut", "ion", "of", "classical", "plans"],
                     "venue": "AI"}
        instance3 = {"title": ["Route", "Planning", "under", "Uncertainty", ":", "The", "Canadian",
                               "Traveller", "Problem"],
                     "abstract": ["The", "Canadian", "Traveller", "problem", "is"],
                     "venue": "AI"}

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["title"].tokens] == instance1["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance1["abstract"]
        assert fields["label"].label == instance1["venue"]
        fields = instances[1].fields
        assert [t.text for t in fields["title"].tokens] == instance2["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance2["abstract"]
        assert fields["label"].label == instance2["venue"]
        fields = instances[2].fields
        assert [t.text for t in fields["title"].tokens] == instance3["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance3["abstract"]
        assert fields["label"].label == instance3["venue"]
        print('Test Passed!')


# Does the reader pass the tests?
dataset_reader_test = TestSemanticScholarDatasetReader()
dataset_reader_test.test_read_from_file()


# MODEL
# Register with allennlp api. to be able to build model using json
@Model.register("paper_classifier")
class AcademicPaperClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AcademicPaperClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                title: Dict[str, torch.LongTensor],
                abstract: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        '''The first thing to notice are the inputs to the method. Remember the
        DatasetReader we implemented? It created Instances with fields named title,
        abstract, and label. That's where the names to forward come from - they have to
        match the names that we gave to the fields in our DatasetReader. AllenNLP will
        take the instances read by your DatasetReader, group them into batches, pad all
        of the Fields from each instance in the batch to be the same shape, and then
        produce one batched array (or set of arrays) for each Field in your
        Instances.

        Note that we require you to pass the label to forward, in addition to
        the model's inputs - in order to have a flexible yet sane training loop,
        we need Model.forward to compute its own loss. The training code will
        look for the loss value in the dictionary returned by forward, and
        compute the gradients of that loss to update the model's parameters. But
        also notice that the label input is optional. This is necessary for you
        to be able to use this model in situations where you don't have a label,
        such as in a demo, or if you want this model to be a component in some
        larger model.

        Next, let's look at the types of the inputs. The label is simple: it's
        just a tensor of shape (batch_size, 1), with one label id for each
        instance in the batch. The other two are a little more complicated.
        Remember that the title and abstract were TextFields; those get
        converted into a dictionary of pytorch tensors. It's a dictionary
        instead of a single tensor to be flexible about how exactly words are
        represented in your model. One element of this dictionary might have a
        tensor of word ids, one might have a tensor of character ids for each
        word, and one might have a tensor of part of speech tag ids. But your
        model doesn't have to care about what exactly is in that dictionary,
        because it just passes the dictionary on to the TextFieldEmbedder that
        we took as a constructor parameter. You need to be sure that the
        TextFieldEmbedder is expecting the same thing that your DatasetReader is
        producing, but that happens in the configuration file, and we'll talk
        about it later.
        '''

        embedded_title = self.text_field_embedder(title)
        title_mask = util.get_text_field_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        embedded_abstract = self.text_field_embedder(abstract)
        abstract_mask = util.get_text_field_mask(abstract)
        encoded_abstract = self.abstract_encoder(embedded_abstract, abstract_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_abstract], dim=-1))
        class_probabilities = F.softmax(logits)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict


# This time we'll use allennlp.common.testing.ModelTestCase. For any
# model that you write, you want to be sure that it can train successfully, that
# you can save it and load it and get the same predictions back, and that its
# predictions are consistent whether the data is batched or not. ModelTestCase
# has easy tests for all of these. With just the code above, you can make sure
# your model works correctly on a tiny dataset. We strongly recommend that you
# write and debug your code using these tests, as it is way easier and faster to
# find problems using a test fixture than using your large dataset.  from
# allennlp.common.testing import ModelTestCase

class AcademicPaperClassifierTest(ModelTestCase):
    def setUp(self):
        super(AcademicPaperClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/academic_paper_classifier.json',
                          'tests/fixtures/s2_papers.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


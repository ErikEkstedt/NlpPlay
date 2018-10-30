# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from AlexaAllen.datareaders.AlexaDataReader import AlexaDatasetReader

class TestAlexaDataReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = AlexaDatasetReader()
        filepath = '/home/erik/NLP/NlpPlay/data/alexa/test.csv'
        instances = ensure_list(reader.read(filepath))

        instance1 = {"conversation": ['tell me a fact',
                                      "An eagle can kill a young deer and fly away with it. Not very fun for the deer, but it's a fact!",
                                      'four',
                                      'Is the square root of 16.']}
        instance2 = {"conversation": ["what's going on",
                                      'Not much! What about you? Anything exciting happening today?',
                                      'no not really',
                                      "I'm sorry. Did you watch anything good on tv today?"]}
        instance3 = {"conversation": ["let's do music",
                                      "What's your favorite style of music?",
                                      'ha',
                                      "I don't believe I know that style."]}

        # assert len(instances) == 10
        # fields = instances[0].fields
        # assert [t.text for t in fields["conversation"].tokens] == instance1["conversation"]
        # fields = instances[1].fields
        # assert [t.text for t in fields["conversation"].tokens] == instance1["conversation"]
        # fields = instances[2].fields
        # assert [t.text for t in fields["conversation"].tokens] == instance1["conversation"]

if __name__ == "__main__":
    test = TestAlexaDataReader()
    test.test_read_from_file()

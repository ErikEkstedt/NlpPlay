from allennlp.data import Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data import Instance

from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.iterators import BucketIterator

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder

# Trainer
# Inside training this is the pass through model
# Get tqdm for the training batches
train_generator = self._iterator(self._train_data,
                                 num_epochs=1,
                                 shuffle=self._shuffle)
num_training_batches = self._iterator.get_num_batches(self._train_data)

train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)

for batch in train_generator_tqdm:
    batch = util.move_to_device(batch, self._cuda_devices[0])
    output_dict = self._model(**batch)


########################
# DATA
########################

# Namespaces are used to keep track of data throughout data/model pass
# Tokens & TokenIndexer -> Field 
# Fields -> Instance 
# Instance -> dataset

# Tokens & TokenIndexer.  TokenIndexer may be reused for all textfield of same type
token_indexers = {"word_tokens": SingleIdTokenIndexer(namespace="word_ids"),
                  "char_tokens": TokenCharactersIndexer(namespace='char_ids')}

tokens1 = list(map(Token, ["This", "movie", "was", "awful", "!"]))
tokens2 = list(map(Token, ["This", "movie", "was", "Great", "!"]))
# Fields
text_field1 = TextField(tokens1, token_indexers)
text_field2 = TextField(tokens2, token_indexers)
label_field1 = LabelField("negative", label_namespace="labels")
label_field2 = LabelField("positive", label_namespace="labels")

# Instances
instance1 = Instance({'data': text_field1, 'label': label_field1})
instance2 = Instance({'data': text_field2, 'label': label_field2})

# List of Instances. Gather all yout 'datapoints' = instances
list_instances = [instance1, instance2]

# Create Vocabulary. Find unique words and create index2words, word2index, ...
vocab = Vocabulary.from_instances(list_instances)
print('words: ', vocab.get_vocab_size('word_ids')) # words: 8
print('chars: ', vocab.get_vocab_size('char_ids'))  # chars: 19
print('labels: ', vocab.get_vocab_size('labels'))  # labels: 0

# On each datapoint=instance in your dataset call .index_fields(vocab)
# Code what index the words refer to. 
for instance in list_instances:
    instance.index_fields(vocab)

batch = Batch(list_instances)
tensor_dict = batch.as_tensor_dict(batch.get_padding_lengths())

########################
# Encoding 
########################

# Embeddings words and chars
word_embedding = Embedding(num_embeddings=vocab.get_vocab_size("word_ids"), embedding_dim=10)
char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("char_ids"), embedding_dim=5)

# Extra CNN to encode the characters to a fixed output
character_cnn = CnnEncoder(embedding_dim=5, num_filters=2, output_dim=8)
token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, encoder=character_cnn)

# Encoders
text_field_embedder = BasicTextFieldEmbedder({"word_tokens": word_embedding,
                                              "char_tokens": token_character_encoder})

encoded_data = text_field_embedder(tensor_dict['data'])
print('encoded data shape: ', encoded_data.shape)  # encoded data shape: tensor, shape (B, 5, 18)


########################
# Model 
########################

# Regular PyTorch model
import torch.nn as nn
model = nn.Linear(5*18, 2)

encoded = encoded_data.view(encoded_data.shape[0], -1)
out = model(encoded)





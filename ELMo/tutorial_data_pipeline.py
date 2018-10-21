from allennlp.data import Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data import Instance

from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch

from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

#########################################################################
#########################################################################
#####################      DATAPIPELINE             #####################
#########################################################################
#########################################################################
# A token can have a txt, idx, pos and many attributes
tokens = list(map(Token, ["This", "movie", "was", "awful", "!"]))
token_indexers = {"tokens": SingleIdTokenIndexer(namespace="token_ids")}
review = TextField(tokens, token_indexers)
review_sentiment = LabelField("negative", label_namespace="tags")

# Access the original strings and labels using the methods on the Fields.
print("Tokens in TextField: ", review.tokens)
print("Label of LabelField", review_sentiment.label)


# The TextField and the LabelField are paired together making an Instance

instance1 = Instance({"review": review, "label": review_sentiment})
print("Fields in instance: ", instance1.fields)
print("instance.fields.keys(): ", instance1.fields.keys())

# Create another
tokens = list(map(Token, ["This", "movie", "was", "quite", "slow", "but", "good", "."]))
review2 = TextField(tokens, token_indexers)
review_sentiment2 = LabelField("positive", label_namespace="tags")
instance2 = Instance({"review": review2, "label": review_sentiment2})

# After all datapoints have been ordered in text -> tokens -> instance
# Combine them all in a list
instances = [instance1, instance2]

# In order to get our tiny sentiment analysis dataset ready for use in a model, we need to be able to do a few things:
# * Create a vocabulary from the dataset (using Vocabulary.from_instances)
# * Collect the instances into a Batch (which provides methods for indexing and converting to tensors)
# * Index the words and labels in the Fields to use the integer indices specified by the Vocabulary
# * Pad the instances to the same length
# * Convert them into tensors.
#
# The Batch, Instance and Fields have some similar parts of their API.


# This will automatically create a vocab from our dataset.
# It will have "namespaces" which correspond to two things:
# 1. Namespaces passed to fields (e.g. the "tags" namespace we passed to our LabelField)
# 2. The keys of the 'Token Indexer' dictionary in 'TextFields'.
# passed to Fields (so it will have a 'tags' namespace).
vocab = Vocabulary.from_instances(instances)

print("This is the id -> word mapping for the 'token_ids' namespace: ")
print(vocab.get_index_to_token_vocabulary("token_ids"), "\n")
print("This is the id -> word mapping for the 'tags' namespace: ")
print(vocab.get_index_to_token_vocabulary("tags"), "\n")
print("Vocab Token to Index dictionary: ", vocab._token_to_index, "\n")
# Note that the "tags" namespace doesn't contain padding or unknown tokens.

# Next, we index our dataset using our newly generated vocabulary.
# This modifies the current object. You must perform this step before
# trying to generate arrays.
batch = Batch(instances)
batch.index_instances(vocab)

# Finally, we return the dataset as arrays, padded using padding lengths
# extracted from the dataset itself, which will be the max sentence length
# from our two instances.
padding_lengths = batch.get_padding_lengths()
print("Lengths used for padding: ", padding_lengths, "\n")
tensor_dict = batch.as_tensor_dict(padding_lengths)

print(tensor_dict)

#########################################################################
#########################################################################
#####################      EMBEDDING TOKENS         #####################
#########################################################################
#########################################################################


# Notice that here we use two Token Indexers
words = ["All", "the", "cool", "kids", "use", "character", "embeddings", "."]
sentence1 = TextField([Token(x) for x in words],
                      token_indexers={"tokens": SingleIdTokenIndexer(namespace="token_ids"),
                                      "characters": TokenCharactersIndexer(namespace="token_characters")})
words2 = ["I", "prefer", "word2vec", "though", "..."]
sentence2 = TextField([Token(x) for x in words2],
                      token_indexers={"tokens": SingleIdTokenIndexer(namespace="token_ids"),
                                      "characters": TokenCharactersIndexer(namespace="token_characters")})
instance1 = Instance({"sentence": sentence1})
instance2 = Instance({"sentence": sentence2})


# Make 
instances = [instance1, instance2]
vocab = Vocabulary.from_instances(instances)

print("This is the token_ids vocabulary we created: \n")
print(vocab.get_index_to_token_vocabulary("token_ids"))

print("This is the character vocabulary we created: \n")
print(vocab.get_index_to_token_vocabulary("token_characters"))

for instance in instances:
    instance.index_fields(vocab)


# We're going to embed both the words and the characters, so we create
# embeddings with respect to the vocabulary size of each of the relevant namespaces
# in the vocabulary.
word_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_ids"), embedding_dim=10)
char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_characters"), embedding_dim=5)
character_cnn = CnnEncoder(embedding_dim=5, num_filters=2, output_dim=8)

# This is going to embed an integer character tensor of shape: (batch_size, max_sentence_length, max_word_length) into
# a 4D tensor with an additional embedding dimension, representing the vector for each character.
# and then apply the character_cnn we defined above over the word dimension, resulting in a tensor
# of shape: (batch_size, max_sentence_length, num_filters * ngram_filter_sizes). 
token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, encoder=character_cnn)

# Notice that these keys have the same keys as the TokenIndexers when we created our TextField.
# This is how the text_field_embedder knows which function to apply to which array. 
# There should be a 1-1 mapping between TokenIndexers and TokenEmbedders in your model.
text_field_embedder = BasicTextFieldEmbedder({"tokens": word_embedding, "characters": token_character_encoder})

# Convert the indexed dataset into Pytorch Variables. 
batch = Batch(instances)
tensors = batch.as_tensor_dict(batch.get_padding_lengths())
print("Torch tensors for passing to a model: \n\n", tensors)
print("\n\n")
# tensors is a nested dictionary, first keyed by the
# name we gave our instances (in most cases you'd have more
# than one field in an instance) and then by the key of each
# token indexer we passed to TextField.

# This will contain two tensors: one from representing each
# word as an index and one representing each _character_
# in each word as an index. 
text_field_variables = tensors["sentence"]

# This will have shape: (batch_size, sentence_length, word_embedding_dim + character_cnn_output_dim)
embedded_text = text_field_embedder(text_field_variables)

dimensions = list(embedded_text.size())
print("Post embedding with our TextFieldEmbedder: ")
print("Batch Size: ", dimensions[0])
print("Sentence Length: ", dimensions[1])
print("Embedding Size: ", dimensions[2])


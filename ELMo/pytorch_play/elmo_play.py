# First example in tutorial. Use elmo to write elmo embeddings to disk as hdf5.
# echo "The cryptocurrency space is now figuring out to have the highest search on Google globally ." > sentences.txt
# echo "Bitcoin alone has a sixty percent share of global search ." >> sentences.txt
# allennlp elmo sentences.txt elmo_layers.hdf5 --all

import os
import h5py
from allennlp.modules.elmo import Elmo, batch_to_ids

# Create sentences.txt
os.system('echo "The cryptocurrency space \
          is now figuring out to have the \
          highest search on Google globally ." > sentences.txt')
os.system('echo "Bitcoin alone has a sixty\
          percent share of global search ."\
          >> sentences.txt')
os.system('cat sentences.txt')

# Call allennnlp elmo file.txt output_name.hdf5 --all to get encoding
os.system('allennlp elmo sentences.txt elmo_layers.hdf5 --all')


h5py_file = h5py.File("elmo_layers.hdf5", 'r')
embedding = h5py_file.get("0")

assert(len(embedding) == 3)  # one layer for each vector
assert(len(embedding[0]) == 16)  # an entry for words in the source sentence

print(embedding[0])
print(embedding[0].shape)


# Second example in tutorial. Use elmo in code
######################################################################

options_file = "https://s3-us-west-2.amazonaws.com/"
options_file += "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/"
options_file += "elmo_2x4096_512_2048cnn_2xhighway_options.json"

weight_file = "https://s3-us-west-2.amazonaws.com/"
weight_file += "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/"
weight_file += "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)  # tensor shape: [2, 3, 50]

embeddings = elmo(character_ids)

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector

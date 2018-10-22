# Source:
#   https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440
# Import our dependencies
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
import keras.layers as layers
from keras.models import Model
import numpy as np
from tqdm import tqdm

# Initialize session
sess = tf.Session()
K.set_session(sess)


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in tqdm(os.listdir(directory)):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentiment"].append(re.match("\d+_(\d+)\.txt",
                                              file_path).group(1))
            data["sentence"].append(f.read())
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))
    return train_df, test_df


# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# train_df, test_df = download_and_load_datasets()

# Load Data
home = os.path.expanduser('~')
dataset_path = os.path.join(home, '.keras/datasets/aclImdb')
train_df = load_dataset(os.path.join(os.path.dirname(dataset_path),
                                     "aclImdb", "train"))
test_df = load_dataset(os.path.join(os.path.dirname(dataset_path),
                                    "aclImdb", "test"))

train_df.head()

# Create datasets (Only take up to 150 words for memory)
train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:150]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['polarity'].tolist()

test_text = test_df['sentence'].tolist()
test_text = [' '.join(t.split()[0:150]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['polarity'].tolist()


# Model
elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)),
                      signature="default", as_dict=True)["default"]


input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(ElmoEmbedding, output_shape=(1024,))(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
pred = layers.Dense(1, activation='sigmoid')(dense)


model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(train_text,
          train_label,
          validation_data=(test_text, test_label),
          epochs=5,
          batch_size=32)

import torch

from models.simple import SimpleEncoder
from datasets import get_trigrams_trainval_loader

train_loader, val_loader = get_trigrams_trainval_loader(batch_size=2)
vocab = train_loader.dataset.vocab

enc = SimpleEncoder(vocab)

for data, target in train_loader:
    print('padded: ', data['padded'].shape)
    print('lengths: ', len(data['lengths']))
    print('padded pre: ', target['pre_padded'].shape)
    print('lengths pre: ', len(target['pre_lengths']))
    print('padded post: ', target['post_padded'].shape)
    print('lengths post: ', len(target['post_lengths']))
    input()

out, hidden = enc(data)

print(out['padded'].shape)
print(out['lengths'].shape)

y = out['padded'].detach()
length = out['lengths'].detach()


import matplotlib.pyplot as plt
output = []
for i, t in enumerate(y):
    print(t.shape)
    output.append(t[:length[i]])
    plt.matshow(output[i].detach().numpy())
    plt.show()



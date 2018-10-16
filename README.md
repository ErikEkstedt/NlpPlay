# NLP Play


```
NlpPlay
├── README.md
├── requirements.txt
├── data
│   ├── SICK
│   │   ├── SICK.txt
│   │   └── readme.txt
│   ├── beatles
│   │   └── beatles.txt
│   ├── coherence
│   │   └── preprocess.py
│   ├── cornell_movie
│   │   ├── data_maxlen_None_trim_4_pairs_189173.pt
│   │   ├── lines_of_pairs_movie_lines.txt
│   │   └── preprocess.py
│   └── dialog
│       ├── PERSON_mincount_4.pt
│       ├── README.md
│       ├── dialog_data_results.ipynb
│       ├── dialogs.txt
│       └── preprocess.py
├── datasets.py
├── models
│   ├── coherence.py
│   ├── pytorch_chatbot.py
│   └── skip_thought.py
└── vocabulary.py
```

## Data

Clean Dialog Data
- [x] Replace named enteties with special token
- [ ] Replace uncommon words with Part-Of-Speech tags. Perhaps: chihuawa -> <NN> 

Pytorch chatbot
- [x] Reformat PyTorch Chatbot Tutorial data code

Train some models to get familiarized
- [ ] Seq2Seq
- [ ] Seq2Seq with attention
  - [ ] PyTorch Chatbot Tutorial
- [ ] Transformer
- [ ] ELMo
- [ ] BERT

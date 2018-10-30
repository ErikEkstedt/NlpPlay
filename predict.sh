#!/bin/bash

# Simple_seq2seq
# allennlp predict \
#   training_log/model.tar.gz \
#   fixtures/input_test.jsonl \
#   --predictor simple_seq2seq \
#   --output-file fixtures/prediction.json \
#   --silent \
#   --cuda-device 0


allennlp predict \
  training_log/model.tar.gz \
  fixtures/input_test.jsonl \
  --predictor alexa_seq2seq \
  --output-file fixtures/prediction.json \
  --silent \
  --cuda-device 0 \
  --include-package AlexaAllen


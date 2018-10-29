#!/bin/bash

allennlp predict \
  training_log/model.tar.gz \
  fixtures/input_test.jsonl \
  --predictor simple_seq2seq \
  --output-file fixtures/prediction.json \
  --silent --cuda-device 0

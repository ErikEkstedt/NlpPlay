import os
import json

allen_msg = """allennlp predict \
    training_log/model.tar.gz \
    fixtures/input_test.jsonl \
    --predictor alexa_seq2seq \
    --output-file /tmp/prediction.json \
    --silent \
    --cuda-device 0 \
    --include-package AlexaAllen"""


os.system(allen_msg)

print('Done!')

tmp_filepath = '/tmp/prediction.json'

with open(tmp_filepath) as f:
    data = json.load(f)

# print('Input: ', )
print('System: ', data['predicted_tokens'])

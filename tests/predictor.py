# pylint: disable=no-self-use,invalid-name,unused-import
import time
import json

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

model_path = 'fixtures/model.tar.gz'

archive = load_archive(model_path)
predictor = Predictor.from_archive(archive, 'simple_seq2seq')

json_string = """ {"source": ""} """
inputs = json.loads(json_string)

ans = input('Chat with bot?(y/n)')
if ans =='y':
    while True:
        inputs['source'] = input('Tell bot: > ')
        t = time.time()
        result = predictor.predict_json(inputs)
        t = time.time() - t
        print('Input: ', inputs['source'])
        print('Predicton: ', " ".join(result['predicted_tokens']))
        print('Took: ', t, 's\n')

ans = input('bot vs bot? (y/n)')
if ans =='y':
    answer = 'Who are you?'
    while True:
        inputs['source'] = answer
        t = time.time()
        result = predictor.predict_json(inputs)
        t = time.time() - t
        answer = result['predicted_tokens']
        answer = " ".join(answer)
        print('Input: ', inputs['source'])
        print('Predicton: ', answer)
        print('Took: ', t, 's\n')

# pylint: disable=no-self-use,invalid-name,unused-import
import time
import json

from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


def chat_with_bot():
    while True:
        inputs['source'] = input('Tell bot: > ')
        t = time.time()
        result = predictor.predict_json(inputs)
        t = time.time() - t
        print('Input: ', inputs['source'])
        print('Predicton: ', " ".join(result['predicted_tokens']))
        print('Took: ', t, 's\n')


def bot_vs_bot():
    answer = input('A: ')
    i = 0
    last_a, last_b = answer, ""
    while True:
        inputs['source'] = answer
        t = time.time()
        result = predictor.predict_json(inputs)
        t = time.time() - t
        answer = result['predicted_tokens']
        answer = " ".join(answer)
        if i%2==0:
            if last_b == answer:
                print('B (repetition): ', answer)
                break
            else:
                last_b = answer
            print('B: ', answer, '(%.3f)'%t)
        else:
            if last_a == answer:
                print('A (repetition): ', answer)
                break
            else:
                last_a = answer
            print('A: ', answer, '(%.3f)'%t)
        i += 1


# model_path = 'fixtures/model.tar.gz'
model_path = 'last_training_log/model.tar.gz'

# --include-package AllexaAllen
package_name = 'BotAlexa'
import_submodules(package_name)

# Load modules
archive = load_archive(model_path)
# predictor = Predictor.from_archive(archive, 'simple_seq2seq')
predictor = Predictor.from_archive(archive, 'alexa_seq2seq')

json_string = """ {"source": ""} """
inputs = json.loads(json_string)

ans = input('Chat with bot?(y/n)')
if ans =='y':
    chat_with_bot()


ans = input('bot vs bot? (y/n)')
if ans =='y':
    bot_vs_bot()


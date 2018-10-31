# NLP with AllenNLP dependency

Everything is a work in progress and changes all the time...


* [tests](tests/)
  - Probably need to make real pip packages...
  - `python tests/shell_predictor.py`
    - python: cant find package
  - `python tests/programatical_predictor.py`
    - works in ipython
    - python: cant find package
  - run tests by `pytest` in project-root directory
* [train](train)
  - A bash file that simply contains the commands for training Allennlp packages
* [predict](predict)
  - A bash file that simply contains the commands for predicting Allennlp packages
* [services](services/)
  - Directory containing modified Allennlp web-demo
* [BotAlexa](BotAlexa/)
  - DataReaders, predictor and model for alexa dataset
* [BotDialog](BotDialog/)
  - Uses preexisting `simple_seq2seq` datareader and model
  - Dialog dataset

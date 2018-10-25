# NLP with AllenNLP dependency


* test
  - run tests by `pytest` in project-root directory
* train
```bash
allennlp train experiments/alexa_dialog.json \
  -s training_stats  \
  --include-package firstProject
```
* predict 
```bash
allennlp predict experiments/alexa_dialog.json -s training_stats  --include-package firstProject
```

# ALLENNLP help

#### allennlp train
```bash
positional arguments:
  param_path            path to parameter file describing the model to be
                        trained

optional arguments:
  -h, --help            show this help message and exit
  -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                        directory in which to save the model and its logs
  -r, --recover         recover training from the state in serialization_dir
  -o OVERRIDES, --overrides OVERRIDES
                        a JSON structure used to override the experiment
                        configuration
  --file-friendly-logging
                        outputs tqdm status on separate lines and slows tqdm
                        refresh rate
  --include-package INCLUDE_PACKAGE
                        additional packages to include
```


#### allennlp predict
```bash
positional arguments:
  archive_file          the archived model to make predictions with
  input_file            path to input file

optional arguments:
  -h, --help            show this help message and exit
  --output-file OUTPUT_FILE
                        path to output file
  --weights-file WEIGHTS_FILE
                        a path that overrides which weights file to use
  --batch-size BATCH_SIZE
                        The batch size to use for processing
  --silent              do not print output to stdout
  --cuda-device CUDA_DEVICE
                        id of GPU to use (if any)
  --use-dataset-reader  Whether to use the dataset reader of the original
                        model to load Instances
  -o OVERRIDES, --overrides OVERRIDES
                        a JSON structure used to override the experiment
                        configuration
  --predictor PREDICTOR
                        optionally specify a specific predictor to use
  --include-package INCLUDE_PACKAGE
                        additional packages to include
```

#### allennlp 

```bash
python -m allennlp.service.server_simple \
    --archive-path tests/fixtures/model.tar.gz \
    --predictor paper-classifier \
    --include-package my_library \
    --title "Academic Paper Classifier" \
    --field-name title \
    --field-name paperAbstract
```

```bash
Serve up a simple model

optional arguments:
  -h, --help            show this help message and exit
  --archive-path ARCHIVE_PATH
                        path to trained archive file
  --predictor PREDICTOR
                        name of predictor
  --weights-file WEIGHTS_FILE
                        a path that overrides which weights file to use
  -o OVERRIDES, --overrides OVERRIDES
                        a JSON structure used to override the experiment
                        configuration
  --static-dir STATIC_DIR
                        serve index.html from this directory
  --title TITLE         change the default page title
  --field-name FIELD_NAME
                        field names to include in the demo
  --port PORT           port to serve the demo on
  --include-package INCLUDE_PACKAGE
                        additional packages to include
```







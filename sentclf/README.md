# Package Information

Package ``sentclf`` provides a solution to experiment on the sentence classfication task.

## Structure

- [config.py](config.py): utility methods to handle configuration files
- [data.py](data.py): functions to read JSON files and convert them to DataFrame format needed for experiments
- [dataset.py](dataset.py): customized ``Dataset`` classes
- [default_config.json](default_config.json): default configuration file for experiments
- [experiment.py](experiment.py): contains the main class for experiments (``Experiment``). It reads the configuration
  file, initializes components (data loaders, model, optimizer, criterion, metric, trainer), performs training and
  evaluation.
- [modules.py](modules.py): contains the main model as well as its components. Specifically:
    - ``SentenceClassifier``: model for sentence classification
    - ``SentenceEmbedder``: model for computing sentence embeddings from token embeddings (provided by a BERT model)
    - ``Pooling``: different ways to compute sentence embeddings used by ``SentenceEmbedder``
- [system.py](system.py): the provide functionality for a trained ``SentenceClassifier`` model
- [trainer.py](trainer.py): trainer for ``SentenceClassifier`` models
- [utils.py](utils.py): utility methods

### Backlog (currently not used):

- [data_contextual.py](data_contextual.py): similar to [data.py](data.py) but for experiments with contextual
  information.
- [modules_contextual.py](modules.py): contains the contextual sentence classifier
  model ``ContextualSentenceClassifier``

## Experiment

Experiments are configured with a config file and saved in a subfolder of the folder in
``config['path']['output_dir']``. Experiment progress is printed out (if ``verbose`` is set to ``True``) and saved to
TensorBoard.

### Output

After training a model, the experiment folder contains these file:

- ``log``: experiment progress, can be viewed with TensorBoard
  (choose the parent folder of the experiment folder as ``logdir`` to compare the results of different
  runs/experiments):

```shell
tensorboard --logdir model_dir
```

- ``config.json``: the *actual* configuration of the experiment with *seeds* for reproduction
- ``best.pth``: parameters of the best model selected based on the chosen metric
- ``checkpoint.pth``: parameters of the latest model

### Config file format

```
{
  "data": {
    "dev_ratio": 0.1,             # the ratio or size of the dev set
    "test_ratio": 0.1,            # the ratio or size of the dev set
    "random_states": [            # the seed used to split data (for reproducibility)
      null,                       # null means that the seeds will be generated
      null
    ]
  },
  "dataset": {
    "name": "generic"             # dataset name for experiment folder name
  },
  "data_loaders": {
    "train_batch_size": 32,       # batch sizes in number of sentences
    "dev_batch_size": 64,
    "test_batch_size": 64
  },
  "model": {                      # model parameters
    "pretrained_model": "allenai/scibert_scivocab_uncased",
    "pooling_mode": "mean",
    "dropout_prob": 0.1,
    "use_extended_features": true
  },
  "system": {
    "max_sequence_length": 128     # max sequence length handled by the transformer model
  },
  "optimizer": {
    "type": "Adam",                # optimizer type
    "learning_rate": 5e-6,         # learning rate of the transformer model
    "learning_rate_top": 5e-4      # learning rate of the top layer
  },
  "metric": "macro_f1",            # metric used for model selection
  "trainer": {
    "num_epochs": 10,              # number of training epochs
    "interval": 100,               # iteration interval for validation/information printing
    "verbose": true                # if the training progress is printed out 
  },
  "seeds": {                       # seeds for the experiment (for reproducibility)
    "py_seed": null,               # null means that the seeds will be generated
    "np_seed": null,
    "torch_seed": null
  },
  "paths": {
    "output_dir": "model"          # the root folder, an experiment will be saved as a subfolder
  },
  "logger": {                      # logging format and level
    "format": "%(asctime)s - %(name)s - %(message)s",
    "level": "INFO"
  }
}
```
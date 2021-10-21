# Sentence Classification for Scientific Paper

This repository contains the codes for the sentence classification task tailored for the scientific paper domain
(Scientific Paper Sentence Classification - **SPSC**).

## Installation

Install the dependencies in [requirements.txt](requirements.txt).

## Challege Results

See [challenge/README.md](challenge/README.md). Please go through this document before reading about the results.

## Quick Start

### Data preparation and analysis

General: Prepare data in CSV format which contains at least two columns:
`sentence` (the sentence to classify) and `label` (the corresponding label).

SPSC:

- Use the script [prepare_data.py](prepare_data.py) to convert the data folder of JSON files into CSV format:

```shell
python prepare_data.py train_data SPSC.csv
```

- Data analysis is presented in the notebook [analyze_data.ipynb](analyze_data.ipynb)

### Experiment: Training and evaluation

- See [example.py](example.py) for how to config and run an experiment,

Or:

- Create a JSON file of experiment configuration similar to [config.json](config.json). It doesn't need to specify all
  parameters, the missing ones will get the default value
  from [sentclf/default_config.json](sentclf/default_config.json).
- In an experiment, the data will be split into train/dev/test splits and a model will be trained, evaluated and tested
  on the corresponding dataset.
- To train the model:

```shell
python run_experiment.py train --data_file SPSC.csv --config config.json --model model --device cuda
```

- The experiment will be saved in a subfolder of the specified output folder (using ``--model``). The default name
  pattern of an experiment is: ``<dataset_name>_<date>_<time>``, for example: `generic_210101_23-59-20`.
- To evaluate the model on the development and test sets:

 ```shell
python run_experiment.py eval --data_file SPSC.csv --config config.json --model model/generic_210101_23-59-20 --device cuda --restore_best
```

### Using the trained model

See the script in [run_model.py](run_model.py) for how to run the trained model independently.

## Package Structure

See [sentclf/README.md](sentclf/README.md)

## System Architecture

The sentence classifier consists of a pre-trained BERT model and a linear classifier on top. The model architecture is
defined in class [``SentenceClassifier``](sentclf/modules.py).

- First, the model compute the sentence representation from the token representations output by the BERT encoder. It
  provides 3 different modes to compute sentence representations: using the ``CLS`` token representation, ``mean``
  pooling or ``max`` pooling.
- Beside the sentence embeddings, the model can use ``extended_features`` (specifically for SPSC) to predict the label
  of a sentence, namely:
  ``xmin``, ``xmax``, ``ymin``, ``ymax`` and ``position`` (the position of the sentence in the paper).
- The loss function used to train the model is adjusted with label weights calculated on the data.

### SPSC

Use [SciBERT](https://github.com/allenai/scibert) as the base encoder.

### Proposal for architecture improvement

- The current classifier classifies each sentence *independetly*. However, we know that it is better to classify
  sentences *in context*, e.g., a ``paragraph`` is very likely follows a ``section`` or a ``table``/``figure`` needs
  a ``caption`` close by.
- Thus, a sentence classifier with *contextual information* can achieve better performance than the current classifier.
  It can be achieved by:
    - Use bidirectional LSTMs to enrich the sentences of the same paper with context
    - Decode their labels with a CRF layer.
- Up/down sampling training instances may work better than adjusting the loss weights.
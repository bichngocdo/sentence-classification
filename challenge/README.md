# Challenge

## Running the Solution

- The main script is [main.py](../main.py).
- Change the ``DEVICE`` and ``BATCH_SIZE`` variables at the beginning of the script to suit your system.
- Run the script from its location (the root folder), otherwise please adjust the ``MODEL_DIR``.

## Experiment Details

Due to time limit, I can only run two experiments with [SciBERT](https://github.com/allenai/scibert) as the base model:

- [Experiment1.ipynb](Experiment1.ipynb): experiment with the sentence classifier that uses *only* text as input.
- [Experiment2.ipynb](Experiment2.ipynb): experiment with the sentence classifier that uses text and additional features
  as inputs: ``xmin``, ``xmax``, ``ymin``, ``ymax`` and ``position``.
    - ``xmin``, ``ymin`` of a sentence are the *minimum* of the corresponding features of its tokens.
    - ``xmax``, ``ymax`` of a sentence are the *maximum* of the corresponding features of its tokens.
    - ``position`` is the order of the sentence in the paper.

The features are extracted using the script [prepare_data.py](../prepare_data.py).

I chose SciBERT because it was trained on the same domain as the data of this challenge. I limit the sequence length
to ``128`` because of the [data analysis](../analyze_data.ipynb) results that most of the sentences have their lengths
below this threshold.

Both experiments are performed on the *same* data split with the train/dev/test ratio of 8:1:1. I configured to train
two models with 1 epoch, but manually stopped training after ~7.000 iterations due to no further improvement.

Using extended features (experiment 2) results in higher macro F1 on the test set (~**69**%). The experiment details,
model hyperparameters, training progress and *detailed results* can be seen in the corresponding notebooks (the Test
Results section).

The best model in experiment 2 can be found in folder [model/Uizard_211020_17-13-10](model/Uizard_211020_17-13-10).
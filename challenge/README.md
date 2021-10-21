# Challenge: Experiment Details

Due to time limit, I can only run two experiments:

- [Experiment1.ipynb](Experiment1.ipynb): experiment with the sentence classifier that uses *only* text as input.
- [Experiment2.ipynb](Experiment2.ipynb): experiment with the sentence classifier that uses text and additional features
  as inputs: ``xmin``, ``xmax``, ``ymin``, ``ymax`` and ``position``.
    - ``xmin``, ``ymin`` of a sentence are the *minimum* of the corresponding features of its tokens.
    - ``xmax``, ``ymax`` of a sentence are the *maximum* of the corresponding features of its tokens.
    - ``position`` is the order of the sentence in the paper.

The features are extracted using the script [prepare_data.py](../prepare_data.py).

Both experiments are performed on the *same* data split. Using extended features (experiment 2) results in higher macro
F1 on the test set (~**69**%). The experiment details, model hyperparameters and training progress can be seen in the
corresponding notebooks.

I configured to train two models with 1 epoch, but manually stopped training after ~7.000 iterations due to no
improvement.

The best model in experiment 2 can be found in folder [Uizard_211020_17-13-10](Uizard_211020_17-13-10).
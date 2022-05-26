# task12
Evaluating Rik's models


# Addendum 2022-05-26T09:16:29

My first results proved to be tricky.
* Simpletransformers have to be trained with `model.train_model()` and not `model.train()`
* When evaluating, pass a list of strings, not a string directly.
* I still have issues when evaluating: tokenizer multiprocessing warnings start popping up.

Question for Nikola:

We have two classifiers (cause and effect). We also have two plausible alternatives (answer 1 and answer 2). This would make the output a tensor of 2x2x2:

| answer | model cause  | model effect |
|:------:|:------------:|:------------:|
|   A1   | [0.1, 0.11]  |  [0.5, 0.9]  |
|   A2   | [0.2, 0.08]  |  [0.8, 0.6]  |

Model outputs are in the order [False, True]. What is to be done in this case? A simple trick would be to take the highest value. In this case this would mean answer 1 is correct and the answer is the effect of the premise. Should this be done differently?
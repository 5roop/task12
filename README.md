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

# Addendum 2022-05-26T10:19:37

Question has been resolved. We train AND eval models separately. Meaning that the output matrix is no longer 2x2x2, but only 1x2x2:

| answer | model        |
|:------:|:------------:|
|   A1   | [0.1, 0.11]  |
|   A2   | [0.2, 0.08]  |

In this case I would search for maximal value. In the current example it would be first value for A2. The first value is False, and so this means that the second answer is false and the first answer is more plausible.


Another fun fact:
When instantiating Bertic models, set model type as bert. The warnings will be popping up to change model type to electra, but in this case the model will never finish training. So keep it 'bert'.

# Addendum 2022-05-28T15:50:44
So it turned out that I can only train 'Bertić' with type 'bert', else it trains forever.
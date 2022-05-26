democonfig = {
    "lang": "EN",
    "NUM_EPOCH": 5,
    "model_name": "xlm-roberta-base",
    "model_type": "xlmroberta",
}

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

col_rename_dict = {
    "premise": "premise-EN",
    "answer1": "answer1-EN",
    "answer2": "answer2-EN",
}

test = pd.read_excel("COPA-MT-MK.BG.TR.MT.IS.HR.xlsx", sheet_name="test").dropna(axis=1, how="all").rename(columns=col_rename_dict)
dev = pd.read_excel("COPA-MT-MK.BG.TR.MT.IS.HR.xlsx", sheet_name="dev").dropna(axis=1, how="all").rename(columns=col_rename_dict)
train = pd.read_excel("COPA-MT-MK.BG.TR.MT.IS.HR.xlsx", sheet_name="train").dropna(axis=1, how="all").rename(columns=col_rename_dict)

def filter_dataframe(df: pd.DataFrame, lang: str ="EN", asksfor="cause", sep_token = None) -> pd.DataFrame:
    lang = lang.upper()
    assert lang in "EN MK BG TR MT IS HR.MT".split(), f"Input language {lang} is not supported."
    assert asksfor in {"cause", "effect"}, "Parameter asksfor can be either 'cause' or 'effect'"
    assert sep_token, "Missing sep_token!"

    df = df[df['asks-for'] == asksfor]
    premise_col = df[f"premise-{lang}"].values
    answer1_col = df[f"answer1-{lang}"].values
    answer2_col = df[f"answer2-{lang}"].values
    gold_col = df["gold"].values.astype(int)

    labels = []
    text = []

    for premise, ans1, ans2, gold in zip(premise_col, answer1_col, answer2_col, gold_col):
        t = premise + sep_token + ans1
        l = 1 if gold == 1 else 0

        text.append(t)
        labels.append(l)

        t = premise + sep_token + ans2
        l = 1 if gold == 2 else 0

        text.append(t)
        labels.append(l)
    return pd.DataFrame(data={
        "text": text,
        "labels": labels
    })


def COPA(**config):

    test["y_pred"] = 0
    dev["y_pred"] = 0
    import numpy as np
    from tqdm.auto import tqdm
    NUM_EPOCH = config.get("NUM_EPOCH")
    lang = config.get("lang")
    model_name = config.get("model_name")
    model_type = config.get("model_type")
    from simpletransformers.classification import ClassificationModel

    model_args = {
        "num_train_epochs": NUM_EPOCH,
        # "learning_rate": 4e-5,
        "overwrite_output_dir": True,
        # "train_batch_size": 8,
        "no_save": True,
        "no_cache": True,
        "overwrite_output_dir": True,
        "save_steps": -1,
        "max_seq_length": 512,
        "silent": ~True,
    }

    # Effect:

    model_effect = ClassificationModel(
        model_type, model_name, num_labels=2, use_cuda=True, args=model_args
    )
    sep_token = model_effect.tokenizer.sep_token
    train_effect = filter_dataframe(
        train, lang=lang, asksfor="effect", sep_token=sep_token
    )
    dev_effect = filter_dataframe(dev, lang=lang, asksfor="effect", sep_token=sep_token)
    test_effect = filter_dataframe(
        test, lang=lang, asksfor="effect", sep_token=sep_token
    )
    model_effect.train_model(
        train_effect, output_dir="models", verbose=True, show_running_loss=True
    )

    def classify(logits):
        import numpy as np

        index = np.unravel_index(np.argmax(logits, axis=None), logits.shape)
        if index == (0, 0) or index == (1, 1):
            # A1 false, A2 true
            return 2
        else:
            # A1 true, A2 false
            return 1

    results = []
    for i in tqdm(range(0, test_effect.shape[0], 2)):
        texts = test_effect.iloc[i : i + 2, 0].values.tolist()
        predictions, logits = model_effect.predict(texts)
        result = classify(logits)
        results.append(result)
    test.loc[test["asks-for"] == "effect", "y_pred"] = results

    results = []
    for i in tqdm(range(0, dev_effect.shape[0], 2)):
        texts = dev_effect.iloc[i : i + 2, 0].values.tolist()
        predictions, logits = model_effect.predict(texts)
        result = classify(logits)
        results.append(result)
    dev.loc[dev["asks-for"] == "effect", "y_pred"] = results

    del model_effect
    from torch import cuda
    cuda.empty_cache()


    # Cause:


    model_cause = ClassificationModel(
        model_type, model_name, num_labels=2, use_cuda=True, args=model_args
    )

    train_cause = filter_dataframe(
        train, lang=lang, asksfor="cause", sep_token=sep_token
    )
    test_cause = filter_dataframe(test, lang=lang, asksfor="cause", sep_token=sep_token)
    dev_cause = filter_dataframe(dev, lang=lang, asksfor="cause", sep_token=sep_token)
    model_cause.train_model(
        train_cause,
        output_dir="models",
        verbose=False,
        show_running_loss=True,
        wandb_project="LM_EVAL",
        wandb_kwargs={"entity": "wandb"},
    )



    results = []
    for i in tqdm(range(0, test_cause.shape[0], 2)):
        texts = test_cause.iloc[i : i + 2, 0].values.tolist()
        predictions, logits = model_cause.predict(texts)
        result = classify(logits)
        results.append(result)
    test.loc[test["asks-for"] == "cause", "y_pred"] = results


    results = []
    for i in tqdm(range(0, dev_cause.shape[0], 2)):
        texts = dev_cause.iloc[i : i + 2, 0].values.tolist()
        predictions, logits = model_cause.predict(texts)
        result = classify(logits)
        results.append(result)
    dev.loc[dev["asks-for"] == "cause", "y_pred"] = results
    from torch import cuda

    cuda.empty_cache()

    from sklearn.metrics import accuracy_score
    return {
        "test_accuracy": accuracy_score(test.gold, test.y_pred),
        "dev_accuracy": accuracy_score(dev.gold, dev.y_pred),
        "config": config,
        # "dev_df": dev,
        # "test_df": test,
    }


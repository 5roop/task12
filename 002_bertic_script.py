from utils import *


config = democonfig
config["lang"] = "HR.HT"
config["model_name"] = "classla/bcms-bertic"
config["model_type"] = "bert"
config["NUM_EPOCH"] = 30
results = COPA(**config)


with open("002_results_bertic.jsonl", "a") as f:
    f.write(str(results)+'\n')
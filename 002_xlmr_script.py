from utils import *

for lang in "EN BG MT MK IS TR".split():
    config = democonfig
    config["lang"] = lang
    results = COPA(**config)
    
    with open("002_results.jsonl", "a") as f:
        f.write(str(results)+'\n')
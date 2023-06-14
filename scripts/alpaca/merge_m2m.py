import json
import glob

files = glob.glob("./data/*.json")
files = sorted(files, key=lambda x: int(x.replace(".json", "").split("_")[-1]))

items = []
for file in files:
    item = ""
    with open(file, "r") as f:
        item = json.dumps(json.load(f), ensure_ascii=False)

        with open("../../data/cc-by-sa-3.0/databricks-dolly-15k-translated.jsonl", "a") as output:
            output.write(item + "\n")
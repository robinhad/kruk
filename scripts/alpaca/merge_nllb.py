import json
import glob

files = glob.glob("./data/*.json")
files = sorted(files, key=lambda x: int(x.replace(".json", "").split("_")[-1]))

items = []
for file in files:
    with open(file, "r") as f:
        items.append(json.load(f))

with open("../../data/cc-by-nc/alpaca_data_translated.json", "w") as f:
    json.dump(items, f, indent=4, ensure_ascii=False)
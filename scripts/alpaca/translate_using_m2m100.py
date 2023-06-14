
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from torch import no_grad
import time
import glob
import json
from tqdm import tqdm

# Note: run this script inside the scripts/alpaca folder

#model_name = "facebook/m2m100-12B-avg-5-ckpt"
model_name = "facebook/m2m100_1.2B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, src_lang="en"
)

device = "cuda"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


instructions = []
with open('../../data/cc-by-sa-3.0/databricks-dolly-15k.jsonl') as f:
    for line in f:
        instructions.append(json.loads(line))


def translate(sentence):
    with no_grad():
        inputs = tokenizer(sentence, return_tensors="pt", padding=True)
        translated_tokens = model.generate(
            **inputs.to(device), forced_bos_token_id=tokenizer.lang_code_to_id["uk"], max_length=1024
        )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)#[0]


def translate_item(item):
    new_item = {}
    # for every key in the item
    for key, value in item.items():
        if len(value.strip()) == 0:
            new_item[key] = value
        elif key == "category":
            new_item[key] = value
        else:
            # separate the text into paragraphs
            translated = []
            for part in value.split("\n"):
                if len(part.strip()) == 0:
                    translated.append(part)
                else:
                    # separate the paragraphs into sentences
                    sentences = nltk.sent_tokenize(part)
                    sentence_parts = translate(sentences)
                    translated.append(" ".join(sentence_parts))
            new_item[key] = "\n".join(translated)
    return new_item

def translate_and_save(instruct):
    idx, item = instruct
    translated = translate_item(item)
    with open(f"./data/dolly_data_translated_{idx}.json", "w") as f:
        json.dump(translated, f)

if __name__ == '__main__':
    start = time.perf_counter()
    translated = []
    jsons = [int(file.replace(".json", "").split("_")[-1]) for file in glob.glob("./data/*.json")]
    total_instructions = len(instructions)
    # continue from the last json item
    if len(jsons) == 0:
        last_json_id = 0
    else:
        last_json_id = max(jsons)

    remaining = total_instructions - last_json_id

    for i in tqdm( enumerate(instructions[last_json_id:], start=last_json_id), total=remaining):
        translate_and_save(i)

    end = time.perf_counter() - start
    print(f"Finished in {end} seconds")
    print(f"Item processing time is {end/remaining} seconds")
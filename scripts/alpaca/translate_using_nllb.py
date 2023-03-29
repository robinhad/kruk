
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from torch import no_grad
import time
import glob
import json

# Note: run this script inside the scripts/alpaca folder

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-3.3B", src_lang="eng_Latn"
)

device = "cuda"
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B").to(device)

instructions = json.load(open('../../data/cc-by-nc/alpaca_data.json'))


def translate(sentence):
    with no_grad():
        inputs = tokenizer(sentence, return_tensors="pt")

        translated_tokens = model.generate(

            **inputs.to(device), forced_bos_token_id=tokenizer.lang_code_to_id["ukr_Cyrl"], max_length=512

        )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def translate_item(item):
    new_item = {}
    # for every key in the item
    for key, value in item.items():
        if len(value.strip()) == 0:
            new_item[key] = value
        else:
            # separate the text into paragraphs
            parts = []
            translated = []
            for part in value.split("\n"):
                parts.append(part)
                if len(part.strip()) == 0:
                    translated.append(part)
                else:
                    # separate the paragraphs into sentences
                    sentences = nltk.sent_tokenize(part)
                    sentence_parts = []
                    for sentence in sentences:
                        sentence_parts.append(translate(sentence))

                    translated.append(" ".join(sentence_parts))            
            new_item[key] = "\n".join(translated)
    return new_item

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

    mean_times = []
    for idx, instruction in enumerate(instructions[last_json_id:], start=last_json_id):
        item_start = time.perf_counter()
        with open(f"./data/alpaca_data_translated_{idx}.json", "w") as f:
            json.dump(translate_item(instruction), f)
        item_end = time.perf_counter() - item_start
        # calculate time left based on average of 30
        if len(mean_times) < 30:
            mean_times.append(item_end)
        else:
            mean_times.pop(0)
            mean_times.append(item_end)
        print(f"Item {idx + 1}/{total_instructions} finished in {item_end:.2f} seconds. {(sum(mean_times)/len(mean_times)*(total_instructions-last_json_id)/60/60):.2f} hours left.", end='\r')
    end = time.perf_counter() - start
    print(f"Finished in {end} seconds")
    print(f"Item processing time is {end/idx} seconds")
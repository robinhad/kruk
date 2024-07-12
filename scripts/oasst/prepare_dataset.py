# %%
import datasets


from datasets import load_dataset
ds = load_dataset("OpenAssistant/oasst2")
train = ds['train']      # len(train)=128575 (95%)
val = ds['validation'] 
# %%
print(train[0])
print(len(train))
# %%
df = ds["train"].to_pandas() 
df

#%%

def convert_df(df):
    dialogues = []
    for parent_row in df[(df["parent_id"].isna())].iterrows():
        tree_df = df[df["message_tree_id"] == parent_row[1]["message_tree_id"]]
        messages = [row[1]["text"] for row in tree_df.iterrows()]
        start = "user"
        for idx, message in enumerate(messages):
            messages[idx] = {"role": start, "content": message}
            start = "assistant" if start == "user" else "user"
        dialogues.append(messages)
    return dialogues

dialogues_all = convert_df(df[df["lang"] != "uk-UA"])
dialogues_uk = convert_df(df[df["lang"] == "uk-UA"])
# %%
dialogues_all[0]

len(dialogues_all)


final_dataset = []
for dialogue in dialogues_all:
    final_dataset.append({"conversations": dialogue})


for dialogue in dialogues_uk:
    final_dataset.append({"conversations": dialogue})
    
# %%

phrases = ["переклади", "переклади з англійської", "давай по-нашому", "перекладеш", "переклади, будь ласка, з англійської", "переклади українською", "переклади по-українськи", "переклади на українську", "що тут пише"]
suffix_sep = ["\n", "\n\n", "\n=\n", "\n========\n"]
prefix_sep = [" "] + suffix_sep

from random import choice
def augment_translation(text):
    select_prefix = choice([True, False])
    augment_text = choice([0, 1, 2, 3, 4])
    capitalize = choice([True, False])
    prefix = ""
    suffix = ""
    if augment_text == 0:
        pass
    elif augment_text == 1:
        text = "\"" + text + "\""
    elif augment_text == 2:
        text = "'" + text + "'"
    elif augment_text == 3:
        text = "<< " + text + " >>"
    
    # text = text.capitalize()
    selected_phrase = choice(phrases)
    if capitalize:
        selected_phrase = selected_phrase.capitalize()
    endings = [".", "!", "?", ":", ""]
    selected_phrase += choice(endings)

    if select_prefix:
        prefix = selected_phrase
        prefix = prefix + choice(prefix_sep)
    else:
        suffix = choice(suffix_sep)
        suffix += selected_phrase
    return prefix + text + suffix

print(augment_translation("привіт"))

# %%
printer_uk = load_dataset("lang-uk/multi30k-extended-17k", data_files="multi30k-extended-17k.jsonlines")
printer_uk
# %%

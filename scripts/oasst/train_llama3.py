#%%
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
import transformers
from tqdm import tqdm
# Model
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "UAlpacaOrpoLlama-3-8B"


# Set torch dtype and attention implementation
if False:#torch.cuda.get_device_capability()[0] >= 8:
    #!pip install -qqq flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.bfloat16
    #attn_implementation = "eager"

#%%
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True, add_pad_token=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto",
    torch_dtype=torch_dtype,
    #attn_implementation=attn_implementation
)
model = get_peft_model(model, peft_config)
# model = PeftModel.from_pretrained(base_model, new_model)
#model = model.merge_and_unload()
#model, tokenizer = setup_chat_format(model, tokenizer)
#model = prepare_model_for_kbit_training(model)

#%%
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
    N = len(df[(df["parent_id"].isna())])
    for parent_row in tqdm(df[(df["parent_id"].isna())].iterrows(), desc="Converting trees to dialogues", total=N):
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

#%%
from datasets import load_dataset
from tqdm import tqdm

aya_dataset = load_dataset("CohereForAI/aya_collection_language_split", "ukrainian")
aya_df = aya_dataset["train"].to_pandas()

def aya_convert_to_task(row):
    messages = []
    messages.append({"role": "user", "content": row["inputs"]})
    messages.append({"role": "assistant", "content": row["targets"]})
    return messages

def ayaconvert_df(df):
    df["text"] = df.apply(lambda row: aya_convert_to_task(row), axis=1)
    return df["text"].tolist()

aya_dialogues = ayaconvert_df(aya_df)

#%%
from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
# %%
data = tokenizer.apply_chat_template(aya_dialogues, tokenize=False) + tokenizer.apply_chat_template(dialogues_all, tokenize=False) + tokenizer.apply_chat_template(dialogues_uk, tokenize=False)

import datasets

data = [{"text": text} for text in data]
print(data[0])
dataset = datasets.Dataset.from_list(data)
#%%
dataset = dataset.map(lambda x: tokenizer(x["text"], max_length=512, truncation=True), 
    num_proc= os.cpu_count())

# %%
dataset
# %%
#dataset = dataset.select(range(1000))
#%%
#%%
import transformers
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    #eval_dataset=dataset["test"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=8e-6,
        fp16=True,
        save_total_limit=3,
        #gradient_checkpointing=True,
        #push_to_hub=True,
        #hub_token=hub_token,
        save_strategy="steps",
        save_steps=100,
        group_by_length=True,
        lr_scheduler_type="linear",
        #max_length=1024,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        #evaluation_strategy="steps",
        #eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        report_to="tensorboard",
        output_dir=new_model,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
)
#%%

trainer.train()
trainer.save_model(new_model)
# %%
# Flush memory
del trainer, model
gc.collect()
gc.collect()
torch.cuda.empty_cache()
#%%
# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
fp16_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
#fp16_model, tokenizer = setup_chat_format(fp16_model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(fp16_model, new_model)
model = model.merge_and_unload()
# %%
tokenizer.encode("Hello, how are you?")
#%%
from transformers import StopStringCriteria, StoppingCriteriaList

criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, stop_strings=["<|im_end|>"])])

question = tokenizer.apply_chat_template([{"role": "user", "content": "Дай рецепт борщу українською"}], return_tensors="pt").to("cuda")

print(tokenizer.batch_decode(model.generate(question, do_sample=True, max_length=600, stopping_criteria=criteria))[0])

#%%
dataset = load_dataset("json", data_files="../../data/cc-by-nc/alpaca_data_translated.json", split="all")

# %%

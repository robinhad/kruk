import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM 
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model


MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 512  # 1024 accounts for about 99.5% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

model_name = "openlm-research/open_llama_7b"

model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(
    model_name, add_eos_token=True
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files="../../data/cc-by-sa-3.0/databricks-dolly-15k-translated.jsonl")

# def generate_prompt(data_point):
#     if data_point["input"]:
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# ### Instruction:
# {data_point["instruction"]}
# ### Input:
# {data_point["input"]}
# ### Response:
# {data_point["output"]}"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# {data_point["instruction"]}
# ### Response:
# {data_point["output"]}"""

# TODO: take a look at translation
def generate_prompt(data_point):
    if data_point["context"]:
        return f"""Унизу надається інструкція, яка описує завдання разом із вхідними даними, які надають додатковий контекст. Напиши відповідь, яка правильно доповнює запит.
### Інструкція:
{data_point["instruction"]}
### Вхідні дані:
{data_point["context"]}
### Відповідь:
{data_point["response"]}"""
    else:
        return f"""Унизу надається інструкція, яка описує завдання. Напиши відповідь, яка правильно доповнює запит.
### Інструкція:
{data_point["instruction"]}
### Відповідь:
{data_point["response"]}"""



def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        #padding=True#"max_length",
    )
    return result


data = data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

original_size = len(data["train"])
print(f"Source data size: {original_size}")
#hub_token = os.environ["HUB_TOKEN"]
#print(f"Hub token: {hub_token}")

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        output_dir="ualpaca-7b-llama",
        save_total_limit=3,
        #push_to_hub=True,
        #hub_token=hub_token,
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=1),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)


model.save_pretrained("ualpaca-7b-llama")
#trainer.push_to_hub()



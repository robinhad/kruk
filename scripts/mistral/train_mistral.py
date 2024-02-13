import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from typing import Any, Dict, List, Mapping, Union
import torch
import transformers


MICRO_BATCH_SIZE = 8
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1
LEARNING_RATE = 2e-5
CUTOFF_LEN = 512
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
OUTPUT_MODEL_NAME = "mistral-finetune"

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "mistralai/Mistral-7B-v0.1"


# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Preparing tokenized version according to the comment
# https://github.com/huggingface/transformers/issues/22794#issuecomment-1601482558
class EosCollator(transformers.DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": transformers._torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                if self.tokenizer.padding_side == "right":
                    idx = torch.argmax((labels == -100).to(dtype=torch.int), dim=-1)
                    labels[torch.arange(idx.shape[0]), idx] = self.tokenizer.eos_token_id 
                    labels[:, 0] = self.tokenizer.bos_token_id 
                else:
                    labels[:, -1] = self.tokenizer.eos_token_id

            batch["labels"] = labels
        return batch


def train_on_data(data):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        use_fast=False,
        padding_side="left",
        add_eos_token=True,
        # add_bos_token=False,
    )
    tokenizer.save_pretrained(OUTPUT_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # here or line before?

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token_id = model.config.pad_token_id


    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    data = data.map(lambda x: tokenizer(x["text"]), num_proc=40)
    data = data.filter(lambda x: len(x["input_ids"]) <= CUTOFF_LEN)


    print("Dataset size after cutoff:", len(data))
    print("Max len:", max([len(x["input_ids"]) for x in data]))

    total_steps = int((len(data) // (MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS)
    warmup_steps = min(100, int(total_steps * 0.1))
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")


    run_name = (
        f"{OUTPUT_MODEL_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    output_dir = f"exps/{OUTPUT_MODEL_NAME}"


    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=5,
            output_dir=output_dir,
            save_total_limit=2,
            save_strategy="steps",
            save_steps=20,
            report_to="tensorboard",
            run_name=run_name,
            warmup_steps=warmup_steps,
        ),
        data_collator=EosCollator(
            tokenizer,
            pad_to_multiple_of=8,
            mlm=False,
        ),
    )
    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(output_dir)


def main():
    data = load_dataset("dataset.json", split="train")
    train_on_data(data)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

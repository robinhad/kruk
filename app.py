from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import gradio as gr
from torch.cuda import is_available

if is_available():
    options = dict(
        load_in_8bit=True,
        device_map="auto",
    )
else:
    options = {}

tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_7b")
model = LlamaForCausalLM.from_pretrained(
    "openlm-research/open_llama_7b",
    **options
)
model = PeftModel.from_pretrained(model, "robinhad/open_llama_7b_uk")


def generate_prompt(instruction, input=None, output=""):
    if input:
        return f"""Унизу надається інструкція, яка описує завдання разом із вхідними даними, які надають додатковий контекст. Напиши відповідь, яка правильно доповнює запит.
### Інструкція:
{instruction}
### Вхідні дані:
{input}
### Відповідь:
{output}"""
    else:
        return f"""Унизу надається інструкція, яка описує завдання. Напиши відповідь, яка правильно доповнює запит.
### Інструкція:
{instruction}
### Відповідь:
{output}"""


generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=4,
)

def evaluate(instruction, input=None):
    if input.strip() == "":
        input = None
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if is_available():
        input_ids = input_ids.cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=64
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s, skip_special_tokens=True)
        print("============")
        print(output)
        return output.split("### Відповідь:")[1].strip()


gr.Interface(
    evaluate,
    [
        gr.inputs.Textbox(lines=5, label="Інструкція"),
        gr.inputs.Textbox(lines=5, label="Вхідні дані (необов'язково)"),
    ],
    gr.outputs.Textbox(label="Відповідь"),
    title="Kruk",
    description="Open Llama is a Ukrainian language model trained on the machine-translated Dolly dataset.",
    examples=[
        [
            "Яка найвища гора в Україні?",
            "",
        ],
        [
            "Розкажи історію про Івасика-Телесика.",
            "",
        ],
        [
            "Яка з цих гір не знаходиться у Європі?",
            "Говерла, Монблан, Гран-Парадізо, Еверест"
        ],
        [
            "Чому качки жовтоногі?",
            "",
        ],
        [
            "Чому у качки жовті ноги?",
            "",
        ],
    ]
).launch()
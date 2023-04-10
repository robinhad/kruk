# How to generate dataset and train new model

1. `cd` into this folder.
2. Run [translate_using_nllb.py](translate_using_nllb.py) to translate dataset from Ukrainian To English using NLLB model.
3. Run [merge_nllb.py](merge_nllb.py) to merge translated scripts into single file [data/cc-by-nc/alpaca_data_translated.json](../../data/cc-by-nc/alpaca_data_translated.json).
4. Run [train_ualpaca.py](train_ualpaca.py) to start training (about 8 hours on 3090).

# Merge checkpoint to use with llama.cpp

1. `cd` into project root.
2. `python scripts/alpaca/merge_checkpoint.py --base_model "decapoda-research/llama-7b-hf" --lora_model "robinhad/ualpaca-7b-llama" --output_dir merged`

# Run llama.cpp with the same parameters
1. Install `llama.cpp` from https://github.com/ggerganov/llama.cpp#usage
2. Convert the 7B model to ggml FP16 format
`python3 convert-pth-to-ggml.py models/7B/ 1`
3. (optional) quantize the model to 4-bits (using method 2 = q4_0)
`./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin 2`
2. Run (for 4-bit model)
```bash
./main -m ./models/merged/ggml-model-q4_0.bin -n 256 --temp 0.2 --top_p 0.75 --top_k 50 --repeat_penalty 1.0 --repeat_last_n 256 -b 4 -c 256 --color -s 1680391377 -p "Унизу надається інструкція, яка описує завдання. Напиши відповідь, яка правильно доповнює запит.

### Інструкція:
Чому у качки жовті ноги?

### Відповідь:"
```
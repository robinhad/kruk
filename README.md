# Kruk 𓄿
🇺🇦 Ukrainian instruction-tuned language models and datasets

This repository contains training scripts and examples for Ukrainian instruction-tuned language models and datasets (chatGPT-like).

[![Ukrainian NLP community](https://img.shields.io/badge/Ukrainian%20NLP%20community-Telegram-blue)](https://t.me/nlp_uk)

# Models 🤖
- [UAlpaca model (HuggingFace) (CC-BY-NC 4.0)](https://huggingface.co/robinhad/ualpaca-7b-llama) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robinhad/kruk/blob/main/notebooks/ualpaca-lora.ipynb)

Example:

```
>> Яка найвища гора України?
Найвища гора України - Говерла (2061 м).

>> Як звали батька Тараса Григоровича Шевченка?
Батько Тараса Григоровича Шевченка - Григорій Шевченко.

>> Як можна заробити нелегально швидко гроші?
Один з способів заробити нелегально швидко гроші - це торгівля на чорному ринку. Торгівля на чорному ринку - це продаж товарів або послуг, які не законно продаються в країні, де вони продаються. Торгівля на чорному ринку може бути небезпечною, оскільки вона може призвести до конфліктів з правоохоронними органами.

>> Чому у качки жовті ноги?
У качки жовті ноги, щоб вони могли швидко пересуватися по воді.

>> Чому качки жовтоногі?
Качки жовтоногі - це вид птахів, який відрізняється від інших качок своїм яскраво-жовтим забарвленням. Його яскраве забарвлення, ймовірно, служить для захисту від хижаків.
```

# Data 📚
- [Alpaca Dataset](data/cc-by-nc/alpaca_data.json) - under CC-BY-NC 4.0 License. Used for generating Ukrainian transcripts.
- [UAlpaca Dataset](data/cc-by-nc/alpaca_data_translated.json) - under CC-BY-NC 4.0 License. Was generated using NLLB-3.3B model with [translation script](scripts/alpaca/translate_using_nllb.py)

# How to train 🏋️
- [UAlpaca guide](scripts/alpaca/STEPS.md)

# Support ❤️
If you like my work, please support ❤️ -> [https://send.monobank.ua/jar/48iHq4xAXm](https://send.monobank.ua/jar/48iHq4xAXm)  
You're welcome to join Ukrainian NLP community: [Telegram https://t.me/nlp_uk](https://t.me/nlp_uk)


# Attribution 🤝
- This repository: [@robinhad](https://github.com/robinhad)
- Alpaca scripts: [teelinsan/camoscio](https://github.com/teelinsan/camoscio/)
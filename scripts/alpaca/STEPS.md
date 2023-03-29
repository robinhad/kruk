# How to generate dataset and train new model

1. `cd` into this folder.
2. Run [translate_using_nllb.py](translate_using_nllb.py) to translate dataset from Ukrainian To English using NLLB model.
3. Run [merge_nllb.py](merge_nllb.py) to merge translated scripts into single file [data/cc-by-nc/alpaca_data_translated.json](../../data/cc-by-nc/alpaca_data_translated.json).
4. Run [train_ualpaca.py](train_ualpaca.py) to start training (about 8 hours on 3090).